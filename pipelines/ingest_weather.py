"""
pipelines/ingest_weather.py

Fetch historical weather from Open-Meteo and backfill weather_id into
player_match_stats.

Stadium coordinates are resolved in three tiers (cheapest first):
  1. The stadiums table — once a venue is geocoded its coords are stored, so
     subsequent runs skip the work entirely.
  2. A curated coordinate table for well-known venues (_STADIUM_COORDS_RAW) —
     high-confidence, no network call.
  3. geopy / Nominatim online geocoding for anything else, written back into the
     stadiums table so it becomes tier 1 next time.

Weather itself comes from Open-Meteo's free historical archive API (no key),
one daily record per match keyed by the stadium's lat/lng and the match date.
"""

import logging
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests

from core.utils import norm_name
from load.postgres import connect, upsert_weather
from config.settings import DB_DSN, OPEN_METEO_URL

logger = logging.getLogger(__name__)

MAX_CONCURRENT  = 4
REQUEST_TIMEOUT = 20
MAX_RETRIES     = 4
RETRY_BASE_SECS = 2.0

# Nominatim usage policy: max 1 request/second, descriptive user-agent required.
GEOCODE_MIN_INTERVAL = 1.1
GEOCODE_USER_AGENT   = "soccer_analytics_eda/1.0"

_DAILY_VARS = (
    "temperature_2m_max,temperature_2m_min,"
    "precipitation_sum,windspeed_10m_max,"
    "relative_humidity_2m_max"
)

# ---------------------------------------------------------------------------
# Curated stadium coordinates (high-confidence seed; the rest are geocoded)
# ---------------------------------------------------------------------------
_STADIUM_COORDS_RAW = {
    "Camp Nou":                          (41.3809,   2.1228),
    "Santiago Bernabeu":                 (40.4531,  -3.6883),
    "Estadio Wanda Metropolitano":       (40.4361,  -3.5995),
    "Wanda Metropolitano":               (40.4361,  -3.5995),
    "Estádio Cívitas Metropolitano":     (40.4361,  -3.5995),
    "Estadio Ramon Sanchez-Pizjuan":     (37.3841,  -5.9706),
    "Estadio Ramón Sánchez Pizjuán":     (37.3841,  -5.9706),
    "Estadio de Mestalla":               (39.4750,  -0.3583),
    "Mestalla":                          (39.4750,  -0.3583),
    "Estadio San Mames":                 (43.2642,  -2.9494),
    "San Mames":                         (43.2642,  -2.9494),
    "Estadio de la Ceramica":            (39.9444,  -0.1028),
    "Estadio El Madrigal":               (39.9444,  -0.1028),
    "Estadio de Vallecas":               (40.3920,  -3.6600),
    "Estadio de Balaidos":               (42.2117,  -8.7397),
    "Abanca-Balaídos":                   (42.2117,  -8.7397),
    "Estadio Municipal de Ipurua":       (43.1864,  -2.4714),
    "Estadio de la Rosaleda":            (36.7167,  -4.4500),
    "Estadio La Rosaleda":               (36.7167,  -4.4500),
    "Estadio Nuevo Los Carmenes":        (37.1506,  -3.5986),
    "Estadio de Gran Canaria":           (28.1000, -15.4361),
    "Estadio El Molinon":                (43.5314,  -5.6361),
    "Estadio Municipal El Molinón":      (43.5314,  -5.6361),
    "Estadio Municipal de Mendizorroza": (42.8494,  -2.6819),
    "Estadio de Mendizorroza":           (42.8494,  -2.6819),
    "Power Horse Stadium":               (36.8417,  -2.4556),
    "Estadio Municipal de El Alcoraz":   (42.1361,  -0.4111),
    "Estadio El Alcoraz":                (42.1361,  -0.4111),
    "Estadio de Anoeta":                 (43.3014,  -1.9736),
    "Reale Arena":                       (43.3014,  -1.9736),
    "Estadio Municipal de Butarque":     (40.3517,  -3.7914),
    "Estadio Nuevo Mirandilla":          (36.5064,  -6.2722),
    "Estadio de los Juegos Mediterraneos": (36.8417, -2.4556),
    "Estadio Municipal de Montilivi":    (41.9833,   2.8167),
    "Estadi Municipal de Montilivi":     (41.9833,   2.8167),
    "Estadio RCDE":                      (41.3473,   2.0758),
    "Estadio Benito Villamarin":         (37.3567,  -5.9814),
    "Coliseum Alfonso Pérez":            (40.3256,  -3.7143),
    "Estadi Mallorca Son Moix":          (39.5899,   2.6303),
    "Estadio Abanca-Riazor":             (43.3687,  -8.4173),
    "Estadio Alfredo Di Stéfano":        (40.4762,  -3.6163),
    "Estadio Ciudad de Valencia":        (39.4933,  -0.3642),
    "Estadio El Sadar":                  (42.7963,  -1.6373),
    "Estadio Manuel Martínez Valero":    (38.2669,  -0.6635),
    "Estadio Municipal José Zorrilla":   (41.6443,  -4.7612),
    "Estadio Nuevo Arcángel":            (37.8886,  -4.7896),
    "Estadio Vicente Calderón":          (40.4017,  -3.7206),
    "Luzhniki Stadium":                  (55.7317,  37.5600),
    "Stadion Luzhniki":                  (55.7317,  37.5600),
    "Saint Petersburg Stadium":          (59.9724,  30.2219),
    "Fisht Stadium":                     (43.4010,  39.9514),
    "Ekaterinburg Arena":                (56.8429,  60.5935),
    "Kazan Arena":                       (55.8483,  49.0675),
    "Ak Bars Arena":                     (55.8483,  49.0675),
    "Nizhny Novgorod Stadium":           (56.3379,  43.9633),
    "Stadion Nizhny Novgorod (Nizhniy Novgorod)": (56.3379, 43.9633),
    "Mordovia Arena":                    (54.1831,  45.1747),
    "Rostov Arena":                      (47.2289,  39.7158),
    "Volgograd Arena":                   (48.7074,  44.5534),
    "Cosmos Arena":                      (53.4133,  50.1725),
    "Solidarnost Arena":                 (53.4133,  50.1725),
    "Kaliningrad Stadium":               (54.7138,  20.5167),
    "Stadion Kaliningrad":               (54.7138,  20.5167),
    "Otkritie Bank Arena":               (55.8178,  37.4403),
    "Lusail Iconic Stadium":             (25.4333,  51.5000),
    "Al Bayt Stadium":                   (25.6572,  51.5150),
    "Ahmad Bin Ali Stadium":             (25.2477,  51.4041),
    "Education City Stadium":            (25.3117,  51.4230),
    "Al Thumama Stadium":                (25.2364,  51.5361),
    "Khalifa International Stadium":     (25.2632,  51.4500),
    "Stadium 974":                       (25.2735,  51.5497),
    "Al Janoub Stadium":                 (25.1270,  51.5000),
    "Anfield":                           (53.4308,  -2.9608),
    "Johan Cruijff Arena":               (52.3143,   4.9418),
    "Tottenham Hotspur Stadium":         (51.6042,  -0.0664),
    "Estadio da Luz":                    (38.7525,  -9.1842),
    "Signal Iduna Park":                 (51.4926,   7.4519),
    "Parc des Princes":                  (48.8414,   2.2530),
    "Allianz Arena":                     (48.2188,  11.6247),
    "Stamford Bridge":                   (51.4816,  -0.1910),
    "Old Trafford":                      (53.4631,  -2.2913),
    "Juventus Stadium":                  (45.1096,   7.6413),
    "San Siro":                          (45.4781,   9.1240),
    "Olimpiyskiy":                       (50.4339,  30.5214),
    "Estadio Jose Alvalade":             (38.7613,  -9.1603),
    "Jan Breydel Stadion":               (51.1944,   3.1600),
    "Estadio do Dragao":                 (41.1614,  -8.5839),
    # Venues Nominatim could not resolve (apostrophes / qualifiers / trailing
    # spaces) — added explicitly so all in-scope stadiums get weather.
    "Stadio Marc'Antonio Bentegodi":     (45.4353,  10.9686),  # Verona
    "Stade Yves Allainmat - Le Moustoir":(47.7486,  -3.3700),  # Lorient
    "Stade Auguste-Delaune II":          (49.2466,   4.0250),  # Reims
    "Stade Matmut Atlantique":           (44.8975,  -0.5614),  # Bordeaux
    "Boleyn Ground":                     (51.5320,   0.0392),  # West Ham (Upton Park)
    "Stadio Comunale Matusa":            (41.6360,  13.3508),  # Frosinone
    "Wohninvest Weserstadion":           (53.0664,   8.8378),  # Werder Bremen
    "Trainingszentrum RB Leipzig Platz 1": (51.2950, 12.4150), # RB Leipzig training ground
}

STADIUM_COORDS: dict[str, tuple] = {
    norm_name(k): v for k, v in _STADIUM_COORDS_RAW.items()
}


# ---------------------------------------------------------------------------
# Stadium name cleaning + geocoding
# ---------------------------------------------------------------------------
def _clean_name(name: str | None) -> str:
    """Tidy a raw stadium name for geocoding: trim whitespace/tabs, strip any
    Unicode replacement chars from upstream mojibake, and drop parentheticals."""
    if not name:
        return ""
    s = name.replace("�", " ")          # lost-byte replacement char
    s = re.sub(r"\(.*?\)", " ", s)            # "Stadion X (City)" -> "Stadion X"
    s = re.sub(r"\s+", " ", s).strip()
    return s


def _curated_coords(stadium_name: str | None) -> tuple | None:
    """Exact-then-substring match against the curated table."""
    if not stadium_name:
        return None
    nn = norm_name(stadium_name)
    if not nn:
        return None
    coords = STADIUM_COORDS.get(nn)
    if coords:
        return coords
    for key, val in STADIUM_COORDS.items():
        if nn in key or key in nn:
            logger.info("curated substring match: '%s' -> '%s'", stadium_name, key)
            return val
    return None


_geocoder = None
_geocode_lock = threading.Lock()
_last_geocode_ts = 0.0


def _get_geocoder():
    global _geocoder
    if _geocoder is None:
        from geopy.geocoders import Nominatim
        _geocoder = Nominatim(user_agent=GEOCODE_USER_AGENT, timeout=REQUEST_TIMEOUT)
    return _geocoder


def _geocode_online(stadium_name: str) -> tuple | None:
    """Resolve coords via Nominatim, honouring its 1 req/sec rate limit. Tries
    the cleaned name, then the name qualified with 'football stadium'."""
    global _last_geocode_ts
    clean = _clean_name(stadium_name)
    if not clean:
        return None

    geocoder = _get_geocoder()
    for query in (clean, f"{clean} football stadium"):
        with _geocode_lock:
            wait = GEOCODE_MIN_INTERVAL - (time.time() - _last_geocode_ts)
            if wait > 0:
                time.sleep(wait)
            try:
                loc = geocoder.geocode(query)
            except Exception as exc:                       # network / service error
                logger.warning("geocode error for '%s': %s", query, exc)
                loc = None
            finally:
                _last_geocode_ts = time.time()
        if loc is not None:
            return round(loc.latitude, 4), round(loc.longitude, 4)
    return None


def _geocode_stadiums(conn) -> int:
    """Fill stadiums.stadium_lat/lng for every venue still missing coordinates,
    using the curated table first and online geocoding as a fallback. Returns
    the number of venues newly geocoded."""
    with conn.cursor() as cur:
        cur.execute("""
            SELECT stadium_id, stadium_name
            FROM   stadiums
            WHERE  stadium_lat IS NULL OR stadium_lng IS NULL
            ORDER BY stadium_name
        """)
        pending = cur.fetchall()

    if not pending:
        logger.info("Stadium geocoding: all venues already have coordinates")
        return 0

    logger.info("Stadium geocoding: %d venue(s) to resolve", len(pending))
    resolved = 0
    unresolved: list[str] = []

    for stadium_id, name in pending:
        coords = _curated_coords(name) or _geocode_online(name)
        if coords is None:
            unresolved.append(name)
            continue
        with conn.cursor() as cur:
            cur.execute(
                "UPDATE stadiums SET stadium_lat = %s, stadium_lng = %s WHERE stadium_id = %s",
                (coords[0], coords[1], stadium_id),
            )
        conn.commit()
        resolved += 1

    logger.info("Stadium geocoding: %d resolved, %d unresolved", resolved, len(unresolved))
    if unresolved:
        logger.warning("Unresolved stadiums (no weather will be fetched): %s",
                       [n.strip() for n in unresolved])
    return resolved


# ---------------------------------------------------------------------------
# Open-Meteo fetch
# ---------------------------------------------------------------------------
def _derive_condition(temp_c, precip_mm, wind_kmh) -> str:
    """Map numeric weather values to a label matching the CHECK constraint on
    weather.weather_condition."""
    p = precip_mm or 0.0
    w = wind_kmh  or 0.0
    t = temp_c    if temp_c is not None else 15.0
    if p >= 5.0:  return "heavy_rain"
    if p >= 1.0:  return "rain"
    if w >= 50.0: return "windy"
    if t < 5.0:   return "cold"
    if t >= 30.0: return "hot"
    return "clear"


def _fetch_one(match_id, lat, lng, date_str, sem):
    """Fetch weather for one match with retry + exponential backoff."""
    params = {
        "latitude":   lat,
        "longitude":  lng,
        "start_date": date_str,
        "end_date":   date_str,
        "daily":      _DAILY_VARS,
        "timezone":   "UTC",
    }
    with sem:
        for attempt in range(1, MAX_RETRIES + 1):
            try:
                resp = requests.get(OPEN_METEO_URL, params=params, timeout=REQUEST_TIMEOUT)
                if resp.status_code == 429:
                    time.sleep(RETRY_BASE_SECS * (2 ** (attempt - 1)))
                    continue
                resp.raise_for_status()
                daily = resp.json().get("daily", {})

                t_max  = (daily.get("temperature_2m_max") or [None])[0]
                t_min  = (daily.get("temperature_2m_min") or [None])[0]
                temp   = (t_max + t_min) / 2 if (t_max is not None and t_min is not None) else None
                precip = (daily.get("precipitation_sum")        or [None])[0]
                wind   = (daily.get("windspeed_10m_max")        or [None])[0]
                humid  = (daily.get("relative_humidity_2m_max") or [None])[0]

                return match_id, {
                    "temperature_c":     temp,
                    "precipitation_mm":  precip,
                    "wind_speed_kmh":    wind,
                    "humidity_pct":      humid,
                    "weather_condition": _derive_condition(temp, precip, wind),
                }
            except requests.exceptions.RequestException as exc:
                if attempt == MAX_RETRIES:
                    logger.warning("Weather fetch failed for match %d after %d attempts: %s",
                                   match_id, MAX_RETRIES, exc)
                    return match_id, None
                time.sleep(RETRY_BASE_SECS * (2 ** (attempt - 1)))
    return match_id, None


def _backfill_weather_ids(conn):
    """Set weather_id on player_match_stats rows written before weather existed."""
    with conn.cursor() as cur:
        cur.execute("""
            UPDATE player_match_stats pms
            SET    weather_id = w.weather_id
            FROM   weather w
            WHERE  w.match_id = pms.match_id
              AND  pms.weather_id IS NULL
        """)
        updated = cur.rowcount
    conn.commit()
    if updated:
        logger.info("Backfilled weather_id on %d player_match_stats rows", updated)


def run(conn=None, max_concurrent=MAX_CONCURRENT) -> dict:
    """Geocode any missing stadiums, fetch and store weather for all matches
    without a weather row, and backfill weather_id. Returns {match_id -> weather_id}."""
    if conn is None:
        conn = connect(DB_DSN)

    _geocode_stadiums(conn)

    with conn.cursor() as cur:
        cur.execute("""
            SELECT m.match_id, m.match_date, s.stadium_lat, s.stadium_lng
            FROM   matches m
            LEFT JOIN stadiums s ON s.stadium_id = m.stadium_id
            LEFT JOIN weather  w ON w.match_id   = m.match_id
            WHERE  w.weather_id IS NULL
              AND  s.stadium_lat IS NOT NULL
              AND  s.stadium_lng IS NOT NULL
              AND  m.match_date  IS NOT NULL
            ORDER BY m.match_date
        """)
        work_items = [(mid, lat, lng, str(d)) for mid, d, lat, lng in cur.fetchall()]

    logger.info("Weather ingestion: %d matches need data", len(work_items))
    if not work_items:
        _backfill_weather_ids(conn)
        with conn.cursor() as cur:
            cur.execute("SELECT match_id, weather_id FROM weather")
            return {mid: wid for mid, wid in cur.fetchall()}

    sem = threading.Semaphore(max_concurrent)
    inserted = failed = 0
    with ThreadPoolExecutor(max_workers=max_concurrent * 2) as pool:
        futures = {
            pool.submit(_fetch_one, mid, lat, lng, date_str, sem): mid
            for mid, lat, lng, date_str in work_items
        }
        for fut in as_completed(futures):
            match_id, weather = fut.result()
            if weather is None:
                failed += 1
                continue
            upsert_weather(conn, match_id, weather)
            conn.commit()
            inserted += 1
            if inserted % 100 == 0:
                logger.info("  Weather progress: %d / %d", inserted, len(work_items))

    logger.info("Weather complete: %d inserted | %d failed", inserted, failed)
    _backfill_weather_ids(conn)

    with conn.cursor() as cur:
        cur.execute("SELECT match_id, weather_id FROM weather")
        return {mid: wid for mid, wid in cur.fetchall()}


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    run()
