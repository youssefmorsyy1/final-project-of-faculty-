"""
config/settings.py

All configuration is read from environment variables.
Copy .env.example to .env and fill in your values before running anything.
"""

import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()


def _require(key: str) -> str:
    val = os.getenv(key)
    if not val:
        raise RuntimeError(
            f"Required environment variable '{key}' is not set. "
            "Copy .env.example to .env and fill in your values."
        )
    return val


# ---------------------------------------------------------------------------
# Database
# ---------------------------------------------------------------------------
DB_DSN = _require("DB_DSN")

# ---------------------------------------------------------------------------
# Data paths
# ---------------------------------------------------------------------------
DATA_ROOT                 = Path(_require("DATA_ROOT"))
TRANSFERMARKT_CSV         = Path(_require("TRANSFERMARKT_CSV"))
TRANSFERMARKT_PLAYERS_CSV = Path(_require("TRANSFERMARKT_PLAYERS_CSV"))

# ---------------------------------------------------------------------------
# External APIs
# ---------------------------------------------------------------------------
OPEN_METEO_URL = os.getenv(
    "OPEN_METEO_URL",
    "https://archive-api.open-meteo.com/v1/archive",
)

# ---------------------------------------------------------------------------
# (competition_id, season_id) pairs that are in scope.
#
# Scope = the complete 2015/16 season of all five major European leagues
# (full team coverage, no single-club bias) plus the full men's international
# tournaments. This replaces the earlier Barcelona-skewed set (one full La Liga
# season + five Barcelona-only seasons + 1 UCL match), giving a balanced,
# multi-league dataset (~2,025 matches across ~100 clubs and 50+ nations).
# ---------------------------------------------------------------------------
COMPETITIONS = {
    (11, 27),   # La Liga 2015/16        (380 matches, 20 teams)
    ( 2, 27),   # Premier League 2015/16 (380, 20)
    (12, 27),   # Serie A 2015/16        (380, 20)
    ( 7, 27),   # Ligue 1 2015/16        (377, 20)
    ( 9, 27),   # 1. Bundesliga 2015/16  (306, 18)
    (43,  3),   # FIFA World Cup 2018     (64, 32)
    (43, 106),  # FIFA World Cup 2022     (64, 32) -- has 360 data
    (55, 43),   # UEFA Euro 2020          (51, 24) -- has 360 data
    (55, 282),  # UEFA Euro 2024          (51, 24) -- has 360 data
}