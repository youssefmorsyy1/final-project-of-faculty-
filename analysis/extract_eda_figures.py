"""Extract the EDA figures used by the frontend (/artifacts/eda/*.png) straight
from the executed eda.ipynb outputs, so the served images always match the
notebook's latest run against the current database scope.

Run after executing the notebook:
    jupyter nbconvert --to notebook --execute --inplace analysis/eda.ipynb
    python analysis/extract_eda_figures.py
"""
from __future__ import annotations

import base64
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
NB_PATH = ROOT / "analysis" / "eda.ipynb"
OUT_DIR = ROOT / "artifacts" / "eda"

# Map: absolute cell index in eda.ipynb -> output filename. Kept here (not in the
# notebook) so the notebook stays a pure analysis document.
CELL_TO_FILE = {
    7:  "competition_coverage.png",   # top-20 teams + team refs per competition
    10: "player_demographics.png",    # age / nationalities / position
    14: "match_distributions.png",    # goals, home-away, results, per-season
    19: "weather_overview.png",       # temp/precip/wind/humidity + condition mix
    24: "injury_overview.png",        # days-out, types, by year/position, games missed
    29: "metric_distributions.png",   # per-player stat histograms
    30: "correlation_matrix.png",     # Spearman correlation of player metrics
    46: "season_trends.png",          # metric trends across seasons
    47: "injury_by_position.png",     # 30-day injury rate by starting position
}


def _last_png(cell: dict) -> str | None:
    """Return the base64 image/png from the last display/execute output of a cell."""
    png = None
    for out in cell.get("outputs", []):
        data = out.get("data", {})
        if "image/png" in data:
            img = data["image/png"]
            png = "".join(img) if isinstance(img, list) else img
    return png


def main() -> int:
    nb = json.loads(NB_PATH.read_text(encoding="utf-8"))
    cells = nb["cells"]
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    written, missing = [], []
    for idx, fname in CELL_TO_FILE.items():
        cell = cells[idx] if idx < len(cells) else {}
        png = _last_png(cell) if cell.get("cell_type") == "code" else None
        if not png:
            missing.append((idx, fname))
            continue
        (OUT_DIR / fname).write_bytes(base64.b64decode(png))
        written.append(fname)

    print(f"Wrote {len(written)} figures to {OUT_DIR}:")
    for f in written:
        print(f"  + {f}")
    if missing:
        print("MISSING (no image output in cell — re-run the notebook?):")
        for idx, f in missing:
            print(f"  ! cell {idx} -> {f}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
