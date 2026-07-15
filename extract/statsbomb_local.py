from pathlib import Path
import pandas as pd

DATA_ROOT = None  # injected from config


def set_root(path):
    global DATA_ROOT
    DATA_ROOT = Path(path)


def competitions():
    return pd.read_json(DATA_ROOT / "competitions.json")


def matches(cid, sid):
    return pd.read_json(DATA_ROOT / "matches" / str(cid) / f"{sid}.json")


def events(match_id):
    return pd.read_json(DATA_ROOT / "events" / f"{match_id}.json")


def lineups(match_id):
    return pd.read_json(DATA_ROOT / "lineups" / f"{match_id}.json")