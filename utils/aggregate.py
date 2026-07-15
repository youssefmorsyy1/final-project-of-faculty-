"""
utils/aggregate.py

Higher-level aggregation utilities built on top of transform/features.py.
Uses the correct StatsBomb nested-dict schema throughout.

Key difference from transform/features.py
------------------------------------------
transform/features.agg_player_events() -- per-player, per-match aggregation
utils/aggregate functions               -- match-level and batch-level helpers
                                           used by downstream ML feature extraction
"""

import logging
from typing import Dict, Any, List, Tuple

import pandas as pd

from transform.features import agg_player_events
from transform.schema import event_type as get_event_type

logger = logging.getLogger("aggregate")
logger.setLevel(logging.DEBUG)


def _get_player_id(ev) -> int | None:
    """Extract StatsBomb player id from an event row."""
    p = ev.get("player")
    return p.get("id") if isinstance(p, dict) else None


def _get_team_id(ev) -> int | None:
    """Extract StatsBomb team id from an event row."""
    t = ev.get("team")
    return t.get("id") if isinstance(t, dict) else None


def aggregate_player_events(
    events: pd.DataFrame,
    player_id: int,
    event_type_filter: str = "all",
) -> Dict[str, Any]:
    """
    Aggregate all events for a single player across a match.

    Parameters
    ----------
    events          : DataFrame of all events for the match.
    player_id       : StatsBomb player id (integer).
    event_type_filter : 'all' | 'attack' | 'defence' | 'setpiece'
                        Filters events to a subset before aggregation.
                        Note: filtering happens AFTER extracting the player
                        subset so that minutes_played is always correct.

    Returns
    -------
    dict of aggregated stats matching player_match_stats columns.
    """
    # First pass: get all events for this player (for time/minutes)
    player_events = events[events["player"].apply(
        lambda x: isinstance(x, dict) and x.get("id") == player_id
    )]

    if player_events.empty:
        return _empty_stats()

    # Apply optional event-type filter to a copy for counting purposes
    # but keep full player_events for the canonical aggregation below
    if event_type_filter != "all":
        _FILTER_MAP = {
            "attack":   {"Shot", "Dribble", "Carry"},
            "defence":  {"Duel", "Interception", "Clearance", "Pressure", "Block"},
            "setpiece": {"Pass", "Free Kick", "Corner Received"},
        }
        keep = _FILTER_MAP.get(event_type_filter, set())
        player_events = player_events[player_events["type"].apply(
            lambda t: (isinstance(t, dict) and t.get("name") in keep)
        )]

    # Delegate to the canonical aggregation function
    return agg_player_events(player_events, player_id, get_event_type)


def aggregate_match_stats(events: pd.DataFrame, match_id: int) -> Dict[str, Any]:
    """
    Compute match-level summary stats from a DataFrame of events.

    Parameters
    ----------
    events   : DataFrame of all events for the match.
    match_id : Internal (PostgreSQL) match id -- stored in the result dict.

    Returns
    -------
    dict with match-level aggregated stats.
    """
    if events.empty:
        return _empty_match_stats(match_id)

    home_goals = away_goals = 0
    home_xg    = away_xg    = 0.0

    # Identify home and away teams from Starting XI events
    home_team_id = away_team_id = None
    starting_xi = events[events["type"].apply(
        lambda t: isinstance(t, dict) and t.get("name") == "Starting XI"
    )]
    team_ids = [_get_team_id(r) for _, r in starting_xi.iterrows() if _get_team_id(r)]
    if len(team_ids) >= 2:
        home_team_id, away_team_id = team_ids[0], team_ids[1]

    # Shot events
    shot_events = events[events["type"].apply(
        lambda t: isinstance(t, dict) and t.get("name") == "Shot"
    )]
    for _, ev in shot_events.iterrows():
        shot = ev.get("shot") or {}
        xg   = shot.get("statsbomb_xg") or 0.0
        outcome = shot.get("outcome")
        if isinstance(outcome, dict):
            outcome = outcome.get("name", "")
        team = _get_team_id(ev)

        if team == home_team_id:
            home_xg += xg
            if outcome == "Goal":
                home_goals += 1
        elif team == away_team_id:
            away_xg += xg
            if outcome == "Goal":
                away_goals += 1

    return {
        "match_id":    match_id,
        "home_team_id": home_team_id,
        "away_team_id": away_team_id,
        "home_goals":   home_goals,
        "away_goals":   away_goals,
        "home_xg":      round(home_xg, 4),
        "away_xg":      round(away_xg, 4),
        "total_events": len(events),
    }


def process_batch(
    events_df: pd.DataFrame,
) -> Tuple[List[Dict[str, Any]], Dict[int, Dict[str, Any]]]:
    """
    Process a DataFrame that spans multiple matches.

    Parameters
    ----------
    events_df : DataFrame with a 'match_id' column (internal pg id) plus
                all standard StatsBomb event columns.

    Returns
    -------
    (player_stats, match_stats)
      player_stats : list of per-player stat dicts, each including match_id
      match_stats  : dict of match_id -> match-level aggregation dict
    """
    player_stats: List[Dict[str, Any]] = []
    match_stats:  Dict[int, Dict[str, Any]] = {}

    if "match_id" not in events_df.columns:
        logger.error("process_batch: events_df missing 'match_id' column")
        return player_stats, match_stats

    for match_id, match_events in events_df.groupby("match_id"):
        match_agg = aggregate_match_stats(match_events, match_id)
        match_stats[match_id] = match_agg

        unique_players = set()
        for _, ev in match_events.iterrows():
            pid = _get_player_id(ev)
            if pid:
                unique_players.add(pid)

        for pid in unique_players:
            try:
                stat = aggregate_player_events(match_events, pid)
                stat["match_id"]  = match_id
                stat["player_id"] = pid
                player_stats.append(stat)
            except Exception as exc:
                logger.error(
                    "Error processing player %d in match %d: %s", pid, match_id, exc
                )
                empty = _empty_stats()
                empty["match_id"]  = match_id
                empty["player_id"] = pid
                player_stats.append(empty)

    return player_stats, match_stats


def _empty_stats() -> Dict[str, Any]:
    return {
        "goals": 0, "assists": 0, "shots": 0,
        "xg": 0.0, "xa": 0.0,
        "key_passes": 0,
        "passes_attempted": 0, "passes_completed": 0, "pass_accuracy": 0.0,
        "progressive_passes": 0,
        "carry_distance": 0.0, "progressive_carries": 0,
        "dribbles_completed": 0,
        "tackles": 0, "interceptions": 0, "clearances": 0, "pressures": 0,
        "yellow_cards": 0, "red_cards": 0,
        "minutes_played": 0, "sub_minute": None,
    }


def _empty_match_stats(match_id: int) -> Dict[str, Any]:
    return {
        "match_id": match_id,
        "home_team_id": None, "away_team_id": None,
        "home_goals": 0, "away_goals": 0,
        "home_xg": 0.0, "away_xg": 0.0,
        "total_events": 0,
    }


__all__ = [
    "aggregate_player_events",
    "aggregate_match_stats",
    "process_batch",
]