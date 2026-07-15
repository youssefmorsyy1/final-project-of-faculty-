"""
utils/validate_data.py

Validation helpers for StatsBomb event data.

StatsBomb events do NOT have a top-level 'match' key -- match context is
passed in at the pipeline level.  This module validates the fields that
actually exist in event rows.
"""

import logging
from typing import Dict, Any, List

import pandas as pd

logger = logging.getLogger("data")
logger.setLevel(logging.WARNING)


def validate_event(ev: Dict[str, Any]) -> bool:
    """
    Validate a single StatsBomb event dict.

    Required fields:
      - player  (dict with 'id')
      - type    (dict with 'name', or a string)

    The 'match' key does NOT exist in StatsBomb event rows -- match context
    is held at the pipeline level.
    """
    player_data = ev.get("player")
    if not isinstance(player_data, dict):
        return False
    if "id" not in player_data:
        return False

    type_data = ev.get("type")
    if type_data is None:
        return False
    if isinstance(type_data, dict) and "name" not in type_data:
        return False

    return True


def validate_batch(events: pd.DataFrame) -> Dict[str, Any]:
    """
    Validate a batch of events and return statistics.

    Parameters
    ----------
    events : pd.DataFrame
        DataFrame of StatsBomb events for a single match.

    Returns
    -------
    dict with keys: valid_count, invalid_count, valid_rate, invalid_reasons
    """
    valid_count    = 0
    invalid_count  = 0
    invalid_reasons: List[tuple] = []

    for idx, ev in events.iterrows():
        ev_dict = ev.to_dict() if hasattr(ev, "to_dict") else dict(ev)
        if validate_event(ev_dict):
            valid_count += 1
        else:
            invalid_count += 1
            player_data = ev_dict.get("player")
            type_data   = ev_dict.get("type")

            if not isinstance(player_data, dict):
                reason = "Missing or invalid player field"
            elif "id" not in player_data:
                reason = "Player dict missing 'id'"
            elif type_data is None:
                reason = "Missing type field"
            else:
                reason = "Invalid type dict (missing 'name')"

            invalid_reasons.append((idx, reason))

    stats = {
        "valid_count":    valid_count,
        "invalid_count":  invalid_count,
        "valid_rate":     valid_count / max(1, valid_count + invalid_count),
        "invalid_reasons": invalid_reasons[:10],
    }

    if invalid_count > 0:
        logger.warning("Batch validation: %d invalid events", invalid_count)
        for idx, reason in invalid_reasons[:5]:
            logger.debug("  Invalid event %s: %s", idx, reason)

    return stats


def validate_aggregated_stats(stats: Dict[str, Any]) -> bool:
    """
    Validate a player stats dict before DB insertion.

    Checks that all required keys are present and values are non-negative
    numbers.  Pass accuracy is expected as a percentage (0-100).
    """
    required = [
        "goals", "assists", "shots", "xg", "xa",
        "passes_attempted", "passes_completed", "pass_accuracy",
        "tackles", "interceptions", "clearances", "pressures",
        "yellow_cards", "red_cards", "minutes_played",
        "key_passes", "progressive_passes",
        "carry_distance", "progressive_carries",
        "dribbles_completed",
    ]

    for key in required:
        if key not in stats:
            logger.warning("Missing stat key: %s", key)
            return False

    numeric = [k for k in required if k != "sub_minute"]
    for field in numeric:
        value = stats.get(field, 0)
        if not isinstance(value, (int, float)):
            logger.warning("Stat %s is not numeric: %s", field, type(value))
            return False
        if value < 0:
            logger.warning("Stat %s is negative: %s", field, value)
            return False

    accuracy = stats.get("pass_accuracy", 0)
    if not (0 <= accuracy <= 100):
        logger.warning("Pass accuracy out of expected 0-100 range: %s", accuracy)
        return False

    return True


__all__ = ["validate_event", "validate_batch", "validate_aggregated_stats"]