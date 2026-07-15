def player_id(ev):
    p = ev.get("player")
    return p["id"] if isinstance(p, dict) else ev.get("player_id")


def team_id(ev):
    t = ev.get("team")
    return t["id"] if isinstance(t, dict) else ev.get("team_id")


def event_type(ev):
    t = ev.get("type")
    return t["name"] if isinstance(t, dict) else t