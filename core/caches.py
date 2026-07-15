"""In-memory caches for teams and players."""

from psycopg2.extras import execute_values
from core.utils import norm_name


class TeamCache:
    def __init__(self, conn):
        self.conn = conn
        self.cache: dict[int, int] = {}        # sb_team_id -> team_id
        self._pending: dict[int, str] = {}     # sb_team_id -> name
        self._load()

    def _load(self):
        with self.conn.cursor() as cur:
            cur.execute("SELECT team_id, sb_team_id FROM teams")
            for tid, sid in cur.fetchall():
                self.cache[sid] = tid

    def get_or_create(self, sb_id: int, name: str) -> int | None:
        if sb_id in self.cache:
            if name and sb_id in self._pending and not self._pending[sb_id]:
                self._pending[sb_id] = name
            return self.cache[sb_id]

        if sb_id not in self._pending:
            self._pending[sb_id] = name
        elif name and not self._pending[sb_id]:
            self._pending[sb_id] = name
        return None

    def flush(self):
        if not self._pending:
            return
        rows = [
            (name or f"Team {sb_id}", sb_id)
            for sb_id, name in self._pending.items()
        ]
        with self.conn.cursor() as cur:
            execute_values(cur, """
                INSERT INTO teams (team_name, sb_team_id)
                VALUES %s
                ON CONFLICT (sb_team_id) DO UPDATE SET
                    team_name = CASE
                        WHEN EXCLUDED.team_name = teams.team_name THEN teams.team_name
                        WHEN teams.team_name LIKE 'Team %%'       THEN EXCLUDED.team_name
                        ELSE teams.team_name
                    END
                RETURNING team_id, sb_team_id
            """, rows)
            for tid, sid in cur.fetchall():
                self.cache[sid] = tid
        self.conn.commit()
        self._pending.clear()

    def resolve(self, sb_id: int) -> int:
        return self.cache[sb_id]


class PlayerCache:
    def __init__(self, conn):
        self.conn = conn
        self.sb: dict[int, int] = {}            # sb_player_id -> player_id
        self.norm: dict[str, int] = {}          # norm_name    -> player_id
        self._pending: dict[int, tuple[str, str]] = {}
        self._load()

    def _load(self):
        with self.conn.cursor() as cur:
            cur.execute("SELECT player_id, sb_player_id, norm_name FROM players")
            for pid, sid, nn in cur.fetchall():
                if sid:
                    self.sb[sid] = pid
                if nn:
                    self.norm[nn] = pid

    def get_or_create(self, sb_id: int, name: str) -> int | None:
        if sb_id in self.sb:
            return self.sb[sb_id]

        nn = norm_name(name)
        if nn in self.norm:
            pid = self.norm[nn]
            self.sb[sb_id] = pid
            self._pending_backfill = getattr(self, "_pending_backfill", {})
            self._pending_backfill[sb_id] = pid
            return pid

        if sb_id not in self._pending:
            self._pending[sb_id] = (name, nn)
        return None

    def flush(self):
        backfill = getattr(self, "_pending_backfill", {})
        if backfill:
            with self.conn.cursor() as cur:
                execute_values(cur, """
                    UPDATE players AS p
                    SET sb_player_id = v.sb_id
                    FROM (VALUES %s) AS v(sb_id, player_id)
                    WHERE p.player_id = v.player_id
                      AND p.sb_player_id IS NULL
                """, [(sb_id, pid) for sb_id, pid in backfill.items()])
            self._pending_backfill = {}

        if not self._pending:
            if backfill:
                self.conn.commit()
            return

        rows = [(sb_id, name, nn) for sb_id, (name, nn) in self._pending.items()]
        with self.conn.cursor() as cur:
            execute_values(cur, """
                INSERT INTO players (sb_player_id, player_name, norm_name)
                VALUES %s
                ON CONFLICT (sb_player_id) DO NOTHING
                RETURNING player_id, sb_player_id, norm_name
            """, rows)
            for pid, sid, nn in cur.fetchall():
                self.sb[sid] = pid
                self.norm[nn] = pid

        self.conn.commit()
        self._pending.clear()

    def resolve(self, sb_id: int) -> int:
        return self.sb[sb_id]