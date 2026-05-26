"""Session and run persistence management.

Directory layout:
    .agent/
      sessions/
        <session_id>.json   # Q&A history, title, timestamps
      runs/
        <session_id>/
          <run_id>/
            trace.jsonl     # per-event trace log
            report.json     # structured run summary
"""

import json
import os
import shutil
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast


def _now_iso() -> str:
    return datetime.now(UTC).isoformat()


class SessionManager:
    def __init__(self, base_dir: str | Path = ".agent") -> None:
        self._base = Path(base_dir)
        self._sessions_dir = self._base / "sessions"
        self._runs_dir = self._base / "runs"
        self._sessions_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Session helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _session_path(sessions_dir: Path, session_id: str) -> Path:
        return sessions_dir / f"{session_id}.json"

    def list_sessions(self) -> list[dict[str, Any]]:
        """Return sessions sorted by updated_at descending (newest first)."""
        sessions: list[dict[str, Any]] = []
        for f in sorted(self._sessions_dir.glob("*.json"), key=os.path.getmtime, reverse=True):
            try:
                data = json.loads(f.read_text(encoding="utf-8"))
                sessions.append(data)
            except (json.JSONDecodeError, OSError):
                continue
        return sessions

    def load_session(self, session_id: str) -> dict[str, Any] | None:
        path = self._session_path(self._sessions_dir, session_id)
        if not path.exists():
            return None
        return cast(dict[str, Any], json.loads(path.read_text(encoding="utf-8")))

    def create_session(self, title: str) -> str:
        now = _now_iso()
        session_id = f"session_{datetime.now(UTC).strftime('%Y%m%d_%H%M%S')}"
        session = {
            "id": session_id,
            "title": title,
            "created_at": now,
            "updated_at": now,
            "messages": [],
        }
        self._session_path(self._sessions_dir, session_id).write_text(
            json.dumps(session, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        return session_id

    def save_session(self, session: dict[str, Any]) -> None:
        session["updated_at"] = _now_iso()
        self._session_path(self._sessions_dir, session["id"]).write_text(
            json.dumps(session, ensure_ascii=False, indent=2), encoding="utf-8"
        )

    def delete_session(self, session_id: str) -> None:
        path = self._session_path(self._sessions_dir, session_id)
        if path.exists():
            path.unlink()
        run_dir = self._runs_dir / session_id
        if run_dir.exists():
            shutil.rmtree(run_dir)

    # ------------------------------------------------------------------
    # Run helpers
    # ------------------------------------------------------------------

    def create_run_dir(self, session_id: str) -> tuple[str, Path]:
        run_id = f"run_{datetime.now(UTC).strftime('%Y%m%d_%H%M%S')}"
        run_path = self._runs_dir / session_id / run_id
        run_path.mkdir(parents=True, exist_ok=True)
        return run_id, run_path

    def append_trace(self, run_path: Path, event: dict[str, Any]) -> None:
        """Append one event line to trace.jsonl.

        *event* should contain at least ``{"event": "<name>", ...}`` and
        this method adds ``timestamp`` automatically.
        """
        entry = {"timestamp": _now_iso(), **event}
        trace_file = run_path / "trace.jsonl"
        with open(trace_file, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(entry, ensure_ascii=False) + "\n")

    def write_report(self, run_path: Path, report: dict[str, Any]) -> None:
        report_file = run_path / "report.json"
        report_file.write_text(
            json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
        )
