import json
import tempfile
from collections.abc import Generator
from pathlib import Path

import pytest

from src.session_manager import SessionManager


@pytest.fixture
def sm() -> Generator[SessionManager]:
    with tempfile.TemporaryDirectory() as tmpdir:
        yield SessionManager(base_dir=tmpdir)


class TestSessionCRUD:
    def test_create_session_returns_id_and_writes_file(self, sm: SessionManager) -> None:
        sid = sm.create_session("测试会话")
        assert sid
        assert sid.startswith("session_")
        assert len(sid) == 23  # session_YYYYMMDD_HHMMSS
        path = sm._sessions_dir / f"{sid}.json"
        assert path.exists()

    def test_create_session_has_correct_fields(self, sm: SessionManager) -> None:
        sid = sm.create_session("标题")
        data = sm.load_session(sid)
        assert data is not None
        assert data["id"] == sid
        assert data["title"] == "标题"
        assert "created_at" in data
        assert "updated_at" in data
        assert data["messages"] == []

    def test_list_sessions_newest_first(self, sm: SessionManager) -> None:
        sm.create_session("first")
        import time

        time.sleep(1.1)  # ensure different second in ID
        sm.create_session("second")
        sessions = sm.list_sessions()
        assert len(sessions) == 2
        assert sessions[0]["title"] == "second"
        assert sessions[1]["title"] == "first"

    def test_list_sessions_filters_corrupt_files(self, sm: SessionManager) -> None:
        sm.create_session("good")
        bad = sm._sessions_dir / "bad.json"
        bad.write_text("not json", encoding="utf-8")
        sessions = sm.list_sessions()
        assert len(sessions) == 1
        assert sessions[0]["title"] == "good"

    def test_load_session_missing(self, sm: SessionManager) -> None:
        assert sm.load_session("nonexistent") is None

    def test_save_session_updates_timestamp(self, sm: SessionManager) -> None:
        sid = sm.create_session("original")
        data = sm.load_session(sid)
        assert data is not None
        data["title"] = "modified"
        import time

        time.sleep(0.01)
        sm.save_session(data)
        reloaded = sm.load_session(sid)
        assert reloaded is not None
        assert reloaded["title"] == "modified"
        assert reloaded["updated_at"] > reloaded["created_at"]

    def test_delete_session_removes_json(self, sm: SessionManager) -> None:
        sid = sm.create_session("to delete")
        path = sm._sessions_dir / f"{sid}.json"
        assert path.exists()
        sm.delete_session(sid)
        assert not path.exists()

    def test_delete_session_removes_runs(self, sm: SessionManager) -> None:
        sid = sm.create_session("with runs")
        run_id, run_path = sm.create_run_dir(sid)
        sm.write_report(run_path, {"key": "value"})
        assert run_path.exists()
        sm.delete_session(sid)
        assert not run_path.exists()

    def test_delete_session_does_not_raise_for_missing(self, sm: SessionManager) -> None:
        sm.delete_session("nonexistent")


class TestRunManagement:
    def test_create_run_dir_returns_id_and_creates_path(self, sm: SessionManager) -> None:
        sid = sm.create_session("s")
        run_id, run_path = sm.create_run_dir(sid)
        assert run_id
        assert run_id.startswith("run_")
        assert len(run_id) == 19  # run_YYYYMMDD_HHMMSS
        assert run_path.exists()
        assert run_path.is_dir()

    def test_run_dir_under_session_subdir(self, sm: SessionManager) -> None:
        sid = sm.create_session("s")
        run_id, run_path = sm.create_run_dir(sid)
        assert run_path.parent == sm._runs_dir / sid
        assert run_path.name == run_id

    def test_multiple_runs_under_same_session(self, sm: SessionManager) -> None:
        sid = sm.create_session("s")
        _, p1 = sm.create_run_dir(sid)
        import time

        time.sleep(1.1)  # ensure different second in ID
        _, p2 = sm.create_run_dir(sid)
        assert p1 != p2
        assert p1.parent == p2.parent

    def test_append_trace_writes_jsonl(self, sm: SessionManager) -> None:
        sid = sm.create_session("s")
        _, rp = sm.create_run_dir(sid)
        sm.append_trace(rp, {"event": "user_query", "question": "你好"})
        sm.append_trace(rp, {"event": "intent", "label": "criminal_law"})
        trace_file = rp / "trace.jsonl"
        assert trace_file.exists()
        lines = trace_file.read_text(encoding="utf-8").strip().split("\n")
        assert len(lines) == 2
        e1 = json.loads(lines[0])
        assert e1["event"] == "user_query"
        assert e1["question"] == "你好"
        assert "timestamp" in e1

    def test_append_trace_always_adds_timestamp(self, sm: SessionManager) -> None:
        sid = sm.create_session("s")
        _, rp = sm.create_run_dir(sid)
        sm.append_trace(rp, {"k": "v"})
        line = (rp / "trace.jsonl").read_text(encoding="utf-8").strip()
        entry = json.loads(line)
        assert "timestamp" in entry

    def test_write_report_creates_json(self, sm: SessionManager) -> None:
        sid = sm.create_session("s")
        _, rp = sm.create_run_dir(sid)
        report = {
            "question": "什么是刑法",
            "intent": "criminal_law",
            "answer": "刑法是...",
            "status": "success",
        }
        sm.write_report(rp, report)
        report_file = rp / "report.json"
        assert report_file.exists()
        loaded = json.loads(report_file.read_text(encoding="utf-8"))
        assert loaded == report


class TestDirectoryLayout:
    def test_creates_agent_and_sessions_dirs(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            base = Path(tmpdir) / ".agent"
            assert not base.exists()
            SessionManager(base_dir=base)
            assert base.exists()
            assert (base / "sessions").exists()

    def test_runs_dir_created_lazily(self, sm: SessionManager) -> None:
        assert not sm._runs_dir.exists()
        sid = sm.create_session("s")
        sm.create_run_dir(sid)
        assert sm._runs_dir.exists()
