import logging
import sys
from datetime import datetime
from pathlib import Path

LOG_FORMAT = (
    "[%(asctime)s] [%(levelname)s] [%(name)s] "
    "[session_id=%(session_id)s] %(message)s"
)


class SessionFilter(logging.Filter):
    """Inject session_id into log records."""

    def __init__(self, session_id: str | None = None) -> None:
        super().__init__()
        self.session_id = session_id or "N/A"

    def filter(self, record: logging.LogRecord) -> bool:
        record.session_id = self.session_id  # type: ignore[attr-defined,unused-ignore]
        return True


def get_logger(
    name: str,
    log_dir: str = "logs",
    level: str = "INFO",
    session_id: str | None = None,
) -> logging.Logger:
    """Get a configured logger with console and file handlers.

    Handlers are only added once per logger name.
    """
    logger = logging.getLogger(name)

    if logger.handlers:
        return logger

    logger.setLevel(getattr(logging, level.upper(), logging.INFO))

    # Console handler (stdout, INFO+)
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(logging.Formatter(LOG_FORMAT))
    logger.addHandler(console_handler)

    # File handler (DEBUG+)
    log_path = Path(log_dir)
    log_path.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    file_handler = logging.FileHandler(
        log_path / f"app_{timestamp}.log", encoding="utf-8"
    )
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(logging.Formatter(LOG_FORMAT))
    logger.addHandler(file_handler)

    # Session filter
    logger.addFilter(SessionFilter(session_id))

    return logger
