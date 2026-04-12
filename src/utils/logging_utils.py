from datetime import datetime
from pathlib import Path

import src.config as config


def _ensure_log_directory(file_path: Path) -> None:
    file_path.parent.mkdir(parents=True, exist_ok=True)


def _get_timestamp() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def log_message(
        message: str,
        file_name: str,
        base_path: Path = config.LOGS_BASE_PATH,
) -> None:
    file_path = base_path / file_name

    _ensure_log_directory(file_path)

    timestamp = _get_timestamp()
    log_line = f"[{timestamp}] {message}\n"

    with file_path.open("a", encoding="utf-8") as file_obj:
        file_obj.write(log_line)


def log_and_print_message(
        message: str,
        file_name: str,
        base_path: Path = config.LOGS_BASE_PATH,
) -> None:
    log_message(message, file_name, base_path)
    print(message)


def clear_log_file(
        file_name: str,
        base_path: Path = config.LOGS_BASE_PATH,
) -> None:
    file_path = base_path / file_name

    if file_path.exists():
        file_path.unlink()
