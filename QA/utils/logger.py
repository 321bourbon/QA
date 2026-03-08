import os
import sys
from datetime import datetime
from pathlib import Path

try:
    from loguru import logger as _loguru_logger
except Exception:
    _loguru_logger = None

_RUNTIME_LOGGER = None
_RUNTIME_STATE = None


class _SimpleRuntimeLogger:
    def __init__(self, log_path, stdout_stream):
        self.log_path = log_path
        self.stdout_stream = stdout_stream
        self._fp = open(log_path, "a", encoding="utf-8", buffering=1)

    def _emit(self, level, message):
        line = str(message)
        try:
            self.stdout_stream.write(line + "\n")
            self.stdout_stream.flush()
        except Exception:
            pass
        try:
            self._fp.write(line + "\n")
            self._fp.flush()
        except Exception:
            pass

    def log(self, level, message):
        self._emit(str(level).upper(), str(message))

    def info(self, message):
        self._emit("INFO", str(message))

    def close(self):
        try:
            self._fp.close()
        except Exception:
            pass


class _StdRedirect:
    def __init__(self, level="INFO"):
        self.level = level
        self._buffer = ""

    def write(self, message):
        if not message:
            return
        self._buffer += message
        while "\n" in self._buffer:
            line, self._buffer = self._buffer.split("\n", 1)
            line = line.rstrip("\r")
            if line and _RUNTIME_LOGGER is not None:
                _RUNTIME_LOGGER.log(self.level, line)

    def flush(self):
        if self._buffer.strip() and _RUNTIME_LOGGER is not None:
            _RUNTIME_LOGGER.log(self.level, self._buffer.strip())
        self._buffer = ""


def _safe_cmdline(argv):
    try:
        if os.name == "nt":
            import subprocess
            return os.path.basename(sys.executable) + " " + subprocess.list2cmdline(argv)
        return " ".join(argv)
    except Exception:
        return " ".join(str(x) for x in argv)


def _resolve_log_path(log_dir, file_prefix):
    shared = os.environ.get("LOGICQA_LOG_PATH", "").strip()
    if shared:
        p = Path(shared).expanduser()
        if not p.is_absolute():
            p = (Path.cwd() / p).resolve()
        p.parent.mkdir(parents=True, exist_ok=True)
        return str(p)

    Path(log_dir).mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    return str((Path(log_dir) / f"{file_prefix}+{ts}+results.log").resolve())


def _emit_session_header(log_path, reused_path):
    script = os.path.basename(sys.argv[0]) if sys.argv else "<unknown>"
    cmdline = _safe_cmdline(sys.argv)
    cwd = str(Path.cwd())
    pid = os.getpid()
    ppid = os.getppid() if hasattr(os, "getppid") else -1
    mode = "attach-existing-log" if reused_path else "new-log"
    start_ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    if _RUNTIME_LOGGER is not None:
        _RUNTIME_LOGGER.info("=" * 96)
        _RUNTIME_LOGGER.info(f"session.start | time={start_ts} | mode={mode} | script={script}")
        _RUNTIME_LOGGER.info(f"session.command | {cmdline}")
        _RUNTIME_LOGGER.info(f"session.process | pid={pid} ppid={ppid}")
        _RUNTIME_LOGGER.info(f"session.cwd | {cwd}")
        _RUNTIME_LOGGER.info(f"session.log | {log_path}")
        _RUNTIME_LOGGER.info("=" * 96)
    else:
        print("=" * 96)
        print(f"session.start | time={start_ts} | mode={mode} | script={script}")
        print(f"session.command | {cmdline}")
        print(f"session.process | pid={pid} ppid={ppid}")
        print(f"session.cwd | {cwd}")
        print(f"session.log | {log_path}")
        print("=" * 96)


def setup_runtime_logger(log_dir="./runtime_logs", file_prefix="GPT", redirect_stdout=True):
    global _RUNTIME_LOGGER, _RUNTIME_STATE
    if _RUNTIME_STATE is not None:
        return _RUNTIME_STATE["log_path"]

    reused_path = bool(os.environ.get("LOGICQA_LOG_PATH", "").strip())
    log_path = _resolve_log_path(log_dir, file_prefix)

    if _loguru_logger is None:
        _RUNTIME_LOGGER = _SimpleRuntimeLogger(log_path=log_path, stdout_stream=sys.__stdout__)
        _RUNTIME_STATE = {
            "log_path": log_path,
            "original_stdout": sys.stdout,
            "original_stderr": sys.stderr,
            "stdout_redirected": False,
        }

        if redirect_stdout:
            sys.stdout = _StdRedirect("INFO")
            sys.stderr = _StdRedirect("ERROR")
            _RUNTIME_STATE["stdout_redirected"] = True

        os.environ["LOGICQA_LOG_PATH"] = log_path
        os.environ["LOGICQA_LOGGER_ACTIVE"] = "1"
        _emit_session_header(log_path, reused_path)
        return log_path

    _RUNTIME_LOGGER = _loguru_logger.bind(component="logicqa")
    _RUNTIME_LOGGER.remove()
    fmt = "{message}"
    _RUNTIME_LOGGER.add(sys.__stdout__, level="INFO", format=fmt, enqueue=False, backtrace=False, diagnose=False)
    _RUNTIME_LOGGER.add(
        log_path,
        level="DEBUG",
        format=fmt,
        enqueue=False,
        backtrace=False,
        diagnose=False,
        encoding="utf-8",
    )

    state = {
        "log_path": log_path,
        "original_stdout": sys.stdout,
        "original_stderr": sys.stderr,
        "stdout_redirected": False,
    }

    if redirect_stdout:
        sys.stdout = _StdRedirect("INFO")
        sys.stderr = _StdRedirect("ERROR")
        state["stdout_redirected"] = True

    _RUNTIME_STATE = state
    os.environ["LOGICQA_LOG_PATH"] = log_path
    os.environ["LOGICQA_LOGGER_ACTIVE"] = "1"
    _emit_session_header(log_path, reused_path)
    return log_path


def shutdown_runtime_logger():
    global _RUNTIME_STATE, _RUNTIME_LOGGER
    if _RUNTIME_STATE is None:
        return

    if _RUNTIME_STATE.get("stdout_redirected"):
        try:
            sys.stdout.flush()
            sys.stderr.flush()
        except Exception:
            pass
        sys.stdout = _RUNTIME_STATE["original_stdout"]
        sys.stderr = _RUNTIME_STATE["original_stderr"]

    if _RUNTIME_LOGGER is not None:
        try:
            end_ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            _RUNTIME_LOGGER.info(f"session.end | time={end_ts} | log={_RUNTIME_STATE['log_path']}")
        except Exception:
            pass
        try:
            close_fn = getattr(_RUNTIME_LOGGER, "close", None)
            if callable(close_fn):
                close_fn()
        except Exception:
            pass

    _RUNTIME_STATE = None
    _RUNTIME_LOGGER = None
    os.environ.pop("LOGICQA_LOGGER_ACTIVE", None)


def get_runtime_logger():
    return _RUNTIME_LOGGER


def runtime_logger_active():
    return _RUNTIME_STATE is not None and _RUNTIME_LOGGER is not None


class Logger:
    """Backward compatible wrapper."""

    def __init__(self, log_dir="./runtime_logs"):
        self.log_path = setup_runtime_logger(log_dir=log_dir, file_prefix="GPT", redirect_stdout=True)

    def close(self):
        shutdown_runtime_logger()
