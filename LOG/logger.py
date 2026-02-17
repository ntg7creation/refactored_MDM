# LOG/logger.py

import os
import threading
from datetime import datetime
from pathlib import Path
from collections import defaultdict


class LOG:
    """
    A static, global logger with two modes:
    - direct_mode = True : immediately append to file
    - direct_mode = False: bucket-only (training-safe), dump at epoch end

    Usage:
        LOG.log("foo")
        LOG.dump()
    """

    # ========================
    # STATIC (CLASS) VARIABLES
    # ========================
    _initialized = False
    _lock = threading.Lock()
    direct_mode = True         
    bucket = defaultdict(int)
    log_file_path = None

    @classmethod
    def _init(cls):
        """Initialize once."""
        if cls._initialized:
            return

        with cls._lock:
            if cls._initialized:
                return

            log_dir = Path("LOG")
            log_dir.mkdir(exist_ok=True)

            date_str = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            cls.log_file_path = log_dir / f"log_{date_str}.txt"

            cls.bucket = defaultdict(int)
            cls.direct_mode = False
            cls._initialized = True

    # ========================
    # PUBLIC STATIC API
    # ========================

    @classmethod
    def log(cls, tag: str):
        """Record a log entry."""
        cls._init()

        if cls.direct_mode:
            # Write directly (slow)
            with cls._lock:
                with open(cls.log_file_path, "a", encoding="utf-8") as f:
                    f.write(f"{datetime.now()} - {tag}\n")
        else:
            # Training-safe: no disk I/O
            with cls._lock:
                cls.bucket[tag] += 1

    @classmethod
    def dump(cls, clear_after=True):
        """Write bucket summary to file."""
        cls._init()

        with cls._lock:
            if not cls.bucket:
                return

            with open(cls.log_file_path, "a", encoding="utf-8") as f:
                f.write("\n==== EPOCH SUMMARY ====\n")
                for tag, count in sorted(cls.bucket.items()):
                    f.write(f"{tag}: {count}\n")
                f.write("=======================\n\n")

            if clear_after:
                cls.bucket.clear()

    @classmethod
    def enable_direct_mode(cls):
        cls._init()
        cls.direct_mode = True

    @classmethod
    def disable_direct_mode(cls):
        cls._init()
        cls.direct_mode = False

    @classmethod
    def set_mode(cls, direct: bool):
        cls._init()
        cls.direct_mode = direct
