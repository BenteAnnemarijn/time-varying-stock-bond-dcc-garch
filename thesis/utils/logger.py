import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path

from colorlog import ColoredFormatter

LOG_DIR = Path("logs")

# Enhanced formatters
console_formatter = ColoredFormatter(
    "%(log_color)s[%(levelname)-8s]%(reset)s %(asctime)s - %(name)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    log_colors={"DEBUG": "cyan", "INFO": "green", "WARNING": "yellow", "ERROR": "red", "CRITICAL": "bold_red"},
)

file_formatter = logging.Formatter(
    "[%(levelname)-8s] %(asctime)s - %(name)s - %(funcName)s:%(lineno)d - %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
)

logger = logging.getLogger("MasterThesisLogger")
logger.setLevel(logging.DEBUG)

LOG_DIR.mkdir(exist_ok=True)

# Rotating file handler for INFO and above
file_handler = RotatingFileHandler(
    LOG_DIR / "master_thesis.log",
    maxBytes=5 * 1024 * 1024,  # 5MB
    backupCount=5,
)
file_handler.setLevel(logging.INFO)
file_handler.setFormatter(file_formatter)

# Rotating file handler for DEBUG only
debug_file_handler = RotatingFileHandler(
    LOG_DIR / "master_thesis_debug.log",
    maxBytes=5 * 1024 * 1024,  # 5MB
    backupCount=3,
)
debug_file_handler.setLevel(logging.DEBUG)
debug_file_handler.addFilter(lambda record: record.levelno == logging.DEBUG)
debug_file_handler.setFormatter(file_formatter)

# Console handler
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
console_handler.setFormatter(console_formatter)

logger.addHandler(file_handler)
logger.addHandler(debug_file_handler)
logger.addHandler(console_handler)
