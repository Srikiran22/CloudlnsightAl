# Lightweight observability for a local-first app: stdlib logging to the
# console Streamlit was launched from. Nothing sensitive is ever logged --
# no API keys, no dataset contents, no raw model responses.
#
# Where logs go: stderr of the `streamlit run App.py` terminal.
# Level is configurable via the CLOUDINSIGHT_LOG_LEVEL env var
# (DEBUG / INFO / WARNING / ERROR; default WARNING).

import logging
import os
import sys


_LOGGER_NAME = "cloudinsight"
_configured = False


def _configure():
    global _configured
    if _configured:
        return
    level_name = os.environ.get("CLOUDINSIGHT_LOG_LEVEL", "WARNING").upper()
    level = getattr(logging, level_name, logging.WARNING)
    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(logging.Formatter(
        "%(asctime)s %(levelname)s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    ))
    logger = logging.getLogger(_LOGGER_NAME)
    logger.setLevel(level)
    logger.addHandler(handler)
    logger.propagate = False
    _configured = True


def get_logger(module):
    """Return the project logger namespaced per module, e.g. cloudinsight.ML."""
    _configure()
    return logging.getLogger(f"{_LOGGER_NAME}.{module}")
