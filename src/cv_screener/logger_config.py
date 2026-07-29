"""Shared logger. Import `logger` everywhere instead of calling logging.getLogger again."""

from __future__ import annotations

import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)

logger = logging.getLogger("cv_screener")
