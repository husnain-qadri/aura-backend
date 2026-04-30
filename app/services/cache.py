"""Simple in-memory LRU cache for LLM responses."""

import hashlib
import threading
from collections import OrderedDict
from typing import Any

_lock = threading.Lock()
_cache: OrderedDict[str, Any] = OrderedDict()
MAX_ENTRIES = 200


def _make_key(*parts: str) -> str:
    combined = '||'.join(parts)
    return hashlib.sha256(combined.encode()).hexdigest()[:32]


def get(namespace: str, *parts: str) -> Any | None:
    key = _make_key(namespace, *parts)
    with _lock:
        if key in _cache:
            _cache.move_to_end(key)
            return _cache[key]
    return None


def put(namespace: str, *parts: str, value: Any) -> None:
    key = _make_key(namespace, *parts)
    with _lock:
        _cache[key] = value
        _cache.move_to_end(key)
        while len(_cache) > MAX_ENTRIES:
            _cache.popitem(last=False)
