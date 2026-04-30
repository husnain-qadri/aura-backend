import hashlib
import threading
from typing import Any

_lock = threading.Lock()
_store: dict[str, dict[str, Any]] = {}


def doc_id_from_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()[:24]


def put(doc_id: str, document: dict[str, Any]) -> None:
    with _lock:
        _store[doc_id] = document


def get(doc_id: str) -> dict[str, Any] | None:
    with _lock:
        return _store.get(doc_id)


def has(doc_id: str) -> bool:
    with _lock:
        return doc_id in _store
