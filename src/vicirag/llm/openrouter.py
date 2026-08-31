"""Client LLM dùng OpenRouter (OpenAI-compatible).

Cơ chế cache SQLite + retry được port từ repo gốc
(`reference/CIRAG-official/src/cirag/llm/qwen_gpt.py`). Cache là thứ giữ chi phí
API xuống: khoá là sha256 của {messages, model, seed, temperature}, nên chạy lại
cùng một cấu hình gần như miễn phí và các ablation dùng chung phần OpenIE.
"""

from __future__ import annotations

import functools
import hashlib
import json
import os
import sqlite3
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from filelock import FileLock
from openai import OpenAI
from tenacity import retry, stop_after_attempt, wait_exponential

TextChatMessage = dict[str, str]


@dataclass
class TokenUsage:
    """Đếm token để dựng lại phần Efficiency Analysis (Sec 4.6 của paper)."""

    prompt_tokens: int = 0
    completion_tokens: int = 0
    num_calls: int = 0
    num_cache_hits: int = 0
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def add(self, prompt_tokens: int, completion_tokens: int, cache_hit: bool) -> None:
        with self._lock:
            self.prompt_tokens += prompt_tokens
            self.completion_tokens += completion_tokens
            self.num_calls += 1
            if cache_hit:
                self.num_cache_hits += 1

    def to_dict(self) -> dict[str, int]:
        return {
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "num_calls": self.num_calls,
            "num_cache_hits": self.num_cache_hits,
        }

    def reset(self) -> None:
        with self._lock:
            self.prompt_tokens = self.completion_tokens = 0
            self.num_calls = self.num_cache_hits = 0


def cache_response(func):
    """Decorator cache kết quả `infer` vào SQLite (port từ repo gốc)."""

    @functools.wraps(func)
    def wrapper(self, messages: list[TextChatMessage], **kwargs):
        key_data = {
            "messages": messages,
            "model": kwargs.get("model", self.model),
            "seed": self.seed,
            "temperature": kwargs.get("temperature", self.temperature),
        }
        key_hash = hashlib.sha256(
            json.dumps(key_data, sort_keys=True, ensure_ascii=False, default=str).encode("utf-8")
        ).hexdigest()

        lock_file = str(self.cache_file_name) + ".lock"
        with FileLock(lock_file):
            conn = sqlite3.connect(self.cache_file_name)
            conn.execute(
                "CREATE TABLE IF NOT EXISTS cache "
                "(key TEXT PRIMARY KEY, message TEXT, metadata TEXT)"
            )
            conn.commit()
            row = conn.execute(
                "SELECT message, metadata FROM cache WHERE key = ?", (key_hash,)
            ).fetchone()
            conn.close()
        if row is not None:
            message, metadata_str = row
            return message, json.loads(metadata_str), True

        message, metadata = func(self, messages, **kwargs)

        with FileLock(lock_file):
            conn = sqlite3.connect(self.cache_file_name)
            conn.execute(
                "CREATE TABLE IF NOT EXISTS cache "
                "(key TEXT PRIMARY KEY, message TEXT, metadata TEXT)"
            )
            conn.execute(
                "INSERT OR REPLACE INTO cache (key, message, metadata) VALUES (?, ?, ?)",
                (key_hash, message, json.dumps(metadata, ensure_ascii=False)),
            )
            conn.commit()
            conn.close()

        return message, metadata, False

    return wrapper


class OpenRouterLLM:
    """Bọc OpenAI SDK trỏ vào OpenRouter.

    Dùng chung interface `infer(messages) -> (text, metadata, cache_hit)` với repo
    gốc, nhờ vậy `ici.py` và `acmg.py` không cần biết bên dưới là model nào.
    """

    def __init__(
        self,
        model: str,
        base_url: str = "https://openrouter.ai/api/v1",
        api_key: str | None = None,
        cache_dir: str | Path = ".cache",
        cache_filename: str | None = None,
        temperature: float = 0.0,
        max_new_tokens: int | None = 2048,
        seed: int = 0,
        usage: TokenUsage | None = None,
    ) -> None:
        self.model = model
        self.temperature = temperature
        self.max_new_tokens = max_new_tokens
        self.seed = seed
        self.usage = usage if usage is not None else TokenUsage()

        api_key = api_key or os.environ.get("OPENROUTER_API_KEY", "")
        if not api_key:
            raise RuntimeError("Thiếu OPENROUTER_API_KEY (xem .env.example).")

        self.client = OpenAI(base_url=base_url, api_key=api_key, timeout=120.0, max_retries=0)

        cache_dir = Path(cache_dir)
        cache_dir.mkdir(parents=True, exist_ok=True)
        safe_model = model.replace("/", "_").replace(":", "_")
        self.cache_file_name = str(cache_dir / (cache_filename or f"llm_cache_{safe_model}.sqlite"))

    @cache_response
    @retry(stop=stop_after_attempt(5), wait=wait_exponential(multiplier=2, min=2, max=60))
    def _call(self, messages: list[TextChatMessage], **kwargs) -> tuple[str, dict[str, Any]]:
        params: dict[str, Any] = {
            "model": kwargs.get("model", self.model),
            "messages": messages,
            "temperature": kwargs.get("temperature", self.temperature),
        }
        max_tokens = kwargs.get("max_tokens", self.max_new_tokens)
        if max_tokens:
            params["max_tokens"] = max_tokens
        if kwargs.get("response_format"):
            params["response_format"] = kwargs["response_format"]

        response = self.client.chat.completions.create(**params)
        content = response.choices[0].message.content or ""
        usage = response.usage
        metadata = {
            "prompt_tokens": getattr(usage, "prompt_tokens", 0) if usage else 0,
            "completion_tokens": getattr(usage, "completion_tokens", 0) if usage else 0,
            "finish_reason": response.choices[0].finish_reason,
        }
        return content, metadata

    def infer(self, messages: list[TextChatMessage], **kwargs) -> tuple[str, dict[str, Any], bool]:
        """Trả về (nội dung, metadata, có phải cache hit)."""
        content, metadata, cache_hit = self._call(messages, **kwargs)
        self.usage.add(
            metadata.get("prompt_tokens", 0), metadata.get("completion_tokens", 0), cache_hit
        )
        return content, metadata, cache_hit
