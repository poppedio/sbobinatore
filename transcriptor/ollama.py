"""Client minimale per Ollama: disponibilità, modelli, chat con output JSON vincolato da schema."""
from __future__ import annotations

import json
import re
import threading
from typing import Dict, List, Optional, Tuple

import requests

DEFAULT_URL = "http://localhost:11434"
PREFERRED_MODELS = ["qwen3:30b", "qwen3", "gemma3", "gemma", "llama3", "mistral", "gpt-oss"]

MIN_CTX = 4096
MAX_CTX = 32768

_CAPS_CACHE: Dict[Tuple[str, str], List[str]] = {}
_CAPS_LOCK = threading.Lock()


class OllamaError(RuntimeError):
    pass


def is_available(base_url: str = DEFAULT_URL, timeout: float = 2.0) -> bool:
    try:
        return requests.get(f"{base_url.rstrip('/')}/api/tags", timeout=timeout).ok
    except Exception:
        return False


def list_models(base_url: str = DEFAULT_URL, timeout: float = 5.0) -> List[str]:
    try:
        response = requests.get(f"{base_url.rstrip('/')}/api/tags", timeout=timeout)
        response.raise_for_status()
        models = response.json().get("models", [])
        return sorted(m["name"] for m in models if isinstance(m, dict) and m.get("name"))
    except Exception:
        return []


def pick_default_model(models: List[str]) -> Optional[str]:
    if not models:
        return None
    for preferred in PREFERRED_MODELS:
        for name in models:
            if name == preferred or name.startswith(preferred):
                return name
    return models[0]


def model_capabilities(base_url: str, model: str) -> List[str]:
    key = (base_url, model)
    with _CAPS_LOCK:
        if key in _CAPS_CACHE:
            return _CAPS_CACHE[key]
    caps: List[str] = []
    try:
        response = requests.post(f"{base_url.rstrip('/')}/api/show", json={"model": model}, timeout=10)
        if response.ok:
            caps = list(response.json().get("capabilities") or [])
    except Exception:
        caps = []
    with _CAPS_LOCK:
        _CAPS_CACHE[key] = caps
    return caps


def estimate_tokens(text: str) -> int:
    # Italiano con tokenizer moderni: circa 3-3.5 caratteri per token. Stima prudente.
    return len(text or "") // 3 + 1


def choose_num_ctx(prompt_text: str, reserve_output: int = 2048) -> int:
    needed = estimate_tokens(prompt_text) + reserve_output
    rounded = ((needed + 1023) // 1024) * 1024
    return max(MIN_CTX, min(MAX_CTX, rounded))


# --- parsing tollerante (il formato JSON vincolato lo rende quasi sempre superfluo) ---
def strip_wrappers(raw: str) -> str:
    cleaned = (raw or "").replace("﻿", "").replace("\x00", "").strip()
    cleaned = re.sub(r"<think>.*?</think>", " ", cleaned, flags=re.DOTALL | re.IGNORECASE)
    cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\s*```$", "", cleaned)
    return cleaned.strip()


def extract_first_json_object(raw: str) -> Optional[str]:
    start = raw.find("{")
    if start == -1:
        return None
    depth = 0
    in_string = False
    escape = False
    for idx in range(start, len(raw)):
        ch = raw[idx]
        if in_string:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_string = False
            continue
        if ch == '"':
            in_string = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return raw[start: idx + 1]
    return None


def parse_json_object(raw: str) -> Dict:
    cleaned = strip_wrappers(raw)
    if not cleaned:
        raise OllamaError("Il modello ha restituito una risposta vuota.")
    candidates = [cleaned]
    block = extract_first_json_object(cleaned)
    if block and block != cleaned:
        candidates.append(block)
    last_error: Optional[Exception] = None
    for candidate in candidates:
        try:
            parsed = json.loads(candidate)
            if isinstance(parsed, str):
                parsed = json.loads(parsed)
            if isinstance(parsed, dict):
                return parsed
            last_error = ValueError("il JSON non è un oggetto")
        except (json.JSONDecodeError, ValueError) as exc:
            last_error = exc
    raise OllamaError(f"Risposta non interpretabile come JSON: {last_error}. Anteprima: {cleaned[:300]}")


def chat_json(
    *,
    base_url: str,
    model: str,
    system: str,
    user: str,
    schema: Dict,
    temperature: float = 0.2,
    reserve_output: int = 2048,
    timeout: float = 900.0,
    keep_alive: str = "15m",
) -> Tuple[Dict, Dict]:
    """Una chiamata /api/chat con output vincolato dallo schema JSON. Ritorna (oggetto, statistiche)."""
    url = f"{base_url.rstrip('/')}/api/chat"
    num_ctx = choose_num_ctx(system + user, reserve_output=reserve_output)
    payload: Dict = {
        "model": model,
        "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
        "stream": False,
        "format": schema,
        "keep_alive": keep_alive,
        "options": {"temperature": temperature, "num_ctx": num_ctx},
    }
    if "thinking" in model_capabilities(base_url, model):
        payload["think"] = False

    try:
        response = requests.post(url, json=payload, timeout=timeout)
    except requests.exceptions.ConnectionError as exc:
        raise OllamaError(f"Ollama non raggiungibile su {base_url}.") from exc
    except requests.exceptions.Timeout as exc:
        raise OllamaError(f"Ollama non ha risposto entro {int(timeout)} s.") from exc

    if response.status_code == 400 and "think" in payload and "think" in response.text.lower():
        payload.pop("think", None)
        response = requests.post(url, json=payload, timeout=timeout)
    if not response.ok:
        detail = response.text[:400]
        try:
            detail = response.json().get("error", detail)
        except Exception:
            pass
        raise OllamaError(f"Ollama ha risposto {response.status_code}: {detail}")

    data = response.json()
    content = (data.get("message") or {}).get("content", "")
    parsed = parse_json_object(content)
    prompt_tokens = data.get("prompt_eval_count")
    stats = {
        "num_ctx": num_ctx,
        "prompt_tokens": prompt_tokens,
        "output_tokens": data.get("eval_count"),
        "total_seconds": round((data.get("total_duration") or 0) / 1e9, 1),
        "truncated": bool(prompt_tokens and prompt_tokens >= num_ctx - reserve_output // 2),
    }
    return parsed, stats
