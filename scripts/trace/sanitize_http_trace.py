#!/usr/bin/env python3
"""Sanitize mitmproxy traffic JSONL from stdin to stdout.

Raw traces can include API keys, auth headers, local paths, and lossless body
bytes.  This script redacts sensitive text and always drops base64 payloads.
"""

from __future__ import annotations

import base64
import ipaddress
import json
import re
import sys
from typing import Any
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit


REDACTED = "<redacted>"
SENSITIVE_HEADERS = re.compile(
    r"^(authorization|proxy-authorization|cookie|set-cookie|x-api-key|api-key|"
    r"anthropic-api-key|openai-api-key|openai-organization|"
    r"cf-access-client-secret|x-request-id|request-id|traceparent|tracestate)$",
    re.I,
)
SENSITIVE_KEYS = re.compile(
    r"(api[_-]?key|token|"
    r"authorization|password|passwd|secret|cookie|session|credential|"
    r"private[_-]?key|openai[_-]?organization|email|phone|user[_-]?id)$",
    re.I,
)
SENSITIVE_QUERY = re.compile(r"(api[_-]?key|token|signature|sig|secret|password|session|code)$", re.I)
TEXT_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("bearer_token", re.compile(r"\bBearer\s+[A-Za-z0-9._~+/=-]{12,}", re.I)),
    ("openai_key", re.compile(r"\bsk-(?:proj-)?[A-Za-z0-9_-]{16,}")),
    ("github_token", re.compile(r"\b(?:ghp|gho|ghu|ghs|ghr|github_pat)_[A-Za-z0-9_]{20,}")),
    ("jwt", re.compile(r"\beyJ[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}\b")),
    ("email", re.compile(r"(?<!\\)\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b")),
    ("ipv4", re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}\b")),
    ("host_home_path", re.compile(r"(?<![\w.-])(?:[A-Za-z]:)?[/\\](?:data\d+[/\\]home|home|Users)[/\\][^/\\\s\"':]+")),
)
IPV6_CANDIDATE = re.compile(r"(?<![\w.])(?:[0-9A-Fa-f]{0,4}:){2,7}[0-9A-Fa-f]{0,4}(?![\w.])")


def bump(counts: dict[str, int], name: str, n: int = 1) -> None:
    counts[name] = counts.get(name, 0) + n


def redact_text(text: str, counts: dict[str, int]) -> str:
    for name, pattern in TEXT_PATTERNS:
        text, n = pattern.subn(REDACTED, text)
        if n:
            bump(counts, name, n)
    text = IPV6_CANDIDATE.sub(lambda match: redact_ipv6(match, counts), text)
    return text


def redact_ipv6(match: re.Match[str], counts: dict[str, int]) -> str:
    value = match.group(0)
    try:
        ipaddress.IPv6Address(value)
    except ValueError:
        return value
    bump(counts, "ipv6")
    return REDACTED


def redact_json(value: Any, counts: dict[str, int], key: str | None = None) -> Any:
    if isinstance(value, dict):
        out = {}
        for raw_key, child in value.items():
            child_key = str(raw_key)
            if SENSITIVE_KEYS.search(child_key):
                out[child_key] = REDACTED
                bump(counts, "sensitive_json_field")
            else:
                out[child_key] = redact_json(child, counts, child_key)
        return out
    if isinstance(value, list):
        return [redact_json(item, counts, key) for item in value]
    if isinstance(value, str):
        if key and SENSITIVE_KEYS.search(key):
            bump(counts, "sensitive_json_field")
            return REDACTED
        return redact_text(value, counts)
    return value


def redact_payload_text(text: str, counts: dict[str, int]) -> str:
    try:
        return json.dumps(redact_json(json.loads(text), counts), ensure_ascii=False, separators=(",", ":"))
    except Exception:
        return redact_text(text, counts)


def sanitize_pair_list(items: Any, sensitive_re: re.Pattern[str], count_name: str, counts: dict[str, int]) -> list[list[str]]:
    out: list[list[str]] = []
    if not isinstance(items, list):
        return out
    for item in items:
        if not (isinstance(item, list) and len(item) == 2):
            continue
        key, value = str(item[0]), str(item[1])
        if sensitive_re.search(key):
            out.append([key, REDACTED])
            bump(counts, count_name)
        else:
            out.append([key, redact_text(value, counts)])
    return out


def sanitize_url(url: str, counts: dict[str, int]) -> str:
    try:
        parts = urlsplit(url)
    except Exception:
        return redact_text(url, counts)
    netloc = parts.netloc
    if "@" in netloc:
        netloc = f"{REDACTED}@{netloc.rsplit('@', 1)[1]}"
        bump(counts, "sensitive_url_auth")
    query = []
    for key, value in parse_qsl(parts.query, keep_blank_values=True):
        if SENSITIVE_QUERY.search(key):
            query.append((key, REDACTED))
            bump(counts, "sensitive_query")
        else:
            query.append((key, redact_text(value, counts)))
    return urlunsplit((parts.scheme, netloc, redact_text(parts.path, counts), urlencode(query), redact_text(parts.fragment, counts)))


def extract_body_text(body: dict[str, Any]) -> str | None:
    if isinstance(body.get("utf8_text"), str):
        return body["utf8_text"]
    if not isinstance(body.get("base64"), str):
        return None
    try:
        return base64.b64decode(body["base64"]).decode("utf-8", errors="replace")
    except Exception:
        return None


def sanitize_body(body: Any, counts: dict[str, int]) -> dict[str, Any]:
    if not isinstance(body, dict):
        bump(counts, "body_removed")
        return {"content_redacted": True}
    out = {
        key: value
        for key, value in body.items()
        if key not in {"base64", "utf8_text", "sha256_full", "sha256_captured"}
    }
    text = extract_body_text(body)
    if text is None:
        out.update({"content_redacted": True, "redaction_reason": "non-text body bytes removed"})
        bump(counts, "body_removed")
        return out
    sanitized = redact_payload_text(text, counts)
    if sanitized != text:
        bump(counts, "body_text_redacted")
    out.update(
        {
            "encoding": "utf-8",
            "utf8_text": sanitized,
            "size_bytes": len(sanitized.encode("utf-8")),
            "captured_bytes": len(sanitized.encode("utf-8")),
            "content_redacted": sanitized != text,
        }
    )
    return out


def sanitize_message(message: Any, counts: dict[str, int]) -> Any:
    if not isinstance(message, dict):
        return message
    out = dict(message)
    out["headers"] = sanitize_pair_list(message.get("headers"), SENSITIVE_HEADERS, "sensitive_header", counts)
    if "query" in out:
        out["query"] = sanitize_pair_list(message.get("query"), SENSITIVE_QUERY, "sensitive_query", counts)
    if isinstance(out.get("path"), str):
        out["path"] = redact_text(out["path"], counts)
    if isinstance(out.get("url"), str):
        out["url"] = sanitize_url(out["url"], counts)
    if "body" in out:
        out["body"] = sanitize_body(out["body"], counts)
    return out


def sanitize_record(record: dict[str, Any], counts: dict[str, int]) -> dict[str, Any]:
    out = dict(record)
    if isinstance(out.get("server_conn"), dict):
        out["server_conn"] = dict(out["server_conn"])
        for key in ("address", "ip_address"):
            if out["server_conn"].get(key) not in (None, REDACTED):
                out["server_conn"][key] = REDACTED
                bump(counts, "server_address")
    for key in ("request", "response"):
        if key in out:
            out[key] = sanitize_message(out[key], counts)
    if isinstance(out.get("error"), dict) and isinstance(out["error"].get("msg"), str):
        out["error"] = dict(out["error"])
        out["error"]["msg"] = redact_text(out["error"]["msg"], counts)
    return out


def sanitize_stream(fin: Any, fout: Any) -> tuple[int, int]:
    counts: dict[str, int] = {}
    records = invalid = 0
    for line in fin:
        if not line.strip():
            continue
        records += 1
        try:
            item = json.loads(line)
        except Exception:
            invalid += 1
            continue
        clean = sanitize_record(item, counts) if isinstance(item, dict) else item
        fout.write(json.dumps(clean, ensure_ascii=True) + "\n")
    return records, invalid


def main() -> int:
    sanitize_stream(sys.stdin, sys.stdout)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
