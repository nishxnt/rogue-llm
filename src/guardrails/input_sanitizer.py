"""Deterministic Layer 1 input sanitizer."""

from __future__ import annotations

import base64
import hashlib
import re
import unicodedata
from dataclasses import dataclass
from typing import Final

from src.guardrails.reasons import GuardrailBlock

_TOKEN_RE: Final[re.Pattern[str]] = re.compile(r"[^\W_]+|[0-9]+", re.UNICODE)
_BASE64_RE: Final[re.Pattern[str]] = re.compile(r"(?<![A-Za-z0-9+/])[A-Za-z0-9+/]{24,}={0,2}")
_LEETSPEAK_TRANSLATION = str.maketrans(
    {
        "0": "o",
        "1": "i",
        "3": "e",
        "4": "a",
        "5": "s",
        "7": "t",
        "@": "a",
        "$": "s",
    }
)

_SUSPICIOUS_BASE64_MARKERS: Final[tuple[str, ...]] = (
    "ignore previous",
    "ignore all previous",
    "disregard system prompt",
    "begin_disclosure",
    "<<<sysprompt>>>",
    "developer mode",
    "dan",
)


@dataclass(frozen=True)
class SanitizerRule:
    """One redacted regex rule."""

    rule_id: str
    family: str
    pattern: re.Pattern[str]


DEFAULT_RULES: tuple[SanitizerRule, ...] = (
    SanitizerRule(
        rule_id="override.ignore_previous",
        family="override_phrase",
        pattern=re.compile(
            r"\b(?:ignore|disregard|forget|bypass)\b.{0,40}\b(?:previous|prior|above|system)\b",
            re.IGNORECASE | re.DOTALL,
        ),
    ),
    SanitizerRule(
        rule_id="override.follow_attacker",
        family="override_phrase",
        pattern=re.compile(
            r"\b(?:from now on|instead|do not follow|stop following)\b.{0,60}\b(?:instructions|system prompt|policy|rules)\b",
            re.IGNORECASE | re.DOTALL,
        ),
    ),
    SanitizerRule(
        rule_id="boundary.begin_disclosure",
        family="boundary_marker",
        pattern=re.compile(
            r"(?:begin[_\s-]?disclosure|begin[_\s-]?exposure|disclosure[_\s-]?start|<<<\s*(?:sysprompt|systemdata|internal)\s*>>>)",
            re.IGNORECASE,
        ),
    ),
    SanitizerRule(
        rule_id="boundary.role_channel_confusion",
        family="boundary_marker",
        pattern=re.compile(
            r"\b(?:system|developer|assistant|user)\s*:\s*(?:system|developer|assistant|user)\b",
            re.IGNORECASE,
        ),
    ),
    SanitizerRule(
        rule_id="roleplay.dan",
        family="roleplay_override",
        pattern=re.compile(
            r"\b(?:dan|developer mode|jailbreak|prompt injection mode)\b",
            re.IGNORECASE,
        ),
    ),
    SanitizerRule(
        rule_id="roleplay.act_as",
        family="roleplay_override",
        pattern=re.compile(
            r"\bact as\b.{0,40}\b(?:unfiltered|unrestricted|another ai|evil twin|system)\b",
            re.IGNORECASE | re.DOTALL,
        ),
    ),
    SanitizerRule(
        rule_id="forbidden.explicit_override",
        family="forbidden_sequence",
        pattern=re.compile(
            r"\b(?:reveal|leak|print|dump|show)\b.{0,40}\b(?:system prompt|hidden prompt|internal instructions)\b",
            re.IGNORECASE | re.DOTALL,
        ),
    ),
)


class InputSanitizer:
    """Aggressive deterministic Layer 1 prompt sanitizer."""

    def __init__(
        self,
        *,
        rules: tuple[SanitizerRule, ...] = DEFAULT_RULES,
        max_prompt_tokens: int = 2000,
    ) -> None:
        if max_prompt_tokens <= 0:
            raise ValueError("max_prompt_tokens must be positive")
        self.rules = rules
        self.max_prompt_tokens = max_prompt_tokens

    def inspect(self, prompt: str) -> GuardrailBlock | None:
        """Return a block decision when prompt matches a sanitizer rule."""
        token_count = count_prompt_tokens(prompt)
        if token_count > self.max_prompt_tokens:
            return GuardrailBlock(
                decision="blocked_l1",
                decision_layer=1,
                evidence={
                    "matched_regex_family": "length_limit",
                    "matched_rule_id": "length.prompt_token_limit",
                    "token_count": token_count,
                    "token_limit": self.max_prompt_tokens,
                },
            )

        normalized_variants = (
            ("raw", prompt),
            ("normalized", normalize_prompt(prompt)),
            ("leetspeak", normalize_prompt(prompt).translate(_LEETSPEAK_TRANSLATION)),
        )
        for variant_name, candidate in normalized_variants:
            for rule in self.rules:
                match = rule.pattern.search(candidate)
                if match is None:
                    continue
                return GuardrailBlock(
                    decision="blocked_l1",
                    decision_layer=1,
                    evidence={
                        "matched_regex_family": rule.family,
                        "matched_rule_id": rule.rule_id,
                        "token_count": token_count,
                        "match_variant": variant_name,
                        "match_hash": _span_hash(candidate[match.start() : match.end()]),
                    },
                )

        decoded_fragment = _decoded_suspicious_base64_fragment(prompt)
        if decoded_fragment is not None:
            return GuardrailBlock(
                decision="blocked_l1",
                decision_layer=1,
                evidence={
                    "matched_regex_family": "encoded_fragment",
                    "matched_rule_id": "encoded.base64_override_fragment",
                    "token_count": token_count,
                    "decoded_fragment_hash": _span_hash(decoded_fragment),
                },
            )

        return None

    def fingerprint(self) -> str:
        """Return a stable fingerprint for cache invalidation."""
        material = "|".join(
            f"{rule.rule_id}:{rule.family}:{rule.pattern.pattern}" for rule in self.rules
        )
        return hashlib.sha256(f"{self.max_prompt_tokens}|{material}".encode()).hexdigest()


def normalize_prompt(prompt: str) -> str:
    """Normalize unicode and whitespace before sanitizer matching."""
    normalized = unicodedata.normalize("NFKC", prompt)
    return re.sub(r"\s+", " ", normalized.strip().lower())


def count_prompt_tokens(prompt: str) -> int:
    """Approximate token count for Layer 1 length enforcement."""
    normalized = unicodedata.normalize("NFKC", prompt)
    return len(_TOKEN_RE.findall(normalized))


def _decoded_suspicious_base64_fragment(prompt: str) -> str | None:
    for match in _BASE64_RE.finditer(prompt):
        fragment = match.group(0)
        try:
            decoded = base64.b64decode(fragment, validate=True).decode("utf-8", errors="ignore")
        except Exception:
            continue
        normalized = normalize_prompt(decoded)
        if any(marker in normalized for marker in _SUSPICIOUS_BASE64_MARKERS):
            return decoded
    return None


def _span_hash(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()
