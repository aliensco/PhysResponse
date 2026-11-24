"""Shared PhysEdit numeric parsing and UMT5 binding rules.

The same module is intended to be imported by manifest construction, training
validation, and inference.  It deliberately accepts only explicit physical
declarations with a unit (except dimensionless friction coefficients).
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import html
import hashlib
import re
from typing import Iterable

import ftfy


NUMBER = r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?"


@dataclass(frozen=True)
class ParameterRule:
    name: str
    type_id: int
    unit: str
    normalization_scale: float
    pattern: str
    minimum: float
    maximum: float


RULES = (
    ParameterRule(
        name="gravity",
        type_id=0,
        unit="m/s^2",
        normalization_scale=25.0,
        pattern=rf"\bGravity\s+is\s+(?P<value>{NUMBER})\s*m/s(?:\^?2|²)\b",
        minimum=0.0,
        maximum=25.0,
    ),
    ParameterRule(
        name="friction_coefficient",
        type_id=1,
        unit="dimensionless",
        normalization_scale=1.0,
        pattern=rf"\bfriction\s+coefficient\s+(?:is|of)\s+(?P<value>{NUMBER})\b",
        minimum=0.0,
        maximum=1.0,
    ),
    ParameterRule(
        name="initial_speed",
        type_id=2,
        unit="m/s",
        normalization_scale=6.0,
        pattern=rf"\binitial\s+speed\s+(?:is|of)\s+(?P<value>{NUMBER})\s*m/s\b",
        minimum=0.0,
        maximum=6.0,
    ),
)


@dataclass(frozen=True)
class ParsedParameter:
    name: str
    type_id: int
    unit: str
    raw_value: float
    normalized_value: float
    char_start: int
    char_end: int
    matched_text: str


def schema_dict() -> dict:
    return {
        "version": "physedit_multi_numeric_v3",
        "rules": [asdict(rule) for rule in RULES],
        "sign_convention": {
            "gravity": "non-negative scalar magnitude in m/s^2",
            "friction_coefficient": "non-negative dimensionless scalar",
            "initial_speed": "non-negative scalar speed magnitude in m/s; direction remains textual",
        },
    }


def wan_whitespace_clean(text: str) -> str:
    """Literal DiffSynth Wan `clean='whitespace'` preprocessing."""
    text = ftfy.fix_text(text)
    text = html.unescape(html.unescape(text))
    text = text.strip()
    return re.sub(r"\s+", " ", text).strip()


def parse_prompt(prompt: str) -> list[ParsedParameter]:
    found: list[ParsedParameter] = []
    occupied: set[tuple[int, int, int]] = set()
    for rule in RULES:
        for match in re.finditer(rule.pattern, prompt, flags=re.IGNORECASE):
            start, end = match.span("value")
            key = (rule.type_id, start, end)
            if key in occupied:
                continue
            value = float(match.group("value"))
            if not rule.minimum <= value <= rule.maximum:
                raise ValueError(
                    f"{rule.name} value {value} outside trusted range "
                    f"[{rule.minimum}, {rule.maximum}]"
                )
            found.append(
                ParsedParameter(
                    name=rule.name,
                    type_id=rule.type_id,
                    unit=rule.unit,
                    raw_value=value,
                    normalized_value=value / rule.normalization_scale,
                    char_start=start,
                    char_end=end,
                    matched_text=match.group(0),
                )
            )
            occupied.add(key)
    return sorted(found, key=lambda item: (item.char_start, item.char_end, item.type_id))


def bind_token_offsets(
    parsed: Iterable[ParsedParameter], offsets: Iterable[tuple[int, int]], max_tokens: int,
    token_strings: Iterable[str] | None = None,
) -> list[dict]:
    offset_list = [(int(a), int(b)) for a, b in offsets]
    tokens = list(token_strings) if token_strings is not None else None
    if tokens is not None and len(tokens) != len(offset_list):
        raise ValueError("token string and offset lengths differ")
    if len(offset_list) > max_tokens:
        raise ValueError(f"prompt has {len(offset_list)} tokens, exceeds {max_tokens}")
    bindings: list[dict] = []
    claimed: dict[int, int] = {}
    for parameter in parsed:
        indices = [
            index
            for index, (start, end) in enumerate(offset_list)
            if (end > start and start < parameter.char_end and end > parameter.char_start
                and (tokens is None or tokens[index] not in {"▁", "Ġ"}))
        ]
        if not indices:
            raise ValueError(
                f"no tokenizer offsets overlap {parameter.name} span "
                f"[{parameter.char_start}, {parameter.char_end})"
            )
        for index in indices:
            previous = claimed.get(index)
            if previous is not None and previous != parameter.type_id:
                raise ValueError(f"token {index} is claimed by two parameter types")
            claimed[index] = parameter.type_id
        item = asdict(parameter)
        item["token_indices"] = indices
        bindings.append(item)
    return bindings


def flatten_bindings(bindings: Iterable[dict]) -> dict:
    values: list[float] = []
    type_ids: list[int] = []
    token_indices: list[int] = []
    names: list[str] = []
    for binding in bindings:
        for token_index in binding["token_indices"]:
            values.append(float(binding["normalized_value"]))
            type_ids.append(int(binding["type_id"]))
            token_indices.append(int(token_index))
            names.append(str(binding["name"]))
    if not values or not (len(values) == len(type_ids) == len(token_indices) == len(names)):
        raise ValueError("invalid flattened numeric bindings")
    if token_indices != sorted(token_indices) or len(set(token_indices)) != len(token_indices):
        raise ValueError("numeric token indices must be unique and ordered")
    return {
        "numeric_values": values,
        "numeric_type_ids": type_ids,
        "numeric_token_indices": token_indices,
        "numeric_parameter_names": names,
    }


def prompt_sha256(prompt: str) -> str:
    return hashlib.sha256(prompt.encode("utf-8")).hexdigest()
