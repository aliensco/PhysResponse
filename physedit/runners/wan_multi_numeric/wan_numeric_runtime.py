"""Per-process, per-sample Wave01 numeric conditioning state.

The run-local train entrypoint sets this immediately before pipeline execution.
The text projection consumes it once or during checkpoint recomputation for the
same sample. Prompt SHA and lengths are checked fail-closed.
"""
from __future__ import annotations

import hashlib
import math
import re
import threading

from .multi_numeric import (
    NUMBER,
    ParsedParameter,
    bind_token_offsets,
    flatten_bindings,
    parse_prompt,
    prompt_sha256,
    wan_whitespace_clean,
)


_state = threading.local()


def parse_supported_prompt(cleaned: str) -> list[ParsedParameter]:
    """Keep the trained parser and accept explicit PhyResponse compact facts.

    PhyResponse renders `Gravity 3.71 m/s^2.` rather than the training form
    `Gravity is 3.71 m/s^2.`. Both use the identical type IDs, scalar
    normalization, UMT5 offsets, and Harmonic injection after parsing.
    """
    found = parse_prompt(cleaned)
    aliases = (
        ("gravity", 0, "m/s^2", 25.0, 25.0,
         rf"\bGravity\s+(?P<value>{NUMBER})\s*m/s(?:\^?2|²)\b"),
        ("friction_coefficient", 1, "dimensionless", 1.0, 1.0,
         rf"\bfriction\s+(?P<value>{NUMBER})(?=\s*[.,;]|$)"),
        ("initial_speed", 2, "m/s", 6.0, 6.0,
         rf"\binitial\s+speed\s+(?P<value>{NUMBER})\s*m/s\b"),
    )
    for name, type_id, unit, scale, maximum, pattern in aliases:
        for match in re.finditer(pattern, cleaned, flags=re.IGNORECASE):
            start, end = match.span("value")
            if any(start < prior.char_end and end > prior.char_start for prior in found):
                continue
            value = float(match.group("value"))
            if not math.isfinite(value) or not 0 <= value <= maximum:
                raise ValueError(f"{name} value outside trusted range [0, {maximum}]")
            found.append(ParsedParameter(name, type_id, unit, value, value / scale,
                                         start, end, match.group(0)))
    vector = rf"\binitial\s+velocity\s+(?P<value>\[\s*{NUMBER}\s*,\s*{NUMBER}\s*,\s*{NUMBER}\s*\])\s*m/s\b"
    for match in re.finditer(vector, cleaned, flags=re.IGNORECASE):
        start, end = match.span("value")
        if any(start < prior.char_end and end > prior.char_start for prior in found):
            continue
        components = [float(item) for item in match.group("value")[1:-1].split(",")]
        speed = math.sqrt(sum(item * item for item in components))
        if not math.isfinite(speed) or speed > 6.0:
            raise ValueError("initial velocity magnitude outside trusted speed range [0, 6]")
        found.append(ParsedParameter("initial_speed", 2, "m/s", speed, speed / 6.0,
                                     start, end, match.group(0)))
    return sorted(found, key=lambda item: (item.char_start, item.char_end, item.type_id))


def set_current_from_data(data) -> None:
    prompt = data["prompt"]
    if isinstance(prompt, (list, tuple)):
        if len(prompt) != 1:
            raise RuntimeError("numeric runtime requires per-device batch size one")
        prompt = prompt[0]
    expected = data["numeric_prompt_sha256"]
    if isinstance(expected, (list, tuple)):
        expected = expected[0]
    actual = hashlib.sha256(prompt.encode()).hexdigest()
    if actual != expected:
        raise RuntimeError("numeric prompt identity mismatch")
    expected_clean = data["numeric_clean_prompt_sha256"]
    if isinstance(expected_clean, (list, tuple)):
        expected_clean = expected_clean[0]
    actual_clean = prompt_sha256(wan_whitespace_clean(prompt))
    if actual_clean != expected_clean:
        raise RuntimeError("numeric cleaned-prompt identity mismatch")
    values = list(data["numeric_values"])
    type_ids = list(data["numeric_type_ids"])
    token_indices = list(data["numeric_token_indices"])
    names = list(data["numeric_parameter_names"])
    if not values or not (len(values) == len(type_ids) == len(token_indices) == len(names)):
        raise RuntimeError("invalid numeric sidecar lengths")
    expected_types = {
        "gravity": 0,
        "friction_coefficient": 1,
        "initial_speed": 2,
    }
    for name, type_id, value, token_index in zip(names, type_ids, values, token_indices):
        if name not in expected_types or int(type_id) != expected_types[name]:
            raise RuntimeError("numeric parameter name/type mismatch")
        if not 0.0 <= float(value) <= 1.0:
            raise RuntimeError("normalized numeric value is outside [0, 1]")
        if int(token_index) < 0 or int(token_index) >= 512:
            raise RuntimeError("numeric token index is outside the Wan text window")
    _state.current = {
        "values": [float(x) for x in values],
        "type_ids": [int(x) for x in type_ids],
        "token_indices": [int(x) for x in token_indices],
        "names": names,
        "prompt_sha256": actual,
        "clean_prompt_sha256": actual_clean,
    }


def get_current():
    current = getattr(_state, "current", None)
    if current is None:
        raise RuntimeError("numeric sample sidecar was not set before model forward")
    return current


def set_current_from_prompt(prompt: str, tokenizer, max_tokens: int = 512) -> dict:
    """Inference path: derive the exact same bindings without a sidecar."""
    cleaned = wan_whitespace_clean(prompt)
    encoded = tokenizer(
        cleaned,
        add_special_tokens=True,
        truncation=False,
        return_attention_mask=False,
        return_offsets_mapping=True,
    )
    if len(encoded["input_ids"]) > max_tokens:
        raise ValueError("prompt exceeds the Wan text-token limit; refusing truncation")
    parsed = parse_supported_prompt(cleaned)
    if not parsed:
        raise ValueError("prompt contains no reliable supported physical parameter")
    bindings = bind_token_offsets(
        parsed,
        encoded["offset_mapping"],
        max_tokens,
        tokenizer.convert_ids_to_tokens(encoded["input_ids"]),
    )
    flat = flatten_bindings(bindings)
    _state.current = {
        "values": flat["numeric_values"],
        "type_ids": flat["numeric_type_ids"],
        "token_indices": flat["numeric_token_indices"],
        "names": flat["numeric_parameter_names"],
        "prompt_sha256": prompt_sha256(prompt),
        "clean_prompt_sha256": prompt_sha256(cleaned),
    }
    return {
        "prompt_sha256": _state.current["prompt_sha256"],
        "clean_prompt_sha256": _state.current["clean_prompt_sha256"],
        "token_length": len(encoded["input_ids"]),
        "bindings": bindings,
        **flat,
    }
