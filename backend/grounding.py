"""Check that every number in the agent's answer came from a tool result."""

import json
import re

_NUM = re.compile(r"(?<![\w.])[-−]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?")
_REF = re.compile(r"\[(r\d+\.\w+|compare\.\w+)\]")


def _numbers_in(obj, out: list[float]) -> None:
    if isinstance(obj, bool):
        return
    if isinstance(obj, (int, float)):
        out.append(float(obj))
    elif isinstance(obj, dict):
        for v in obj.values():
            _numbers_in(v, out)
    elif isinstance(obj, list):
        for v in obj:
            _numbers_in(v, out)
    elif isinstance(obj, str):
        out.extend(float(m.replace("−", "-")) for m in _NUM.findall(obj))


def check(answer: str, tool_results: list[dict]) -> dict:
    known: list[float] = []
    for r in tool_results:
        _numbers_in(r, known)
    refs_known = set(re.findall(r'"ref": "([^"]+)"', json.dumps(tool_results)))

    text = _REF.sub(" ", answer)
    ungrounded = []
    for raw in _NUM.findall(text):
        value = float(raw.replace("−", "-"))
        mantissa, _, exponent = raw.lower().partition("e")
        decimals = (len(mantissa.split(".")[1]) if "." in mantissa else 0) - int(exponent or 0)
        tol = 0.5 * 10.0 ** -decimals + 1e-12
        if decimals == 0 and abs(value) <= 10:
            continue  # small counts and list numbering
        if any(abs(value - k) <= tol or abs(abs(value) - abs(k)) <= tol for k in known):
            continue
        ungrounded.append(raw)
    cited = _REF.findall(answer)
    return {
        "ungrounded_numbers": ungrounded,
        "refs_cited": cited,
        "unknown_refs": [r for r in cited if r not in refs_known],
        "grounded": not ungrounded and all(r in refs_known for r in cited),
    }
