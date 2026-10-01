"""Small helpers shared by the command-line scripts."""
import sys
from dataclasses import fields

import torch


def parse_overrides(pairs, cls):
    """['lr=1e-4', 'credit_after_death=false'] -> {'lr': 0.0001, 'credit_after_death': False}"""
    out = {}
    types = {f.name: f.type for f in fields(cls)}
    for p in pairs or []:
        k, v = p.split("=", 1)
        if k not in types:
            sys.exit(f"unknown {cls.__name__} field {k!r}; valid: {sorted(types)}")
        t = types[k]
        t = t if isinstance(t, type) else {"int": int, "float": float, "bool": bool}[t]
        if t is bool:
            out[k] = v.lower() in ("1", "true", "yes")
        elif t is int:
            out[k] = int(float(v))
        else:
            out[k] = t(v)
    return out


def pick_device(name="auto"):
    if name != "auto":
        return name
    return "cuda" if torch.cuda.is_available() else "cpu"
