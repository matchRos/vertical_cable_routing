from __future__ import annotations

from typing import Any


def create_cable_tracer() -> Any:
    from handloom_runtime.handloom_pipeline.single_tracer import CableTracer

    return CableTracer()
