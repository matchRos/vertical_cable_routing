from __future__ import annotations

from typing import Any


def create_cable_tracer() -> Any:
    from cable_routing.handloom.handloom_pipeline.single_tracer import CableTracer

    return CableTracer()
