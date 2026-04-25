from typing import Dict, List, Optional

from cable_core.board_service import BoardService
from cable_orchestrator.base_step import BaseStep


class PrepareRoutingStep(BaseStep):
    name = "prepare_routing"
    description = "Load clips, define routing, and build a routing overlay."

    def __init__(self, routing: Optional[List[int]] = None) -> None:
        super().__init__()
        self.board_service = BoardService()
        self.default_routing = routing

    def _resolve_routing(self, state) -> List[int]:
        if state.routing is not None:
            return list(state.routing)
        if self.default_routing is not None:
            return list(self.default_routing)
        if state.config is not None and hasattr(state.config, "default_routing"):
            return list(state.config.default_routing)
        raise RuntimeError("No routing available.")

    def run(self, state) -> Dict[str, object]:
        if state.env is None:
            raise RuntimeError("Debug context not initialized. Run init_environment first.")
        if state.env.board is None:
            raise RuntimeError("Board is not available in debug context.")

        routing = self._resolve_routing(state)
        debug_data = self.board_service.prepare_routing_debug_data(
            board=state.env.board,
            routing=routing,
            image_width=state.config.fallback_image_width,
            image_height=state.config.fallback_image_height,
        )

        state.routing = routing
        state.clips = debug_data["clips"]
        state.crossing_fixture_id_list = debug_data["crossing_fixture_id_list"]
        state.routing_overlay = debug_data["routing_overlay"]

        return {
            "routing": routing,
            "num_clips": debug_data["num_clips"],
            "clip_ids": debug_data["clip_ids"],
            "num_crossing_fixtures": len(debug_data["crossing_fixture_id_list"]),
        }
