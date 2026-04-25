import json
import os
from dataclasses import dataclass
from typing import Dict, List


@dataclass
class DebugClip:
    """
    Minimal standalone clip representation for routing and overlays.
    """

    clip_id: str
    x: int
    y: int
    clip_type: int
    orientation: int


class DebugBoard:
    """
    Minimal standalone board representation without legacy config dependencies.
    """

    def __init__(self, config_path: str) -> None:
        self.config_path = config_path
        self._raw_config: Dict = self._load_config(config_path)
        self._clips: List[DebugClip] = self._parse_clips(self._raw_config)

    def _load_config(self, config_path: str) -> Dict:
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Board config file not found: {config_path}")

        with open(config_path, "r", encoding="utf-8") as handle:
            data = json.load(handle)

        if not isinstance(data, dict):
            raise ValueError("Board config JSON must contain a dictionary at the top level.")

        return data

    def _parse_clips(self, raw_config: Dict) -> List[DebugClip]:
        clips: List[DebugClip] = []
        for clip_id, clip_data in raw_config.items():
            clips.append(
                DebugClip(
                    clip_id=str(clip_id),
                    x=int(clip_data["x"]),
                    y=int(clip_data["y"]),
                    clip_type=int(clip_data["type"]),
                    orientation=int(clip_data["orientation"]),
                )
            )
        return clips

    def get_clips(self) -> List[DebugClip]:
        return list(self._clips)

    def get_clip_ids(self) -> List[str]:
        return [clip.clip_id for clip in self._clips]

    def get_clip_by_index(self, index: int) -> DebugClip:
        return self._clips[index]

    def num_clips(self) -> int:
        return len(self._clips)
