from __future__ import annotations

import numpy as np


class CableOrientationService:
    def compute_tangents(self, path_world):
        tangents = []

        for idx in range(len(path_world)):
            if idx == 0:
                tangent = path_world[1] - path_world[0]
            elif idx == len(path_world) - 1:
                tangent = path_world[-1] - path_world[-2]
            else:
                tangent = path_world[idx + 1] - path_world[idx - 1]

            tangent = tangent / (np.linalg.norm(tangent) + 1e-8)
            tangents.append(tangent)

        return np.array(tangents)
