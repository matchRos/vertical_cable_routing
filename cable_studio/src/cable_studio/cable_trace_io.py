from pathlib import Path
from typing import Iterable

import csv
import numpy as np


class CableTraceIO:
    def save_csv(self, filepath: str, path_in_pixels: Iterable) -> None:
        path = Path(filepath)
        path.parent.mkdir(parents=True, exist_ok=True)

        pts = np.asarray(path_in_pixels, dtype=float)
        if pts.ndim != 2 or pts.shape[1] < 2:
            raise RuntimeError("path_in_pixels must be an Nx2 array-like object.")

        with path.open("w", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(["x", "y"])
            for pt in pts:
                writer.writerow([float(pt[0]), float(pt[1])])

    def load_csv(self, filepath: str) -> np.ndarray:
        path = Path(filepath)
        if not path.exists():
            raise RuntimeError(f"Trace file does not exist: {filepath}")

        with path.open("r", newline="") as handle:
            rows = list(csv.reader(handle))

        if not rows:
            raise RuntimeError("Trace CSV is empty.")

        start_idx = 0
        first = [cell.strip().lower() for cell in rows[0]]
        if len(first) >= 2 and first[0] == "x" and first[1] == "y":
            start_idx = 1

        pts = []
        for row in rows[start_idx:]:
            if len(row) < 2:
                continue
            pts.append([float(row[0]), float(row[1])])

        if len(pts) < 2:
            raise RuntimeError("Loaded trace has too few points.")

        return np.asarray(pts, dtype=float)
