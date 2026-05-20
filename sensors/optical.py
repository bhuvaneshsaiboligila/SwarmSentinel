import numpy as np
from simulator.swarm import Swarm


class OpticalSensor:
    """
    Simulates an optical/camera sensor returning bounding-box detections.

    Returns a list of dicts:
        {drone_id: int | None, bbox: [x, y, w, h], confidence: float}

    drone_id is None for false positive detections.
    """

    def __init__(
        self,
        swarm: Swarm,
        pos_sigma: float = 2.0,
        half_size: float = 5.0,
        miss_prob: float = 0.1,
        fp_rate: float = 0.1,
        area: tuple = (100, 100),
    ):
        self.swarm = swarm
        self.pos_sigma = pos_sigma
        self.half_size = half_size
        self.miss_prob = miss_prob
        self.fp_rate = fp_rate
        self.area = np.asarray(area, dtype=float)

    def sense(self) -> list:
        alive = [d for d in self.swarm.drones if d.alive]
        n = len(alive)
        detections = []

        if n:
            # ── True detections ────────────────────────────────────────────────
            positions   = np.array([d.position  for d in alive])   # (n, 2)
            drone_ids   = np.array([d.drone_id  for d in alive])   # (n,)
            keep        = np.random.uniform(size=n) > self.miss_prob
            noisy_pos   = positions + np.random.normal(0.0, self.pos_sigma, (n, 2))
            confidences = np.random.uniform(0.5, 1.0, size=n)

            for pos, did, conf in zip(noisy_pos[keep], drone_ids[keep], confidences[keep]):
                x1, y1 = pos[0] - self.half_size, pos[1] - self.half_size
                x2, y2 = pos[0] + self.half_size, pos[1] + self.half_size
                x1, x2 = np.clip([x1, x2], 0, self.area[0])
                y1, y2 = np.clip([y1, y2], 0, self.area[1])
                detections.append({
                    "drone_id":   int(did),
                    "bbox":       [x1, y1, x2 - x1, y2 - y1],
                    "confidence": float(conf),
                })

            # ── False positives ────────────────────────────────────────────────
            n_fp = int(np.random.binomial(n, self.fp_rate))
            if n_fp:
                fp_pos   = np.random.uniform([0.0, 0.0], self.area, size=(n_fp, 2))
                fp_conf  = np.random.uniform(0.1, 0.5, size=n_fp)
                for pos, conf in zip(fp_pos, fp_conf):
                    x1, y1 = pos[0] - self.half_size, pos[1] - self.half_size
                    x2, y2 = pos[0] + self.half_size, pos[1] + self.half_size
                    x1, x2 = np.clip([x1, x2], 0, self.area[0])
                    y1, y2 = np.clip([y1, y2], 0, self.area[1])
                    detections.append({
                        "drone_id":   None,
                        "bbox":       [x1, y1, x2 - x1, y2 - y1],
                        "confidence": float(conf),
                    })

        return detections
