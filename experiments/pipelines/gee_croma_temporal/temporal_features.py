import numpy as np


def compare(before: dict, after: dict) -> dict:
    result = {}
    for name in ("SAR_GAP", "optical_GAP", "joint_GAP"):
        left = before[name].detach().cpu().numpy().reshape(-1)
        right = after[name].detach().cpu().numpy().reshape(-1)
        result[name] = {"l2_distance": float(np.linalg.norm(right - left)), "cosine_similarity": float(np.dot(left, right) / (np.linalg.norm(left) * np.linalg.norm(right)))}
    return result