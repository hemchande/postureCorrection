import numpy as np
import json, glob, os

def normalize_keypoints(keypoints, bbox=None, has_visibility=True):
    keypoints = np.array(keypoints)
    if keypoints.ndim == 1:
        if has_visibility:
            keypoints = keypoints.reshape(-1, 3)
        else:
            keypoints = keypoints.reshape(-1, 2)

    if has_visibility:
        vis = keypoints[:, 2] > 0
        valid_kps = keypoints[vis][:, :2]
    else:
        valid_kps = keypoints

    if bbox is None:
        x_min, y_min = valid_kps.min(axis=0)
        x_max, y_max = valid_kps.max(axis=0)
        width = x_max - x_min
        height = y_max - y_min
        center = np.array([(x_max + x_min) / 2, (y_max + y_min) / 2])
    else:
        x_min, y_min, width, height = bbox
        center = np.array([x_min + width / 2, y_min + height / 2])

    width = max(width, 1e-5)
    height = max(height, 1e-5)

    xy = keypoints[:, :2]
    norm_xy = (xy - center) / np.array([width, height]) + 0.5
    return np.clip(norm_xy, 0, 1)


# ========== Process all JSON files ==========
all_normalized_kps = []

for path in glob.glob("COCO-data-processed/*.json"):
    with open(path) as f:
        data = json.load(f)
        for sample in data:
            flat_kps = sample["pose"]
            # Auto-detect shape
            has_vis = len(flat_kps) % 3 == 0
            norm_kps = normalize_keypoints(flat_kps, has_visibility=has_vis)
            
            all_normalized_kps.append({
                "file": os.path.basename(path),
                "normalized_pose": norm_kps.tolist(),
                "label": sample.get("label", ""),          # optional
                "feedback": sample.get("feedback", "")     # optional
            })

# ========== Save to file ==========
with open("normalized_keypoints_final.json", "w") as f_out:
    json.dump(all_normalized_kps, f_out, indent=2)

print(f"✅ Saved {len(all_normalized_kps)} samples to normalized_keypoints.json")
