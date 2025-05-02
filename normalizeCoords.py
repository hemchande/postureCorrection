import json
import numpy as np

def normalize_pose_centered(pose, center_ids=(11, 12), scale=True):
    """
    Normalizes a flat keypoint list [(x0,y0,x1,y1,...)] around a center joint.
    Optionally scale to make it size-invariant.
    """
    keypoints = np.array(pose).reshape(-1, 2)  # shape: (17, 2)
    
    if any(idx >= len(keypoints) for idx in center_ids):
        return pose  # skip if bad data

    # Step 1: Compute center from chosen joints
    center = keypoints[list(center_ids)].mean(axis=0)

    # Step 2: Translate to center at origin
    centered = keypoints - center

    # Step 3: Normalize scale (optional)
    if scale:
        distances = np.linalg.norm(centered, axis=1)
        scale_factor = np.max(distances) if np.max(distances) > 0 else 1.0
        centered /= scale_factor

    return centered.flatten().tolist()

def normalize_pose_file(input_path, output_path, center_ids=(11, 12)):
    with open(input_path, "r") as f:
        data = json.load(f)

    for entry in data:
        entry["pose"] = normalize_pose_centered(entry["pose"], center_ids)

    with open(output_path, "w") as f:
        json.dump(data, f, indent=2)

    print(f"✅ Normalized {len(data)} poses and saved to {output_path}")


# Example usage
normalize_pose_file("COCO-data-processed/pose_coco_feedback_dataset.json", "pose_coco_feedback_dataset_normalized.json")
