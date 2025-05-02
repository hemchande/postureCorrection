# import json
# import matplotlib.pyplot as plt
# import numpy as np
# from collections import defaultdict
# import random

# # ======== Load Data ========
# with open("pose_coco_feedback_dataset_normalized.json", "r") as f:
#     pose_data = json.load(f)

# # ======== COCO Keypoint Connections ========
# COCO_CONNECTIONS = [
#     (5, 7), (7, 9),    # Left arm
#     (6, 8), (8, 10),   # Right arm
#     (11, 13), (13, 15),# Left leg
#     (12, 14), (14, 16),# Right leg
#     (5, 6),            # Shoulders
#     (11, 12),          # Hips
#     (5, 11), (6, 12),  # Torso sides
#     (0, 1), (1, 3),    # Nose to eyes/ears
#     (0, 2), (2, 4)     # Nose to eyes/ears
# ]

# # ======== Normalize Function ========
# def normalize_coco_pose(pose_vector):
#     coords = [(pose_vector[i], pose_vector[i + 1]) for i in range(0, len(pose_vector), 2)]
    
#     # Midpoint of hips
#     if coords[11][0] != 0 and coords[12][0] != 0:
#         mid_hip_x = (coords[11][0] + coords[12][0]) / 2
#         mid_hip_y = (coords[11][1] + coords[12][1]) / 2
#     else:
#         mid_hip_x, mid_hip_y = coords[0]  # fallback to nose
    
#     coords = [(x - mid_hip_x, y - mid_hip_y) for (x, y) in coords]

#     # Shoulder-hip scale
#     def dist(p1, p2):
#         return np.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)
#     scale = dist(coords[5], coords[11]) + dist(coords[6], coords[12])
#     if scale == 0:
#         scale = 1.0
#     coords = [(x / scale, -y / scale) for (x, y) in coords]  # flip y for upright
#     return coords

# # ======== Group Samples by Label ========
# label_groups = defaultdict(list)
# for sample in pose_data:
#     if "pose" in sample and "label" in sample:
#         label_groups[sample["label"]].append(sample["pose"])

# # ======== Plot Grid ========
# samples_per_label = 4
# num_labels = len(label_groups)
# fig, axs = plt.subplots(num_labels, samples_per_label, figsize=(samples_per_label * 3, num_labels * 3))
# fig.suptitle("Normalized COCO Pose Skeletons by Label", fontsize=16)

# if num_labels == 1:
#     axs = [axs]

# for row_idx, (label, poses) in enumerate(label_groups.items()):
#     examples = random.sample(poses, min(len(poses), samples_per_label))
    
#     for col_idx, pose_vector in enumerate(examples):
#         coords = normalize_coco_pose(pose_vector)
#         ax = axs[row_idx][col_idx] if num_labels > 1 else axs[col_idx]

#         for (i, j) in COCO_CONNECTIONS:
#             if i < len(coords) and j < len(coords):
#                 x1, y1 = coords[i]
#                 x2, y2 = coords[j]
#                 ax.plot([x1, x2], [y1, y2], 'k-', linewidth=2)

#         x_vals = [x for x, y in coords]
#         y_vals = [y for x, y in coords]
#         ax.scatter(x_vals, y_vals, c='r', s=10)
#         ax.axis("equal")
#         ax.axis("off")
#         if col_idx == 0:
#             ax.set_title(label, fontsize=10)

# plt.tight_layout()
# plt.subplots_adjust(top=0.93)
# plt.show()

import json
from collections import Counter
import matplotlib.pyplot as plt

# Load your dataset
with open("pose_coco_feedback_dataset_3.json", "r") as f:
    data = json.load(f)

# Count occurrences of each posture label
label_counts = Counter(entry["label"] for entry in data)

# Print distribution
print("📊 Pose Label Distribution:")
for label, count in label_counts.items():
    print(f"{label}: {count}")

# Optional: Plot as a bar chart
plt.figure(figsize=(10, 5))
plt.bar(label_counts.keys(), label_counts.values())
plt.xticks(rotation=45, ha="right")
plt.ylabel("Count")
plt.title("Distribution of Pose Labels")
plt.tight_layout()
plt.show()
