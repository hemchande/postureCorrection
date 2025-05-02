import json
import numpy as np
import os
from tqdm import tqdm

# Load existing COCO and feedback dataset
coco_image_dir = "../Downloads/train2014"
coco_ann_file = "../Downloads/annotations/person_keypoints_train2014.json"

with open(coco_ann_file, 'r') as f:
    coco = json.load(f)

annotations = [ann for ann in coco['annotations'] if ann['num_keypoints'] > 0]
id_to_file = {img['id']: img['file_name'] for img in coco['images']}

# Load feedback dataset
with open("pose_coco_feedback_dataset_3.json", "r") as f:
    dataset = json.load(f)

# Match each pose to its corresponding COCO image
for example in tqdm(dataset):
    flat_pose_example = np.round(example["pose"], 2)
    matching_ann = None

    for ann in annotations:
        keypoints = ann['keypoints']
        if len(keypoints) != 51:
            continue

        coords = np.array(keypoints).reshape(-1, 3)[:, :2].flatten()
        flat = np.round(coords, 2)

        min_len = min(len(flat), len(flat_pose_example))
        if min_len < 30:
            continue

        similarity = np.mean(np.isclose(flat[:min_len], flat_pose_example[:min_len], atol=2.0))
        if similarity == 1.0:
            matching_ann = ann
            image_id = matching_ann["image_id"]
            example["image_path"] = os.path.join(coco_image_dir, id_to_file[image_id])
            break

    if not matching_ann:
        example["image_path"] = None

# Filter out entries without image_path
dataset = [ex for ex in dataset if ex["image_path"] is not None]

# Save the new dataset with image paths
output_path = "pose_coco_feedback_dataset_4.json"
with open(output_path, "w") as f:
    json.dump(dataset, f, indent=2)

output_path
