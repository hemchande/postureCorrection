from multiprocessing import Pool, cpu_count
import torch
import numpy as np
import json
import cv2
import os
from tqdm import tqdm
from PIL import Image
from transformers import pipeline
import openai
from openai import OpenAI
import torch.nn.functional as F
import math


client = OpenAI(api_key="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")

openai.api_key = "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"







# File paths
coco_image_dir = "../Downloads/train2014"
coco_ann_file = "../Downloads/annotations/person_keypoints_train2014.json"
output_file = "pose_coco_feedback_dataset_parallel.json"
SAVE_EVERY = 10

# Load COCO
with open(coco_ann_file, 'r') as f:
    coco = json.load(f)

annotations = [ann for ann in coco['annotations'] if ann['num_keypoints'] > 0]
id_to_file = {img['id']: img['file_name'] for img in coco['images']}

# Utilities
def angle_between(p1, p2, p3):
    a, b, c = map(torch.tensor, [p1, p2, p3])
    ba = a - b
    bc = c - b
    cos_angle = torch.nn.functional.cosine_similarity(ba.unsqueeze(0), bc.unsqueeze(0), dim=1).clamp(-1.0, 1.0)
    return torch.acos(cos_angle).item() * (180 / torch.pi)

def infer_posture_label(keypoints):
    kp = {i: keypoints[i] for i in range(len(keypoints)) if not (keypoints[i][0] == 0 and keypoints[i][1] == 0)}
    labels = []

    if 5 in kp and 7 in kp and kp[7][1] - kp[5][1] < 10:
        labels.append("slouching shoulders")
    if 6 in kp and 8 in kp and kp[8][1] - kp[6][1] < 10:
        labels.append("drooping right shoulder")
    if 5 in kp and 6 in kp and abs(kp[5][1] - kp[6][1]) > 20:
        labels.append("shoulder imbalance")
    if 5 in kp and 11 in kp and abs(kp[5][1] - kp[11][1]) < 30:
        labels.append("arched back")
    if 6 in kp and 12 in kp and abs(kp[6][1] - kp[12][1]) < 30:
        labels.append("arched back")
    if 5 in kp and 6 in kp and 11 in kp and 12 in kp:
        spine_angle = angle_between(kp[5], [(kp[5][0]+kp[6][0])/2, (kp[5][1]+kp[6][1])/2], kp[11])
        if spine_angle < 150:
            labels.append("rounded upper back")
    if 11 in kp and 12 in kp and abs(kp[11][1] - kp[12][1]) > 20:
        labels.append("hip tilt")
    if 13 in kp and 15 in kp and abs(kp[13][0] - kp[15][0]) > 40:
        labels.append("bowed left leg")
    if 14 in kp and 16 in kp and abs(kp[14][0] - kp[16][0]) > 40:
        labels.append("bowed right leg")
    if 11 in kp and 13 in kp and kp[13][1] < kp[11][1]:
        labels.append("hyperextended left knee")
    if 12 in kp and 14 in kp and kp[14][1] < kp[12][1]:
        labels.append("hyperextended right knee")
    return labels if labels else ["good posture"]

def estimate_severity(label, keypoints):
    if label == "slouching shoulders" and 5 in keypoints and 7 in keypoints:
        return min(1.0, abs(keypoints[7][1] - keypoints[5][1]) / 40)
    elif label == "shoulder imbalance" and 5 in keypoints and 6 in keypoints:
        return min(1.0, abs(keypoints[5][1] - keypoints[6][1]) / 40)
    return 0.5

def severity_descriptor(score):
    return "mild" if score < 0.3 else "moderate" if score < 0.7 else "severe"

def generate_feedback(label, severity, img_path):
    with open(img_path, "rb") as img_file:
        base64_image = base64.b64encode(img_file.read()).decode("utf-8")
    image_data_url = f"data:image/png;base64,{base64_image}"
    
    prompt = {
        "role": "user",
        "content": [
            {"type": "text", "text": f"A person is showing the posture issue '{label}' with {severity} severity. Provide concise corrective feedback (max 250 tokens)."},
            # You may include vision when supported in OpenAI API call
        ]
    }

    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "You're a posture correction coach."},
                prompt
            ],
            max_tokens=250,
            temperature=0.7,
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"[Error generating feedback for {label}]"

# Worker function
def process_annotation(ann):
    try:
        img_path = os.path.join(coco_image_dir, id_to_file[ann['image_id']])
        if not os.path.exists(img_path):
            return []
        keypoints = np.array(ann['keypoints']).reshape(-1, 3)[:, :2]
        labels = infer_posture_label(keypoints)

        results = []
        for label in labels:
            severity_score = estimate_severity(label, keypoints)
            severity_label = severity_descriptor(severity_score)
            feedback = generate_feedback(label, severity_label, img_path)
            flat_pose = keypoints.flatten().tolist()

            results.append({
                "pose": flat_pose,
                "label": label,
                "severity_score": severity_score,
                "severity_label": severity_label,
                "feedback": feedback
            })
        return results
    except Exception as e:
        return []

# Load existing (resume-safe)
try:
    with open(output_file, "r") as f:
        data = json.load(f)
        seen = {(tuple(d["pose"]), d["label"]) for d in data}
except:
    data = []
    seen = set()

# Multiprocessing setup
BATCH_SIZE = SAVE_EVERY
with Pool(processes=cpu_count()) as pool:
    buffer = []
    for result_list in tqdm(pool.imap_unordered(process_annotation, annotations), total=len(annotations)):
        for result in result_list:
            key = (tuple(result["pose"]), result["label"])
            if key not in seen:
                buffer.append(result)
                seen.add(key)

        if len(buffer) >= BATCH_SIZE:
            data.extend(buffer)
            with open(output_file, "w") as f:
                json.dump(data, f, indent=2)
            buffer.clear()
            print(f"✅ Autosaved {len(data)} total examples")

# Final flush
if buffer:
    data.extend(buffer)
    with open(output_file, "w") as f:
        json.dump(data, f, indent=2)
    print(f"✅ Final save with {len(data)} total examples")