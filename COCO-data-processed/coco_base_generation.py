# # posture_feedback_dataset.py

# import torch
# import numpy as np
# import json
# import cv2
# import os
# from tqdm import tqdm
# from PIL import Image

# # ============================
# # 1. Load COCO Dataset Locally
# # ============================
# coco_image_dir = "../Downloads/train2014"
# coco_ann_file = "../Downloads/annotations/person_keypoints_train2014.json"

# with open(coco_ann_file, 'r') as f:
#     coco = json.load(f)

# annotations = [ann for ann in coco['annotations'] if ann['num_keypoints'] > 0]
# id_to_file = {img['id']: img['file_name'] for img in coco['images']}

# # ============================
# # 2. Define Posture Heuristic Labeling Logic
# # ============================
# def infer_posture_label(keypoints):
#     kp = {i: keypoints[i] for i in range(len(keypoints)) if keypoints[i] is not None}
#     labels = []

#     if 5 in kp and 7 in kp:  # left shoulder and elbow
#         shoulder_y = kp[5][1]
#         elbow_y = kp[7][1]
#         if elbow_y - shoulder_y < 10:
#             labels.append("slouching shoulders")

#     if 5 in kp and 11 in kp:  # left shoulder and left hip
#         shoulder_y = kp[5][1]
#         hip_y = kp[11][1]
#         if abs(shoulder_y - hip_y) < 30:
#             labels.append("arched back")

#     if 13 in kp and 15 in kp:  # left knee and ankle
#         knee_x = kp[13][0]
#         ankle_x = kp[15][0]
#         if abs(knee_x - ankle_x) > 40:
#             labels.append("bowed legs")

#     if 6 in kp and 8 in kp:  # right shoulder and elbow
#         shoulder_y = kp[6][1]
#         elbow_y = kp[8][1]
#         if elbow_y - shoulder_y < 10:
#             labels.append("drooping right shoulder")

#     if 11 in kp and 13 in kp:  # left hip and knee
#         hip_y = kp[11][1]
#         knee_y = kp[13][1]
#         if knee_y < hip_y:
#             labels.append("hyperextended left knee")

#     if 12 in kp and 14 in kp:  # right hip and knee
#         hip_y = kp[12][1]
#         knee_y = kp[14][1]
#         if knee_y < hip_y:
#             labels.append("hyperextended right knee")

#     if 5 in kp and 6 in kp:  # shoulder alignment
#         left_shoulder_y = kp[5][1]
#         right_shoulder_y = kp[6][1]
#         if abs(left_shoulder_y - right_shoulder_y) > 20:
#             labels.append("shoulder imbalance")

#     if 11 in kp and 12 in kp:  # hip alignment
#         left_hip_y = kp[11][1]
#         right_hip_y = kp[12][1]
#         if abs(left_hip_y - right_hip_y) > 20:
#             labels.append("hip tilt")

#     return labels if labels else ["good posture"]

# # Feedback mapping
# FEEDBACK_MAP = {
#     "slouching shoulders": "Straighten your back and elongate your spine.",
#     "arched back": "Engage your core and tuck your pelvis.",
#     "bowed legs": "Keep your legs aligned with your hips and knees.",
#     "drooping right shoulder": "Lift your right shoulder to balance posture.",
#     "hyperextended left knee": "Soften your left knee and avoid locking it.",
#     "hyperextended right knee": "Soften your right knee and avoid locking it.",
#     "shoulder imbalance": "Align both shoulders evenly to distribute weight.",
#     "hip tilt": "Even out your hips to improve stability and alignment."
# }

# # ============================
# # 3. Create Training Triplets (pose, label, feedback)
# # ============================
# data = []
# for ann in tqdm(annotations[:50]):  # limit to 50 for test
#     img_path = os.path.join(coco_image_dir, id_to_file[ann['image_id']])
#     if not os.path.exists(img_path):
#         continue

#     keypoints = np.array(ann['keypoints']).reshape(-1, 3)[:, :2]  # (17, 2)
#     labels = infer_posture_label(keypoints)

#     for label in labels:
#         feedback = FEEDBACK_MAP.get(label, "Maintain your posture.")
#         flat_pose = keypoints.flatten()
#         data.append((flat_pose, label, feedback))

# # ============================
# # 4. Save for Training
# # ============================
# np.savez("pose_feedback_data.npz", data=data)
# print(f"Saved {len(data)} pose-label-feedback triplets.")

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

# Load local dataset
coco_image_dir = "../Downloads/train2014"
coco_ann_file = "../Downloads/annotations/person_keypoints_train2014.json"

with open(coco_ann_file, 'r') as f:
    coco = json.load(f)

annotations = [ann for ann in coco['annotations'] if ann['num_keypoints'] > 0]
id_to_file = {img['id']: img['file_name'] for img in coco['images']}

# Load LLM feedback generator
# generator = pipeline("text-generation", model="sshleifer/tiny-gpt2")


# Estimate mistake severity
def estimate_severity(label, keypoints):
    # Example heuristics (you can refine this logic per label)
    if label == "slouching shoulders" and 5 in keypoints and 7 in keypoints:
        diff = abs(keypoints[7][1] - keypoints[5][1])
        return min(1.0, diff / 40)  # normalize
    elif label == "shoulder imbalance" and 5 in keypoints and 6 in keypoints:
        diff = abs(keypoints[5][1] - keypoints[6][1])
        return min(1.0, diff / 40)
    # Default fallback
    return 0.5

# Convert severity to human-readable token
def severity_descriptor(score):
    if score < 0.3:
        return "mild"
    elif score < 0.7:
        return "moderate"
    else:
        return "severe"

# Feedback from LLM
# def generate_feedback(label, severity_level):
#     prompt = f"Posture issue: {label}. Severity: {severity_level}. How can I correct it?"
#     output = generator(prompt)[0]["generated_text"]
#     return output.split("How can I correct it?")[-1].strip()

# feedback_cache = {}

# def generate_feedback(label, severity_level):
#     key = f"{label}_{severity_level}"
#     if key in feedback_cache:
#         return feedback_cache[key]

#     prompt = (
#         f"A dancer is exhibiting a posture issue: {label}.\n"
#         f"The severity of the mistake is {severity_level}.\n"
#         f"Write specific, constructive feedback to help them improve without repeating the issue name."
#     )
#     output = generator(prompt)[0]["generated_text"]
#     feedback = output.split("improve")[-1] if "improve" in output else output
#     feedback = feedback.strip().replace("\n", " ")
#     feedback_cache[key] = feedback
#     return feedback


# def generate_feedback(label, severity_level, image_path):
#     with open(image_path, "rb") as img_file:
#         image_data = img_file.read()

#     prompt = (
#         f"You are a posture correction coach.\n"
#         f"A dancer is exhibiting a posture issue labeled: '{label}' with {severity_level} severity.\n"
#         f"Use the attached image to guide feedback.\n"
#         f"Generate specific, constructive advice (max 250 tokens), and do not repeat the issue name."
#     )

#     response = client.chat.completions.create(
#         model="gpt-4o",
#         messages=[
#             {"role": "system", "content": "You're a posture correction coach."},
#             {"role": "user", "content": prompt}
#         ],
#         max_tokens=250,
#         temperature=0.7,
#         tools=[
#             {
#                 "type": "image",
#                 "image": image_data,
#                 "mime_type": "image/png"  # or "image/jpeg" depending on file
#             }
#         ]
#     )
#     return response.choices[0].message.content.strip()


import base64

def generate_feedback(label, severity_level, image_path):
    # Load and encode image
    with open(image_path, "rb") as img_file:
        base64_image = base64.b64encode(img_file.read()).decode("utf-8")
    
    image_data_url = f"data:image/png;base64,{base64_image}"

    # Send image + prompt to GPT-4o vision
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": "You are a posture correction coach specifically for correcting issues like slouching shoulders,dropping right shoulder,shoulder imbalance,arched back,rounded upper back,hip tilt,bowed legs,hyperextended knees,and foward head."},
            {"role": "user", "content": [
                {"type": "text", "text": f"A person engaging in some activity is showing some posture issue labeled: '{label}' with {severity_level} severity. Provide concise corrective feedback (max 250 tokens), without repeating the issue name and compilment user positively if no incorrect posture issue is mentioned or detected in image"},
                # {"type": "image_url", "image_url": {"url": image_data_url}},
            ]}
        ],
        max_tokens=250,
        temperature=0.7,
    )

    print("label feedback",response.choices[0].message.content.strip())

    return response.choices[0].message.content.strip()



# def generate_feedback(label, severity_level):
#     prompt = (
#         f"A dancer is exhibiting a posture issue: {label}.\n"
#         f"The severity is {severity_level}.\n"
#         f"Give rich feedback to correct it, without repeating the issue."
#     )
#     response = client.responses.create(
#         model="gpt-4o",
#         input=[
#             {"role": "system", "content": "You're a posture correction coach."},
#             {"role": "user", "content": prompt}
#         ]
#     )
#     return response.output_text


# import math

# def angle_between(p1, p2, p3):
#     # Computes angle at p2 formed by points (p1, p2, p3)
#     a = torch.tensor(p1)
#     b = torch.tensor(p2)
#     c = torch.tensor(p3)
#     ba = a - b
#     bc = c - b
#     cos_angle = F.cosine_similarity(ba.unsqueeze(0), bc.unsqueeze(0), dim=1).clamp(-1.0, 1.0)
#     return torch.acos(cos_angle).item() * (180 / math.pi)  # in degrees

# def infer_posture_label(keypoints):
#     kp = {i: keypoints[i] for i in range(len(keypoints)) if keypoints[i] is not None}
#     labels = []

#     # Shoulders
#     if 5 in kp and 7 in kp and kp[7][1] - kp[5][1] < 10:
#         labels.append("slouching shoulders")
#     if 6 in kp and 8 in kp and kp[8][1] - kp[6][1] < 10:
#         labels.append("drooping right shoulder")
#     if 5 in kp and 6 in kp and abs(kp[5][1] - kp[6][1]) > 20:
#         labels.append("shoulder imbalance")

#     # Spine / Back
#     if 5 in kp and 11 in kp and abs(kp[5][1] - kp[11][1]) < 30:
#         labels.append("arched back")
#     if 6 in kp and 12 in kp and abs(kp[6][1] - kp[12][1]) < 30:
#         labels.append("arched back")
#     if 5 in kp and 6 in kp and 11 in kp and 12 in kp:
#         spine_angle = angle_between(kp[5], [(kp[5][0]+kp[6][0])/2, (kp[5][1]+kp[6][1])/2], kp[11])
#         if spine_angle < 150:
#             labels.append("rounded upper back")

#     # Hips
#     if 11 in kp and 12 in kp and abs(kp[11][1] - kp[12][1]) > 20:
#         labels.append("hip tilt")

#     # Legs
#     if 13 in kp and 15 in kp and abs(kp[13][0] - kp[15][0]) > 40:
#         labels.append("bowed left leg")
#     if 14 in kp and 16 in kp and abs(kp[14][0] - kp[16][0]) > 40:
#         labels.append("bowed right leg")
#     if 11 in kp and 13 in kp and kp[13][1] < kp[11][1]:
#         labels.append("hyperextended left knee")
#     if 12 in kp and 14 in kp and kp[14][1] < kp[12][1]:
#         labels.append("hyperextended right knee")

#     # Head / Neck
#     if 0 in kp and 1 in kp and 2 in kp:
#         eye_diff = abs(kp[1][1] - kp[2][1])
#         if eye_diff > 10:
#             labels.append("head tilt")
#     if 0 in kp and 5 in kp and 6 in kp:
#         head_mid = [(kp[5][0]+kp[6][0])/2, (kp[5][1]+kp[6][1])/2]
#         neck_angle = angle_between(kp[0], head_mid, [(kp[11][0]+kp[12][0])/2, (kp[11][1]+kp[12][1])/2])
#         if neck_angle < 160:
#             labels.append("forward head posture")

#     return labels if labels else ["good posture"]



def angle_between(p1, p2, p3):
    a = torch.tensor(p1)
    b = torch.tensor(p2)
    c = torch.tensor(p3)
    ba = a - b
    bc = c - b
    cos_angle = F.cosine_similarity(ba.unsqueeze(0), bc.unsqueeze(0), dim=1).clamp(-1.0, 1.0)
    return torch.acos(cos_angle).item() * (180 / math.pi)

def infer_posture_label(keypoints):
    kp = {
        i: keypoints[i]
        for i in range(len(keypoints))
        if keypoints[i] is not None and not (keypoints[i][0] == 0 and keypoints[i][1] == 0)
    }

    labels = []

    if 5 in kp and 7 in kp and (kp[7][1] - kp[5][1] < 10):
        labels.append("slouching shoulders")
    if 6 in kp and 8 in kp and (kp[8][1] - kp[6][1] < 10):
        labels.append("drooping right shoulder")
    if 5 in kp and 6 in kp and (abs(kp[5][1] - kp[6][1]) > 20):
        labels.append("shoulder imbalance")

    if 5 in kp and 11 in kp and (abs(kp[5][1] - kp[11][1]) < 30):
        labels.append("arched back")
    if 6 in kp and 12 in kp and (abs(kp[6][1] - kp[12][1]) < 30):
        labels.append("arched back")
    if 5 in kp and 6 in kp and 11 in kp and 12 in kp:
        spine_angle = angle_between(
            kp[5],
            [(kp[5][0] + kp[6][0]) / 2, (kp[5][1] + kp[6][1]) / 2],
            kp[11]
        )
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

    if 0 in kp and 1 in kp and 2 in kp:
        if abs(kp[1][1] - kp[2][1]) > 10:
            labels.append("head tilt")
    if 0 in kp and 5 in kp and 6 in kp and 11 in kp and 12 in kp:
        head_mid = [(kp[5][0] + kp[6][0]) / 2, (kp[5][1] + kp[6][1]) / 2]
        torso_mid = [(kp[11][0] + kp[12][0]) / 2, (kp[11][1] + kp[12][1]) / 2]
        neck_angle = angle_between(kp[0], head_mid, torso_mid)
        if neck_angle < 160:
            labels.append("forward head posture")

    return labels if labels else ["good posture"]


# # Heuristic posture labeling
# def infer_posture_label(keypoints):
#     kp = {i: keypoints[i] for i in range(len(keypoints)) if keypoints[i] is not None}
#     labels = []

#     if 5 in kp and 7 in kp:
#         if kp[7][1] - kp[5][1] < 10:
#             labels.append("slouching shoulders")

#     if 5 in kp and 11 in kp:
#         if abs(kp[5][1] - kp[11][1]) < 30:
#             labels.append("arched back")

#     if 13 in kp and 15 in kp:
#         if abs(kp[13][0] - kp[15][0]) > 40:
#             labels.append("bowed legs")

#     if 6 in kp and 8 in kp:
#         if kp[8][1] - kp[6][1] < 10:
#             labels.append("drooping right shoulder")

#     if 11 in kp and 13 in kp:
#         if kp[13][1] < kp[11][1]:
#             labels.append("hyperextended left knee")

#     if 12 in kp and 14 in kp:
#         if kp[14][1] < kp[12][1]:
#             labels.append("hyperextended right knee")

#     if 5 in kp and 6 in kp:
#         if abs(kp[5][1] - kp[6][1]) > 20:
#             labels.append("shoulder imbalance")

#     if 11 in kp and 12 in kp:
#         if abs(kp[11][1] - kp[12][1]) > 20:
#             labels.append("hip tilt")

#     return labels if labels else ["good posture"]

# ============================
# 3. Create Training Triplets (pose, label, degree, feedback)
# ============================
# data = []
# for ann in tqdm(annotations[:50]):  # limit for demo
#     img_path = os.path.join(coco_image_dir, id_to_file[ann['image_id']])
#     if not os.path.exists(img_path):
#         continue

#     keypoints = np.array(ann['keypoints']).reshape(-1, 3)[:, :2]
#     labels = infer_posture_label(keypoints)

#     for label in labels:
#         severity = estimate_severity(label, keypoints)
#         severity_label = severity_descriptor(severity)
#         feedback = generate_feedback(label, severity_label)
#         flat_pose = keypoints.flatten()
#         data.append((flat_pose, label, severity, feedback))


# ============================
# 6. Generate Triplets
# ============================
data = []
for ann in tqdm(annotations):  # limit for demo
    img_path = os.path.join(coco_image_dir, id_to_file[ann['image_id']])
    print("image path",img_path)
    if not os.path.exists(img_path):
        continue

    keypoints = np.array(ann['keypoints']).reshape(-1, 3)[:, :2]
    #labels = infer_posture_label(keypoints)
    labels = infer_posture_label(keypoints)

    for label in labels:
        severity = estimate_severity(label, keypoints)
        severity_label = severity_descriptor(severity)
        feedback = generate_feedback(label, severity_label, img_path)  # now passing image too
        flat_pose = keypoints.flatten().tolist()
        data.append({
            "pose": flat_pose,
            "label": label,
            "severity_score": severity,
            "severity_label": severity_label,
            "feedback": feedback
        })

# ============================
# 7. Save Dataset to JSON
# ============================
with open("pose_coco_feedback_dataset_4.json", "w") as f:
    json.dump(data, f, indent=2)

print(f"✅ Saved {len(data)} examples with pose, label, severity, and feedback.")

# ============================
# 4. Save for Training
# ============================
# np.savez("pose_feedback_data_with_severity.npz", data=data)
# print(f"Saved {len(data)} pose-label-severity-feedback triplets.")
