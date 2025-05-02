from io import BytesIO
import os
import json
import numpy as np
from datasets import load_dataset
from tqdm import tqdm
from PIL import Image
import torch
import torch.nn.functional as F
import math
import cv2
from openai import OpenAI
import requests


# ========== SETUP ========== #
client = OpenAI(api_key="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")  # Insert your API key

output_json = "pose_feedback_dataset_mpii_hf2.json"

# # ========== ANGLE FUNCTION ========== #
# def angle_between(p1, p2, p3):
#     a = torch.tensor(p1)
#     b = torch.tensor(p2)
#     c = torch.tensor(p3)
#     ba = a - b
#     bc = c - b
#     cos_angle = F.cosine_similarity(ba.unsqueeze(0), bc.unsqueeze(0), dim=1).clamp(-1.0, 1.0)
#     return torch.acos(cos_angle).item() * (180 / math.pi)

# # ========== POSTURE LOGIC ========== #
# def infer_posture_label(keypoints):
#     kp = {i: keypoints[i] for i in range(len(keypoints)) if keypoints[i] is not None}
#     labels = []
#     if 5 in kp and 7 in kp and kp[7][1] - kp[5][1] < 10:
#         labels.append("slouching shoulders")
#     if 6 in kp and 8 in kp and kp[8][1] - kp[6][1] < 10:
#         labels.append("drooping right shoulder")
#     if 5 in kp and 6 in kp and abs(kp[5][1] - kp[6][1]) > 20:
#         labels.append("shoulder imbalance")
#     if 5 in kp and 11 in kp and abs(kp[5][1] - kp[11][1]) < 30:
#         labels.append("arched back")
#     if 6 in kp and 12 in kp and abs(kp[6][1] - kp[12][1]) < 30:
#         labels.append("arched back")
#     if 5 in kp and 6 in kp and 11 in kp and 12 in kp:
#         spine_angle = angle_between(kp[5], [(kp[5][0]+kp[6][0])/2, (kp[5][1]+kp[6][1])/2], kp[11])
#         if spine_angle < 150:
#             labels.append("rounded upper back")
#     if 11 in kp and 12 in kp and abs(kp[11][1] - kp[12][1]) > 20:
#         labels.append("hip tilt")
#     if 13 in kp and 15 in kp and abs(kp[13][0] - kp[15][0]) > 40:
#         labels.append("bowed left leg")
#     if 14 in kp and 16 in kp and abs(kp[14][0] - kp[16][0]) > 40:
#         labels.append("bowed right leg")
#     if 11 in kp and 13 in kp and kp[13][1] < kp[11][1]:
#         labels.append("hyperextended left knee")
#     if 12 in kp and 14 in kp and kp[14][1] < kp[12][1]:
#         labels.append("hyperextended right knee")
#     if 0 in kp and 1 in kp and 2 in kp:
#         eye_diff = abs(kp[1][1] - kp[2][1])
#         if eye_diff > 10:
#             labels.append("head tilt")
#     if 0 in kp and 5 in kp and 6 in kp and 11 in kp and 12 in kp:
#         head_mid = [(kp[5][0]+kp[6][0])/2, (kp[5][1]+kp[6][1])/2]
#         torso_mid = [(kp[11][0]+kp[12][0])/2, (kp[11][1]+kp[12][1])/2]
#         neck_angle = angle_between(kp[0], head_mid, torso_mid)
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

def estimate_severity(label, keypoints):
    if label == "slouching shoulders" and 11 in keypoints and 13 in keypoints:
        diff = abs(keypoints[13][1] - keypoints[11][1])
        return min(1.0, diff / 40)
    return 0.5

def severity_descriptor(score):
    if score < 0.3:
        return "mild"
    elif score < 0.7:
        return "moderate"
    else:
        return "severe"

# def generate_feedback(label, severity_level, image_pil):
#     from io import BytesIO
#     buffered = BytesIO()
#     image_pil.save(buffered, format="PNG")
#     image_bytes = buffered.getvalue()

#     prompt = (
#         f"You are a posture correction coach.\n"
#         f"A person is exhibiting a posture issue labeled: '{label}' with {severity_level} severity.\n"
#         f"Use the attached image to guide feedback.\n"
#         f"Generate specific, constructive advice (max 100 tokens), and do not repeat the issue name."
#     )

#     response = client.chat.completions.create(
#         model="gpt-4o",
#         messages=[
#             {"role": "system", "content": "You're a posture correction coach."},
#             {"role": "user", "content": prompt}
#         ],
#         max_tokens=100,
#         temperature=0.7,
#         tools=[
#             {
#                 "type": "image",
#                 "image": image_bytes,
#                 "mime_type": "image/png"
#             }
#         ]
#     )
#     return response.choices[0].message.content.strip()


def generate_feedback(label, severity_level, image_path):
    # Load and encode image
    # with open(image_path, "rb") as img_file:
    #     base64_image = base64.b64encode(img_file.read()).decode("utf-8")
    
    # image_data_url = f"data:image/png;base64,{base64_image}"

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


# ========== MPII-TO-ABSOLUTE-KEYPOINTS ========== #
def transform_keypoints(kpts, center, scale):
    scale_factor = 200 * scale
    transformed = []
    print("kpts",kpts[0])
    for (x, y) in kpts[0]['visible_keypoints']:
        if x == 0 and y == 0:
            transformed.append(None)
            continue
        abs_x = x + center[0] - scale_factor / 2
        abs_y = y + center[1] - scale_factor / 2
        transformed.append([abs_x, abs_y])
    return transformed

# ========== MAIN PIPELINE ========== #
if __name__ == "__main__":
    print("🔍 Loading MPII dataset...")
    ds = load_dataset("saifkhichi96/mpii-human-pose-captions", "gpt-4")
    print("datatset length",len(ds['train']))
    train_data = ds["train"]
    #train_data = ds["train"]
    #train_data = ds["train"][:100]

    processed = []

    for entry in tqdm(train_data):  # or full: train_data
        entry = entry.copy()
        print("entry keys",entry.keys())
        print(type(entry))        # <class 'dict'>
        print(entry['description'])   # Access normally
        #print(type(entry))
        image = entry["image"]
        # image = entry["image"]

        # # Hugging Face datasets.Image format: usually dict or PIL.Image
        # if isinstance(image, str):
        #     if image.startswith("http"):
        #         image = Image.open(BytesIO(requests.get(image).content)).convert("RGB")
        #     else:
        #         image = Image.open(image).convert("RGB")

        # elif isinstance(image, dict):
        #     # If Hugging Face returns {'path': ..., 'bytes': ..., 'array': ...}
        #     if "array" in image:
        #         image = Image.fromarray(np.array(image["array"])).convert("RGB")
        #     elif "bytes" in image:
        #         image = Image.open(BytesIO(image["bytes"])).convert("RGB")
        #     elif "path" in image:
        #         image = Image.open(image["path"]).convert("RGB")
        #     else:
        #         raise ValueError(f"Unrecognized image dict structure: {image}")

        # elif isinstance(image, np.ndarray):
        #     image = Image.fromarray(image).convert("RGB")

        # elif isinstance(image, Image.Image):
        #     image = image.convert("RGB")

        # else:
        #     raise TypeError(f"Unsupported image format: {type(image)}")

        # image = entry["image"]
        # if isinstance(image, str):
        #     if image.startswith("http"):
        #         image = Image.open(BytesIO(requests.get(image).content)).convert("RGB")
        #     else:
        #         image = Image.open(image).convert("RGB")
        # elif isinstance(image, np.ndarray):
        #     image = Image.fromarray(image).convert("RGB")
        people = entry["people"]
        processed_keypoints = []
        for person in people:
            kpts = np.array(person["kpts"])
            center = person["center"]
            scale = person["scale"]
            vis = np.array(person["kpts_vis"])
            visible_kpts = kpts[vis == 1]  # Only keep visible keypoints

            person_entry = {
        "id": person["id"],
        "center": person["center"],
        "scale": person["scale"],
        "visible_keypoints": visible_kpts.tolist()
    }
            processed_keypoints.append(person_entry)
        # kpts = entry["kpts"]
        # center = entry["center"]
        # scale = entry["scale"]

            keypoints = transform_keypoints(processed_keypoints, center, scale)
            keypoints = [kp if kp and all(np.isfinite(kp)) else None for kp in keypoints]

            labels = infer_posture_label(keypoints)

            for label in labels:
                severity = estimate_severity(label, keypoints)
                severity_label = severity_descriptor(severity)
                feedback = generate_feedback(label, severity_label, image)
                flat_pose = np.array([[0, 0] if p is None else p for p in keypoints]).flatten().tolist()

                processed.append({
                    "pose": flat_pose,
                    "label": label,
                    "severity_score": severity,
                    "severity_label": severity_label,
                    "feedback": feedback
                })

    with open(output_json, "w") as f:
        json.dump(processed, f, indent=2)

    print(f"✅ Saved {len(processed)} posture feedback samples to {output_json}")
