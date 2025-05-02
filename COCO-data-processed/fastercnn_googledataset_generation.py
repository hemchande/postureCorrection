import os
from PIL import Image
import json
import torch
import torchvision
import torchvision.transforms.functional as TF
from torchvision.models.detection import keypointrcnn_resnet50_fpn
from datasets import Dataset
from tqdm import tqdm
import numpy as np
import math
import torch.nn.functional as F
import base64
import openai
from openai import OpenAI
import logging

# Setup logging
logging.basicConfig(
    filename="feedback_generation.log",
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)


client = OpenAI(api_key="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")

openai.api_key = "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"

# Step 1: Define posture error descriptions
posture_errors = {
    "slouching_shoulders": "A human with rounded shoulders leaning slightly forward.",
    "shoulder_imbalance": "A human with uneven shoulders, one side visibly higher.",
    "arching_back": "A human with a pronounced curve in the lower back.",
    "bowing_legs": "A human standing with legs curved outward like a bow.",
    "hyperextended_knees": "A human with knees pushed backward beyond normal alignment."
}

# Step 2: Define inference functions
fasterrcnn =keypointrcnn_resnet50_fpn(pretrained=True)
fasterrcnn.eval()

# def angle_between(p1, p2, p3):
#     a = torch.tensor(p1)
#     b = torch.tensor(p2)
#     c = torch.tensor(p3)
#     ba = a - b
#     bc = c - b
#     cos_angle = F.cosine_similarity(ba.unsqueeze(0), bc.unsqueeze(0), dim=1).clamp(-1.0, 1.0)
#     return torch.acos(cos_angle).item() * (180 / math.pi)


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
#     if 0 in kp and 5 in kp and 6 in kp:
#         head_mid = [(kp[5][0]+kp[6][0])/2, (kp[5][1]+kp[6][1])/2]
#         neck_angle = angle_between(kp[0], head_mid, [(kp[11][0]+kp[12][0])/2, (kp[11][1]+kp[12][1])/2])
#         if neck_angle < 160:
#             labels.append("forward head posture")
#     return labels if labels else ["good posture"]


# def angle_between(p1, p2, p3):
#     a = torch.tensor(p1)
#     b = torch.tensor(p2)
#     c = torch.tensor(p3)
#     ba = a - b
#     bc = c - b
#     cos_angle = F.cosine_similarity(ba.unsqueeze(0), bc.unsqueeze(0), dim=1).clamp(-1.0, 1.0)
#     return torch.acos(cos_angle).item() * (180 / math.pi)

# def infer_posture_label(keypoints):
#     kp = {
#         i: keypoints[i]
#         for i in range(len(keypoints))
#         if keypoints[i] is not None and not (keypoints[i][0] == 0 and keypoints[i][1] == 0)
#     }

#     labels = []

#     if 5 in kp and 7 in kp and (kp[7][1] - kp[5][1] < 10):
#         labels.append("slouching shoulders")
#     if 6 in kp and 8 in kp and (kp[8][1] - kp[6][1] < 10):
#         labels.append("drooping right shoulder")
#     if 5 in kp and 6 in kp and (abs(kp[5][1] - kp[6][1]) > 20):
#         labels.append("shoulder imbalance")

#     if 5 in kp and 11 in kp and (abs(kp[5][1] - kp[11][1]) < 30):
#         labels.append("arched back")
#     if 6 in kp and 12 in kp and (abs(kp[6][1] - kp[12][1]) < 30):
#         labels.append("arched back")
#     if 5 in kp and 6 in kp and 11 in kp and 12 in kp:
#         spine_angle = angle_between(
#             kp[5],
#             [(kp[5][0] + kp[6][0]) / 2, (kp[5][1] + kp[6][1]) / 2],
#             kp[11]
#         )
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
#         if abs(kp[1][1] - kp[2][1]) > 10:
#             labels.append("head tilt")
#     if 0 in kp and 5 in kp and 6 in kp and 11 in kp and 12 in kp:
#         head_mid = [(kp[5][0] + kp[6][0]) / 2, (kp[5][1] + kp[6][1]) / 2]
#         torso_mid = [(kp[11][0] + kp[12][0]) / 2, (kp[11][1] + kp[12][1]) / 2]
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
    if label == "slouching shoulders" and 5 in keypoints and 7 in keypoints:
        diff = abs(keypoints[7][1] - keypoints[5][1])
        return min(1.0, diff / 40)
    elif label == "shoulder imbalance" and 5 in keypoints and 6 in keypoints:
        diff = abs(keypoints[5][1] - keypoints[6][1])
        return min(1.0, diff / 40)
    return 0.5

def severity_descriptor(score):
    if score < 0.3:
        return "mild"
    elif score < 0.7:
        return "moderate"
    else:
        return "severe"

def generate_feedback(label, severity_level, image_path, max_tokens=250):
    try:
        # Read and encode image as base64
        with open(image_path, "rb") as img_file:
            image_data = base64.b64encode(img_file.read()).decode("utf-8")
        image_url = f"data:image/png;base64,{image_data}"

        # Construct multimodal prompt
        message = [
            {
                "type": "text",
                "text": (
                    f"You are a posture correction coach.\n"
                    f"A person is exhibiting a posture issue labeled: '{label}' "
                    f"with {severity_level} severity.\n"
                    f"Analyze the attached image and provide clear, constructive feedback (max {max_tokens} tokens)."
                )
            },
            {
                "type": "image_url",
                "image_url": {
                    "url": image_url,
                    "detail": "high"
                }
            }
        ]

        # Send to OpenAI
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "You're a posture correction coach."},
                {"role": "user", "content": message}
            ],
            max_tokens=max_tokens,
            temperature=0.7,
        )

        feedback = response.choices[0].message.content.strip()
        logging.info(f"[SUCCESS] Feedback for {label} ({severity_level}): {feedback}")
        return feedback

    except Exception as e:
        logging.error(f"[FAIL] Feedback generation failed for {label} ({severity_level}) — {e}")
        return f"[Error] Could not generate feedback for {label} ({severity_level})"


# def generate_feedback(label, severity_level, image_path):
#     with open(image_path, "rb") as img_file:
#         base64_image = base64.b64encode(img_file.read()).decode("utf-8")
#     return f"{label} correction at {severity_level} severity. (Simulated feedback)"

# Step 3: Collect image paths and descriptions
data = []
for label, desc in posture_errors.items():
    folder = f"posture_images/{label}"
    if not os.path.exists(folder):
        continue

    for fname in os.listdir(folder):
        if fname.lower().endswith(("jpg", "jpeg", "png")):
            path = os.path.join(folder, fname)
            try:
                img = Image.open(path).convert("RGB")
                print(img.size)
                img_tensor = TF.to_tensor(img)
                with torch.no_grad():
                    prediction = fasterrcnn([img_tensor])[0]
                    print(prediction)

                if len(prediction["boxes"]) == 0 or "keypoints" not in prediction or len(prediction["keypoints"]) == 0:
                    continue

                keypoints = prediction["keypoints"][0, :, :2].cpu().numpy()
                flat_pose = keypoints.flatten().tolist()
                print(len(flat_pose))

                severity = estimate_severity(label.replace("_", " "), keypoints)
                severity_label = severity_descriptor(severity)
                print(severity_label)
                feedback = generate_feedback(label, severity_label, path)
                print(feedback)

                data.append({
                    "pose": flat_pose,
                    "label": label.replace("_", " "),
                    "severity_score": severity,
                    "severity_label": severity_label,
                    "feedback": feedback
                })
            except:
                continue

# Step 4: Save as JSON
data_json_path = "pose_google_feedback_dataset_fasterrcnn2.json"
with open(data_json_path, "w") as f:
    json.dump(data, f, indent=2)

print(f"✅ Saved {len(data)} additional examples from internet image dataset to {data_json_path}")
