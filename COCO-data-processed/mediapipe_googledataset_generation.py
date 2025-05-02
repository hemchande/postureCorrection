import os
import json
import math
import base64
import numpy as np
from PIL import Image
from tqdm import tqdm
import torch
import torch.nn.functional as F
import cv2
import mediapipe as mp
from openai import OpenAI

# ============== OpenAI API ==============
client = OpenAI(api_key="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")

# ============== Pose Inference Setup ==============
mp_pose = mp.solutions.pose
pose_detector = mp_pose.Pose(static_image_mode=True)

# ============== Posture Error Descriptions ==============
posture_errors = {
    "slouching_shoulders": "A human with rounded shoulders leaning slightly forward.",
    "shoulder_imbalance": "A human with uneven shoulders, one side visibly higher.",
    "arching_back": "A human with a pronounced curve in the lower back.",
    "bowing_legs": "A human standing with legs curved outward like a bow.",
    "hyperextended_knees": "A human with knees pushed backward beyond normal alignment."
}

# ============== Utility Functions ==============
def angle_between(p1, p2, p3):
    a = torch.tensor(p1)
    b = torch.tensor(p2)
    c = torch.tensor(p3)
    ba = a - b
    bc = c - b
    cos_angle = F.cosine_similarity(ba.unsqueeze(0), bc.unsqueeze(0), dim=1).clamp(-1.0, 1.0)
    return torch.acos(cos_angle).item() * (180 / math.pi)

def infer_posture_label(keypoints):
    kp = {i: keypoints[i] for i in range(len(keypoints)) if keypoints[i] is not None}
    labels = []
    # MediaPipe keypoint indices differ from COCO
    # e.g., LEFT_SHOULDER = 11, RIGHT_SHOULDER = 12, LEFT_ELBOW = 13, etc.
    if 11 in kp and 13 in kp and kp[13][1] - kp[11][1] < 10:
        labels.append("slouching shoulders")
    if 11 in kp and 12 in kp and abs(kp[11][1] - kp[12][1]) > 20:
        labels.append("shoulder imbalance")
    if 11 in kp and 23 in kp and abs(kp[11][1] - kp[23][1]) < 30:
        labels.append("arched back")
    if 23 in kp and 25 in kp and abs(kp[23][0] - kp[25][0]) > 40:
        labels.append("bowed left leg")
    if 24 in kp and 26 in kp and abs(kp[24][0] - kp[26][0]) > 40:
        labels.append("bowed right leg")
    if 23 in kp and 25 in kp and kp[25][1] < kp[23][1]:
        labels.append("hyperextended left knee")
    if 24 in kp and 26 in kp and kp[26][1] < kp[24][1]:
        labels.append("hyperextended right knee")
    return labels if labels else ["good posture"]

def estimate_severity(label, keypoints):
    if label == "slouching shoulders" and 11 in keypoints and 13 in keypoints:
        diff = abs(keypoints[13][1] - keypoints[11][1])
        return min(1.0, diff / 40)
    elif label == "shoulder imbalance" and 11 in keypoints and 12 in keypoints:
        diff = abs(keypoints[11][1] - keypoints[12][1])
        return min(1.0, diff / 40)
    return 0.5

def severity_descriptor(score):
    if score < 0.3:
        return "mild"
    elif score < 0.7:
        return "moderate"
    else:
        return "severe"


import base64

def generate_feedback(label, severity_level, image_path):
    with open(image_path, "rb") as img_file:
        base64_image = base64.b64encode(img_file.read()).decode("utf-8")

    prompt = (
        f"You are a posture correction coach.\n"
        f"A person is exhibiting a posture issue labeled: '{label}' with {severity_level} severity.\n"
        f"Use the attached image to guide feedback.\n"
        f"Generate specific, constructive advice (max 100 tokens), and do not repeat the issue name."
    )

    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": "You're a posture correction coach."},
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": prompt
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/png;base64,{base64_image}",
                            "detail": "high"
                        }
                    }
                ]
            }
        ],
        max_tokens=100,
        temperature=0.7
    )
    return response.choices[0].message.content.strip()


# def generate_feedback(label, severity_level, image_path):
#     with open(image_path, "rb") as img_file:
#         image_data = img_file.read()

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
#                 "image": image_data,
#                 "mime_type": "image/png"
#             }
#         ]
#     )
#     return response.choices[0].message.content.strip()

# ============== Main Dataset Collection Loop ==============
data = []
for label, desc in posture_errors.items():
    folder = f"posture_images/{label}"
    if not os.path.exists(folder):
        continue

    for fname in os.listdir(folder):
        if fname.lower().endswith(("jpg", "jpeg", "png")):
            path = os.path.join(folder, fname)
            try:
                image_bgr = cv2.imread(path)
                print("image_bgr",image_bgr)
                if image_bgr is None:
                    continue
                image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
                result = pose_detector.process(image_rgb)
                #print("result",result)
                if not result.pose_landmarks:
                    continue

                landmarks = result.pose_landmarks.landmark
                #print("landmarks",landmarks)
                keypoints = []
                for lm in landmarks:
                    #print("lm",lm)
                    print("lm x",lm.x)
                    print("lm y",lm.y)
                    print("lm z",lm.z)
                    keypoints.append([lm.x * image_rgb.shape[1], lm.y * image_rgb.shape[0]])

                labels = infer_posture_label(keypoints)
                for label_found in labels:
                    severity = estimate_severity(label_found, keypoints)
                    severity_label = severity_descriptor(severity)
                    feedback = generate_feedback(label_found, severity_label, path)
                    flat_pose = np.array(keypoints).flatten().tolist()

                    data.append({
                        "pose": flat_pose,
                        "label": label_found,
                        "severity_score": severity,
                        "severity_label": severity_label,
                        "feedback": feedback
                    })
            except Exception as e:
                print(f"⚠️ Error processing {path}: {e}")
                continue

# ============== Save to JSON ==============
data_json_path = "pose_mediapipe_google_feedback_dataset.json"
with open(data_json_path, "w") as f:
    json.dump(data, f, indent=2)

print(f"✅ Saved {len(data)} examples from MediaPipe image dataset to {data_json_path}")
