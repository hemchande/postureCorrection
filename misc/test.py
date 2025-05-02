import base64
from openai import OpenAI


client = OpenAI(api_key="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")

def generate_feedback(label, severity_label, image_path):
    # Convert image to base64 Data URL
    with open(image_path, "rb") as f:
        image_data = base64.b64encode(f.read()).decode("utf-8")
    image_url = f"data:image/png;base64,{image_data}"

    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {
                "role": "system",
                "content": "You are a posture correction coach."
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": (
                            f"A person is exhibiting a posture issue labeled: '{label}' "
                            f"with {severity_label} severity. "
                            f"Analyze the attached image and generate 250-token corrective feedback."
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
            }
        ],
        max_tokens=250,
        stream=True
    )

    for chunk in response:
        print(chunk)
        if "choices" in chunk:
            delta = chunk['choices'][0]['delta']
            if 'content' in delta:
                print(delta['content'], end='', flush=True)

    #return response.choices[0].message.content.strip()





generate_feedback("good posture","moderate","../Downloads/train2014/COCO_train2014_000000000086.jpg")



