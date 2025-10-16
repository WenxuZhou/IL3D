import os
import base64
import json
from PIL import Image
from io import BytesIO
from openai import OpenAI
import argparse
from tqdm import tqdm


client = OpenAI(
    base_url="",
    api_key=""
)

def get_prompt(scene_description):
    res = f"""
## Role
A professional evaluator specializing in indoor functional logic, ergonomics, and aesthetic design, tasked with objective, evidence-based assessment of top-down indoor layouts across 5 core dimensions, strictly adhering to "Object Pose, Physical Reality, Semantic Consistency, Scene Functionality, and Visual Aesthetics" principles. Evaluations must rely exclusively on provided text descriptions and top-down images, with no subjective inferences about unmentioned details.

## Core Evaluation Dimensions
1. **Object Pose**: Assesses positional accuracy, orientation rationality, proportional relationships, and spatial distances between objects (e.g., functional alignment, realistic size ratios, appropriate gaps).
2. **Physical Reality**: Judges compliance with physical laws, including absence of floating objects, non-penetrating spatial relationships, reasonable load-bearing, and gravity consistency.
3. **Semantic Consistency**: Evaluates logical matching between objects and scene type, and between objects themselves (e.g., functional relevance, scenario appropriateness).
4. **Scene Functionality**: Measures practical usability via traffic flow smoothness, functional zoning clarity, ergonomic spacing, and space utilization efficiency.
5. **Visual Aesthetics**: Assesses spatial balance, style unity,留白合理性, and arrangement orderliness.

## Evaluation Rules
- **Information Boundary**: Limited to "scene description" and 4 top-down renderings. Note "insufficient image details" for ambiguous elements.
- **Scoring (0-10)**: 10=perfect; 8-9=excellent (negligible flaws); 6-7=good (minor issues); 4-5=partial compliance (obvious defects); 2-3=poor (major flaws); 0-1=non-compliant (invalid layout).
- **Scope**: Comprehensive 5-dimensional assessment of the entire scene.

## Scene Information
- **Description**: {scene_description};
- **Images**: 4 top-down renderings (full scene coverage, no blind spots).

## Response Format
Standard JSON with scores (0-10) and evidence-based comments (linking text and image details) for each dimension:
{{\
"Object Pose": {{"Score": 8, "Comment": "Consistent with scene description stating 'dining chairs arranged around table'—images show 4 chairs aligned with table edges (65cm spacing, consistent with ergonomic standards). Minor deviation in one chair's orientation (5° off) does not affect functionality."}},\
"Physical Reality": {{"Score": 10, "Comment": "All objects in images have valid support (no floating elements); spatial relationships avoid penetration. Consistent with text description of 'stable furniture placement'."}},\
"Semantic Consistency": {{"Score": 6, "Comment": "Most objects match 'living room' description (sofa, TV, coffee table) per images, but text-specified 'bookshelf' is absent, creating a minor semantic gap."}},\
"Scene Functionality": {{"Score": 7, "Comment": "Main passage (100cm) meets standards (≥90cm) as shown in images, aligning with text's 'smooth traffic flow' claim. Minor crowding in corner (20cm gap between cabinet and sofa) reduces efficiency." }},\
"Visual Aesthetics": {{"Score": 9, "Comment": "Images show balanced spatial distribution (no weight bias) and unified modern style, consistent with text's 'neat arrangement' description. Minimal asymmetry in decor placement is negligible." }}\
}}
"""
    return res

if not client.api_key:
    raise ValueError("Please set the OPENAI_API_KEY environment variable")

def read_json_file(file_path):
    with open(file_path, 'r') as file:
        return json.load(file)
    
def save_json_file(data, file_path):
    with open(file_path, 'w') as file:
        file.write(json.dumps(data, indent=4))

def json_str_to_dict(json_str):
    try:
        if json_str.startswith('```json'):
            json_str = json_str[7:]
        if json_str.endswith('```'):
            json_str = json_str[:-3]
            
        json_str = json_str.strip()
        result_dict = json.loads(json_str)
        return result_dict
    except json.JSONDecodeError as e:
        print(f"JSON decode error: {e}")
        return None
    except Exception as e:
        print(f"Error during conversion: {e}")
        return None

def encode_image(image_path):
    try:
        with Image.open(image_path) as img:
            if img.mode in ("RGBA", "P"):
                img = img.convert("RGB")
            buffered = BytesIO()
            img.save(buffered, format="JPEG")
            return base64.b64encode(buffered.getvalue()).decode('utf-8')
    except Exception as e:
        raise ValueError(f"Error processing image {image_path}: {str(e)}")

def call_gpt4v(image_paths, text_input, model="openai/gpt-4o", max_tokens=2000):
    if len(image_paths) != 4:
        raise ValueError("Please provide exactly 4 image paths")
    
    base64_images = [encode_image(path) for path in image_paths]
    
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": text_input}
            ]
        }
    ]
    
    for img in base64_images:
        messages[0]["content"].append({
            "type": "image_url",
            "image_url": {
                "url": f"data:image/jpeg;base64,{img}"
            }
        })
    
    try:
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            max_tokens=max_tokens
        )
        return response.choices[0].message.content
    except Exception as e:
        raise Exception(f"API request failed: {str(e)}")


def eval(scene_dir, id):
    image_paths = [
        os.path.join(scene_dir, "image_000.png"),
        os.path.join(scene_dir, "image_090.png"),
        os.path.join(scene_dir, "image_180.png"),
        os.path.join(scene_dir, "image_270.png")
    ]

    prompt_dir = scene_dir.split("/")[-2]
    prompt_path = os.path.join("infer_res/prompt", f"{prompt_dir}.json")
    prompts = read_json_file(prompt_path)
    prompt = prompts[int(id)]["description"]
    input_text = get_prompt(prompt)

    output_path = os.path.join(scene_dir, "gpt_rating.json")

    if not os.path.exists(output_path):
        result = call_gpt4v(image_paths, input_text)
        result_dict = json_str_to_dict(result)
    return result_dict



