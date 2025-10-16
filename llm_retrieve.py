import os
import ast
import uuid
import argparse
from retrieve.qdrant_3d_client import QdrantMultiVectorFor3D
from retrieve.text_embedding import TextEmbeddingModel
from retrieve.query_asset import query_asset
import numpy as np
from tqdm import tqdm
from utils.meta_data import read_json_file, save_json_file
from transformers import AutoModelForCausalLM, AutoTokenizer


def str_to_dict(raw_text: str) -> dict:
    if "</think>" in raw_text:
        cleaned_text = raw_text.split("</think>")[-1]
    else:
        cleaned_text = raw_text
    cleaned_text = cleaned_text.strip("\n\t\r\f ")
    valid_lines = [line.strip() for line in cleaned_text.splitlines() if line.strip()]
    cleaned_text = "\n".join(valid_lines)
    target_dict = ast.literal_eval(cleaned_text)
    return target_dict

def retrieve_prompt(room_description):
    res = f"""
## Role
Your task is to analyze a room description to identify the room type and all mentioned objects with their basic descriptions.

## Rules
(1) Extract the room type from the description (e.g., BedRoom, LivingRoom, Kitchen, etc.).
(2) Identify all mentioned objects and their basic descriptions exactly as described.
(3) If the description mentions multiple instances of the same object, maintain the exact count.
(4) Don't omit the information about the objects in the description.
(5) When encountering quantity words (e.g., six, two, three, multiple) describing objects, split them into individual objects equal to the quantity. \
Use the singular form of the object name, and apply the same description to each.
For example, "six chairs with blue upholstery" should be split into six separate objects: [{{"name": "Chair", "description": "A chair with blue upholstery"}}, ...] (repeated six times).
(6) Ensure quantity words are not included in the object name. Focus on the core object name in singular form (e.g., use "Chair" instead of "six chairs" or "Chairs").

## Room Description
{room_description}

## Response Format
Report the result of the room_type and dictionaries for each object in JSON format, as follows:
{{
"room_type": "LivingRoom",\
"objects": [{{"name": "Sofa", "A dark green upholstered ottoman with a cushioned lid and decorative brass nailhead trim along its edges."}},\
{{'name': 'Armchair', "description": 'A armchair with a sleek design, featuring a cushioned seat and backrest supported by thin metal legs.'}},\
{{'name': 'Armchair', "description": 'A armchair featuring a curved backrest and armrests. It has a dark green upholstery and thin metal legs.'}},\
{{"name": "Coffee Table", "description": "A modern coffee table with a round wooden top and three sleek legs that taper towards the bottom."}}]\
}}

Important Notes about response format:
- Output nothing but the JSON. No preamble, no explanation, no additional text of any kind
"""
    return res

def retrieve_from_text(prompt, dataset,  tokenizer, model, res_path):
    print("TEXT DESCRIPTION:")
    print(prompt)

    print("ROOM INFORMATION:")
    messages = [{"role": "user", "content": retrieve_prompt(prompt)}]
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=True
    )
    model_inputs = tokenizer([text], return_tensors="pt").to(model.device)
    generated_ids = model.generate(
        **model_inputs,
        max_new_tokens=32768
    )
    output_ids = generated_ids[0][len(model_inputs.input_ids[0]):].tolist() 
    try:
        index = len(output_ids) - output_ids[::-1].index(151668)
    except ValueError:
        index = 0
    output_ids = generated_ids[0][len(model_inputs.input_ids[0]):].tolist()
    # thinking_content = tokenizer.decode(output_ids[:index], skip_special_tokens=True).strip("\n")
    content = tokenizer.decode(output_ids[index:], skip_special_tokens=True).strip("\n")
    room_inf = str_to_dict(content)
    print(room_inf)

    print("Asset Retrieval:")
    room_type = room_inf["room_type"]
    objects = room_inf["objects"]
    obj_res = {'room_type': room_type, 'objects': {}}
    for data in objects:
        key = data['description']
        query_description = text_embedding_model.text_embedding([key])[0].tolist()
        retrieve_res = query_asset(qdrant_client, query_description, limit=1)
        obj_path = retrieve_res.points[0].payload["path"]
        id = obj_path.split("/")[-1].split(".")[0]
        obj_scale = dataset[id]["meta_data"]
        label = dataset[id]["category"].capitalize()
        scale = dataset[id]["meta_data"]["scale"]
        bbox = [obj_scale["width"], obj_scale["length"], obj_scale["height"]]
        bbox = [round(bbox[num] * scale[num], 2) for num in range(len(bbox))]
        description = dataset[id]["meta_data"]["description"]
        if not label in obj_res['objects'].keys():
            obj_res['objects'][label] = []
        obj_res['objects'][label].append({"object_name": "infer_" + dataset[id]["model_id"],
                                 "path": obj_path,
                                 "bbox": bbox,
                                 "description": description})
    print(obj_res)
    save_json_file(obj_res, res_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, default=None, required=True, help="Path to qwen checkpoint")
    parser.add_argument('--res_dir', default=None, required=True, type=str, help="Path to scene res")
    args = parser.parse_args()

    data_dir = os.path.join(os.getcwd(), "data")
    qdrant_path = "data/qdrant"
    txt_emb = "data/text_emb"
    meta_path = "data/assets.json"
    model_path = args.model_path
    res_dir = args.res_dir

    meta_data = read_json_file(meta_path)
    dataset = {}
    for data in meta_data:
        id = data["model_id"]
        dataset[id] = data

    # Perform inference using the native PyTorch engine
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype="auto",
        device_map="auto"
    )

    # Perform retrieve using Qdrant
    qdrant_client = QdrantMultiVectorFor3D(client_path=qdrant_path)
    text_embedding_model = TextEmbeddingModel(cache_dir=txt_emb)

    prompt = input("Please enter a scene description:\n")
    res_path = os.path.join(res_dir, "objects.json")
    os.makedirs(res_dir, exist_ok=True)
    retrieve_from_text(prompt, dataset, tokenizer, model, res_path)


