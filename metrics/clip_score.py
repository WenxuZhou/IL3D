from PIL import Image
import torch
import os
import json
import argparse
from transformers import CLIPProcessor, CLIPModel

def clip_sim(model, processor, device, img_path, prompt):
    image = Image.open(img_path).convert("RGB")
    inputs = processor(images=image, text=prompt, return_tensors="pt").to(device)
    outputs = model(**inputs)
    return outputs.logits_per_image.item()

def read_json_file(file_path):
    with open(file_path, 'r') as file:
        return json.load(file)

def max_sim(model, processor, device, scene_dir, prompt):
    img1 = os.path.join(scene_dir, "image_000.png")
    img2 = os.path.join(scene_dir, "image_090.png")
    img3 = os.path.join(scene_dir, "image_180.png")
    img4 = os.path.join(scene_dir, "image_270.png")
    
    similarity1 = clip_sim(model, processor, device, img1, prompt)
    similarity2 = clip_sim(model, processor, device, img2, prompt)
    similarity3 = clip_sim(model, processor, device, img3, prompt)
    similarity4 = clip_sim(model, processor, device, img4, prompt)

    return max(similarity1, similarity2, similarity3, similarity4)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Check USD object CLIP-Similarity")
    parser.add_argument("--scene_dir", type=str, default="infer_res/sft_IL3D_1.7b/bathroom/0", help="Path to scene directory (contains object.usda and floor.usda)")
    args = parser.parse_args()

    model = CLIPModel.from_pretrained("ckpts/clip-vit-base-patch32")
    processor = CLIPProcessor.from_pretrained("ckpts/clip-vit-base-patch32", use_fast=True)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    id = 18
    prompt_dir = args.scene_dir.split("/")[-2]
    prompt_path = os.path.join("infer_res/prompt", f"{prompt_dir}.json")
    prompts = read_json_file(prompt_path)
    prompt = prompts[int(args.scene_dir.split("/")[-1])]["description"]
    similarity = max_sim(model, processor, device, args.scene_dir, prompt)
    print(f"Max CLIP-Sim score: {similarity:.4f}")
