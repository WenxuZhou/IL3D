import os
from utils.meta_data import read_json_file, save_json_file
import numpy as np
from tqdm import tqdm
import json
import random
import argparse


def get_prompt(objects, room_type):
    res = f"""
## Role
Your task is to arrange some objects within a given {room_type} effectively. Follow these guidance to complete your design:

## Rules
(1) Extract the [Objects] and [Bounding Box Size] from the object information.
(2) Analyze the spatial relationships among [Objects] within the specified [Room Type]. Pay special attention to **avoiding overlap** and **consider other spatial factors like accessibility and aesthetics**.
(3) Determine and design the precise location of all [Objects] ensuring that their bounding boxes do not overlap and that the layout is functional and visually appealing.
(4) I prefer objects to be placed at the edge (the most important constraint) of the room if possible which makes the room look more spacious.
(5) Objects usually need to be aligned in some way (such as parallel or perpendicular to the walls) and **must not extend beyond the floor area**.
(6) Chairs must be placed near to the table/desk and face to the table/desk.
(7) Before specifying the detailed positions of each object, first think about their general arrangement and relative spatial relationships:
    a) Which objects need the most space or have fixed positions (like beds, wardrobes)
    b) Which objects need to be grouped together (like nightstands with bed)
    c) Traffic flow and accessibility considerations.

## Object Information
{objects}
*Note: bbox format is [length, width, height] in meters*

## Response Format    
First design the vertices of the floor, then report the 3D spatial coordinates and rotation angles of each object in JSON format, as follows:
{{\
'Floor': {{'xyz': [[8.0, 0, 6.76], [8.0, 0, 0.0], [0.0, 0, 6.76], [0.0, 0, 0.0]]}},\
'Coffee Tables': [{{'position': [1.62, 0.0, 2.29], 'rotation': [180, 90, 180]}}],\
'Benches': [{{'position': [1.72, 0.0, 3.66], 'rotation': [0, 0, 0]}}, {{'position': [1.63, 0.0, 0.9], 'rotation': [0, 0, 0]}}]\
}}

Important Notes about Coordinate System:
- Y-axis points upward (y=0 is floor level)
- X-axis runs along the room's length from west to east
- Z-axis runs along the room's width from south to north
- All coordinates are in meters
- Output nothing but the JSON (No preamble, no explanation, no additional text of any kind)
"""
    return res

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_folder', type=str, default="data/layout")
    parser.add_argument('--meta_data', default="data/assets.json")
    parser.add_argument('--dataset', type=str, default='IL3D', choices=['FRONT3d', 'HSSD', 'IL3D'])
    args = parser.parse_args()

    dataset = "SFT_" + args.dataset
    room_path = args.input_folder
    meta_data = read_json_file(args.meta_data)
    chiose = args.dataset
    if chiose == 'FRONT3d':
        det = ['3D-FRONT']
    elif chiose == 'HSSD':
        det = ['HSSD']
    else:
        det = ['3D-FRONT', 'HSSD', 'Synthetic Data']

    query = {}
    for data in meta_data:
        id = data["model_id"]
        query[id] = data

    os.makedirs(dataset, exist_ok=True)

    rooms = os.listdir(room_path)
    for r in tqdm(rooms):
        room = read_json_file(os.path.join(room_path, r))
        if room["dataset"] in det:
            obj = {}
            inf = {}

            floor = room["meshes"][0]["xyz"]
            mesh = []
            for i in range(len(floor)):
                mesh.append([round(num, 2) for num in floor[i]])
            obj["Floor"] = {"xyz": mesh}
            random.shuffle(room["objects"])
            for d in room["objects"]:
                id = d["roomId"]
                label = d['category']
                path = d['object_path']
                obj_name = d['object_name']
                if label == None:
                    continue
                pos = d["position"]
                rot = d["rotation"]
                bbox = d["bbox"]
                scale = d["scale"]
                pos = [round(num, 2) for num in pos]
                rot = [round(num, 2) for num in rot]
                bbox = [round(bbox[num] * scale[num], 2) for num in range(len(bbox))]
                description = query[d["assetId"]]["meta_data"]["description"]

                if not label in obj.keys():
                    obj[label] = []
                if not label in inf.keys():
                    inf[label] = []
                inf[label].append({"bbox": bbox, "description": description})
                obj[label].append({"position": pos, "rotation": rot})

            
            mes = {"messages": []}
            room_type = room["objects"][0]["roomId"]
            if room_type == "OtherRoom":
                continue
            
            prompt = get_prompt(inf, room_type)
            res2 = {"role": "user", "content": str(prompt)}
            res3 = {"role": "assistant", "content": str(obj)}
            mes["messages"].append(res2)
            mes["messages"].append(res3)
            mes_name = r.split(".")[0] + ".json"
            save_json_file(mes, os.path.join(dataset, mes_name))
