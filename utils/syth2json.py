import os
import numpy as np
from meta_data import read_json_file, save_json_file
import math
import uuid


def capitalize_first(text):
    if not text:
        return text
    first_char = text[0]
    if first_char.isupper():
        return text
    return first_char.upper() + text[1:]

asset_data = read_json_file("data/assets.json")
query = {}
for data in asset_data:
    id = data["model_id"]
    query[id] = data


def process(meta_data, room):
    res = {"objects": [], "meshes": [], "dataset": "Synthetic Data"}
    
    meshs = meta_data["rooms"][0]
    points = np.array(meshs["vertices"])
    z_value = 0
    x_max, x_min = points[:, 0].max(), points[:, 0].min()
    y_max, y_min = points[:, 1].max(), points[:, 1].min()

    floor_inf = {}
    floor_inf["uid"] = "Synth_Floor"
    floor_inf["object_name"] = "Synth_Floor"
    floor_inf["faces"] = [[0, 1, 2], [1, 3, 2]]
    floor_inf["type"] = "floor"
    floor_inf["material"] = None
    floor_inf["roomId"] = capitalize_first(room).replace(" copy", "")

    z_trans = []
    for obj in meta_data["floor_objects"]:
        z_trans.append(obj["assetId"])

    wall_objs = []
    for obj in meta_data["wall_objects"]:
        wall_objs.append(obj["assetId"])

    objs = meta_data["objects"]
    for obj in objs:
        obj_inf = {}
        obj_inf['object_name'] = obj['object_name']
        obj_inf["rotation"] = [obj["rotation"]['x'], obj["rotation"]['y'], obj["rotation"]['z']]
        obj_inf["object_path"] = obj["payload"]['path']
        obj_inf["assetId"] = obj_inf["object_path"].split("/")[-1].split(".")[0]
        obj_inf["category"] = obj["payload"]['category']
        obj_inf["label"] = obj["payload"]['label']
        obj_inf["kinematic"] = True
        if obj["assetId"] in z_trans:
            obj_inf["position"] = [obj['position']['x'] - x_min, 0, obj['position']['z'] - y_min]
        else:
            obj_inf["position"] = [obj['position']['x'] - x_min, obj['position']['y'], obj['position']['z'] - y_min]
        obj_inf["material"] = obj["material"]
        obj_inf["scale"] = [obj['scale']['x'], obj['scale']['y'], obj['scale']['z']]
        obj_inf["roomId"] = room.replace(" copy", "")
        obj_inf["bbox"] = obj["payload"]['boundingbox']

        data = query[obj_inf["object_path"].split("/")[-1].split(".usdz")[0]]
        w, l, h = data["meta_data"]['width'], data["meta_data"]['length'], data["meta_data"]['height']

        obj_x = obj_inf["position"][0]
        obj_y = obj_inf["position"][2]
        l1 = obj_x + 0.5 * w
        l2 = obj_x - 0.5 * w
        l3 = obj_y + 0.5 * l
        l4 = obj_y - 0.5 * l

        x_max = max(x_max, l1)
        x_min = min(x_min, l2)
        y_max = max(y_max, l3)
        y_min = min(y_min, l4)

        if obj_inf["assetId"] in query.keys():
            if obj["assetId"] in wall_objs:
                det = math.sqrt(l**2 + w**2 + h**2)
                if det > 1 and obj_inf["position"][1] > 0:
                    res["objects"].append(obj_inf)
            else:
                res["objects"].append(obj_inf)

    floor_inf["xyz"] = [
            [x_max - x_min, z_value, y_max - y_min],
            [x_max - x_min, z_value, y_min - y_min],
            [x_min - x_min, z_value, y_max - y_min],
            [x_min - x_min, z_value, y_min - y_min]
        ]
    res["meshes"].append(floor_inf)
    return res

if __name__ == "__main__":
    input_path = "/home/zwx/Desktop/work/scene"
    output_path = "data/layout"
    i = 0
    room_type = os.listdir(input_path)
    for room in room_type:
        room_dir = os.path.join(input_path, room)
        room_files = os.listdir(room_dir)
        for file in room_files:
            file = os.path.join(room_dir, file)
            files = os.listdir(file)
            for layout in files:
                if layout.split(".")[-1] == "json":
                    path = os.path.join(file, layout)
                    save_path = os.path.join(output_path, str(uuid.uuid1()) + ".json")
                    meta_data = read_json_file(path)
                    try:
                        res = process(meta_data, room)
                        if len(res["objects"]) > 0:
                            save_json_file(res, save_path)
                        else:
                            i += 1
                    except:
                        i += 1
    print(i)


