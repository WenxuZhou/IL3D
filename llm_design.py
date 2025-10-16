import os
import ast
import uuid
import numpy as np
from tqdm import tqdm
import argparse
from utils.meta_data import read_json_file, save_json_file
from pxr import Usd, UsdGeom, UsdShade, Sdf, Gf
from swift.llm import PtEngine, InferRequest, RequestConfig


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


def create_scene(scene_data: dict, asset_root_path: str, save_path: str):
    stage = Usd.Stage.CreateInMemory()
    root_prim = stage.DefinePrim("/Root")

    for key in scene_data.keys():
        if key != "Floor":
            data = scene_data[key]
            for obj in data:
                print(obj)
                object_name = obj['object_name']
                object_name = object_name.replace("-", "_")
                object_name = object_name.replace(" ", "_")
                object_name = object_name.replace("'", "")

                prim_path = f"/Root/{object_name}"
                prim = stage.DefinePrim(prim_path)

                asset_path = os.path.join(asset_root_path, obj['path'])
                prim.GetReferences().AddReference(asset_path)

                translate_attr = prim.CreateAttribute("xformOp:translate", Sdf.ValueTypeNames.Float3)
                translate_attr.Set(Gf.Vec3d(
                    obj["position"][0] * 100,
                    obj["position"][1] * 100,
                    obj["position"][2] * 100
                ))

                scale_attr = prim.CreateAttribute("xformOp:scale", Sdf.ValueTypeNames.Float3)
                scale_attr.Set(Gf.Vec3f(1, 1, 1))
                
                det = len(obj['rotation'])
                if det == 3:
                    rotate_attr = prim.CreateAttribute("xformOp:rotateXYZ", Sdf.ValueTypeNames.Float3)
                    rotate_attr.Set(Gf.Vec3d(
                        obj['rotation'][0],
                        obj['rotation'][1],
                        obj['rotation'][2]
                    ))

                    xform_op_order = prim.CreateAttribute("xformOpOrder", Sdf.ValueTypeNames.TokenArray)
                    xform_op_order.Set(["xformOp:translate", "xformOp:rotateXYZ", "xformOp:scale"])
                elif det == 4:
                    rotate_attr = prim.CreateAttribute("xformOp:orient", Sdf.ValueTypeNames.Quatd)
                    rotate_attr.Set(Gf.Quatd(
                        obj['rotation'][3],
                        obj['rotation'][0],
                        obj['rotation'][1],
                        obj['rotation'][2]
                    ))

                    xform_op_order = prim.CreateAttribute("xformOpOrder", Sdf.ValueTypeNames.TokenArray)
                    xform_op_order.Set(["xformOp:translate", "xformOp:orient", "xformOp:scale"])
                else:
                    raise NotImplementedError

    stage.SetDefaultPrim(root_prim)
    stage.GetRootLayer().Export(save_path)

    return stage


def read_mesh_attr(mesh):
    v = np.array(mesh).astype(np.float64) * 100
    faces = np.array([[0, 1, 2], [1, 3, 2]]).astype(np.int8)
    mat = None
    vt = None
    try:
        uv_array = np.reshape(mesh['uv'], [-1, 2]).astype(np.float64)
    except:
        uv_array = None

    if uv_array is not None:
        if mesh['material'] is not None:
            mat = np.reshape(mesh['material'], [3, 3]).astype(np.float64)
            if mat is not None:
                uv_array = (mat[:2,:2] @ uv_array.T).T
        
        vt = []
        for uv in uv_array:
            vt.append([float(uv[0]), float(uv[1])])
    
    return v, faces, vt, mat

def create_floor(mesh: dict, data_dir: str, save_path: str):
    stage = Usd.Stage.CreateNew(save_path)
    root_prim = stage.DefinePrim("/Root")
    stage.SetDefaultPrim(root_prim)
    materials_scope = stage.DefinePrim("/Root/Materials", "Scope")
    
    obj = mesh["xyz"]
    object_name = "infer_" + str(uuid.uuid1())
    object_name = object_name.replace("-", "_")
    object_name = object_name.replace("'", "")
    
    v, faces, vt, mat = read_mesh_attr(obj)
        
    floor_path = f"/Root/{object_name}"
    floor_mesh = UsdGeom.Mesh.Define(stage, floor_path)
    
    points = [Gf.Vec3f(float(point[0]), float(point[1]), float(point[2])) for point in v]
    floor_mesh.CreatePointsAttr(points)
    face_vertex_counts = [len(face) for face in faces]
    floor_mesh.CreateFaceVertexCountsAttr(face_vertex_counts)
    face_vertex_indices = [int(index) for face in faces for index in face]
    floor_mesh.CreateFaceVertexIndicesAttr(face_vertex_indices)
    
    if vt is not None and len(vt) > 0:
        tex_coords = UsdGeom.PrimvarsAPI(floor_mesh).CreatePrimvar(
            "st", Sdf.ValueTypeNames.TexCoord2fArray, UsdGeom.Tokens.varying
        )
        tex_coords.Set([Gf.Vec2f(float(uv[0]), float(uv[1])) for uv in vt])
    
    material_name = f"{object_name}_material"
    material_path = f"{materials_scope.GetPath()}/{material_name}"
    material = UsdShade.Material.Define(stage, material_path)
    
    shader = UsdShade.Shader.Define(stage, f"{material_path}/shader")
    shader.SetSourceAsset("UsdPreviewSurface", "usdShade")
        
    material.CreateSurfaceOutput().ConnectToSource(
        shader.ConnectableAPI(), "surface"
    )
    
    UsdShade.MaterialBindingAPI(floor_mesh).Bind(material)

    stage.Save()
    return stage

def layout_prompt(objects, room_type):
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

def layout_from_retrieve(prompt, lora_engine, request_config, data_dir, res_dir, usd_descript=False):
    print("SPATIAL COMPUTING:")
    room_type = prompt['room_type']
    obj_inf = prompt['objects']
    obj_res = {}
    scene_inf = {}
    for key in obj_inf.keys():
        label = key.capitalize()
        if not label in obj_res.keys():
            obj_res[label] = []
        if not label in scene_inf.keys():
            scene_inf[label] = []
        for obj in obj_inf[label]:
            scene_inf[label].append({
                'object_name': obj['object_name'],
                'path': obj['path']
                })
            if usd_descript:
                obj_res[label].append({
                    'bbox': obj['bbox'],
                    'description': obj['description']
                })
            else:
                obj_res[label].append({
                    'bbox': obj['bbox']
                })
    infer_request = InferRequest(messages=[{'role': 'user', 'content': layout_prompt(room_type, obj_res)}])
    resp_list = lora_engine.infer([infer_request], request_config)
    layout = str_to_dict(resp_list[0].choices[0].message.content)
    print(layout)

    print("SCENE BUILDING:")
    scene_res = {}
    for obj in layout.keys():
        if obj != "Floor":
            if not obj in scene_res.keys():
                scene_res[obj] = []
            for i in range(len(layout[obj])):
                try:
                    scene_res[obj].append(scene_inf[obj.capitalize()][i])
                    scene_res[obj][i]['position'] = layout[obj][i]['position']
                    scene_res[obj][i]['rotation'] = layout[obj][i]['rotation']
                except:
                    print("LAYOUT ERROR!!!")

    scene = {"meshes": layout["Floor"], "objects": scene_res}
    os.makedirs(res_dir, exist_ok=True)
    save_json_file(scene, os.path.join(res_dir, "scene.json"))
    create_scene(scene["objects"], data_dir, os.path.join(res_dir, "object.usda"))
    create_floor(scene["meshes"], data_dir, os.path.join(res_dir, "floor.usda"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, default=None, required=True, help="Path to the base model")
    parser.add_argument('--lora_checkpoint', default=None, required=True, type=str, help="Path to the LoRA checkpoint")
    parser.add_argument('--retrieval', type=str, default=None, required=True, help="Path to the retrieval results")
    parser.add_argument('--res_dir', default=None, required=True, type=str, help="Path to scene res")
    args = parser.parse_args()
    model_path = args.model_path
    lora_checkpoint = args.lora_checkpoint
    data_dir = os.path.join(os.getcwd(), "data")
    if "disp" in lora_checkpoint:
        usd_descript = True
    else:
        usd_descript = False

    # Perform inference using the native PyTorch engine
    lora_engine = PtEngine(model_path, adapters=[lora_checkpoint])
    request_config = RequestConfig(max_tokens=4096, temperature=0)

    input = args.retrieval
    output_dir = args.res_dir
    os.makedirs(output_dir, exist_ok=True)

    prompt = read_json_file(input)
    layout_from_retrieve(prompt, lora_engine, request_config, data_dir, output_dir, usd_descript)



