import os
import shutil
import numpy as np
from tqdm import tqdm
from pxr import Usd, UsdGeom, Gf
from meta_data import read_json_file, save_json_file


def get_usdz_bbox_dimensions(usdz_file_path):
    try:
        stage = Usd.Stage.Open(usdz_file_path)
        if not stage:
            print(f"Can't Open Flie: {usdz_file_path}")
            return None

        root_prim = stage.GetPseudoRoot()
        bbox_cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), ["default"])
        bbox = bbox_cache.ComputeWorldBound(root_prim)
        min_extent = bbox.GetBox().GetMin()
        max_extent = bbox.GetBox().GetMax()
        width = max_extent[0] - min_extent[0]
        length = max_extent[1] - min_extent[1]
        height = max_extent[2] - min_extent[2]
            
        return (width/100.0, length/100.0, height/100.0)
        
    except Exception as e:
        print(f"Error: {str(e)}")
        return None

if __name__ == "__main__":
    mate_data = read_json_file("data/assets.json")
    res = []

    for data in tqdm(mate_data):
        path = data["path"]
        path = os.path.join("data", path)
        width, length, height = get_usdz_bbox_dimensions(path)
        volume = width * length * height
        data["mate_data"]["volume"] = volume 
        data["mate_data"]["width"] = width
        data["mate_data"]["length"] = length
        data["mate_data"]["height"] = height
        res.append(data)
    save_json_file(res, "assets.json")
    