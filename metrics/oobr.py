import pxr
import os
from pxr import Usd, UsdGeom, Gf
import argparse

def get_bbox(prim):
    bbox_cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), ["default"])
    bbox = bbox_cache.ComputeWorldBound(prim)
    range_box = bbox.ComputeAlignedBox()
    return (range_box.min, range_box.max)

def is_outside(bbox, floor_bounds, margin=5.0):
    b_min, b_max = bbox
    x_min, x_max, y_min, y_max, floor_z = floor_bounds
    
    if b_min[0] < (x_min - margin) or b_max[0] > (x_max + margin):
        return True
    if b_min[2] < (y_min - margin) or b_max[2] > (y_max + margin):
        return True
    if b_max[1] < (floor_z - margin):
        return True
        
    return False

def is_overlapping(bbox1, bbox2):
    min1, max1 = bbox1
    min2, max2 = bbox2
    
    for i in range(3):
        if max1[i] < min2[i] or max2[i] < min1[i]:
            return False
    return True

def calculate_bbox_volume(bbox):
    b_min, b_max = bbox
    volume = (b_max[0] - b_min[0]) * (b_max[1] - b_min[1]) * (b_max[2] - b_min[2])
    return max(volume, 0.0)

def calculate_intersection_volume(bbox1, bbox2):
    if not is_overlapping(bbox1, bbox2):
        return 0.0
    min1, max1 = bbox1
    min2, max2 = bbox2
    
    intersect_min = Gf.Vec3f(
        max(min1[0], min2[0]),
        max(min1[1], min2[1]),
        max(min1[2], min2[2])
    )
    intersect_max = Gf.Vec3f(
        min(max1[0], max2[0]),
        min(max1[1], max2[1]),
        min(max1[2], max2[2])
    )
    
    intersection_volume = (intersect_max[0] - intersect_min[0]) * \
                         (intersect_max[1] - intersect_min[1]) * \
                         (intersect_max[2] - intersect_min[2])
    return max(intersection_volume, 0.0)

def get_usdz_objects(stage):
    usdz_objects = []
    
    for prim in stage.Traverse():
        refs = prim.GetMetadata('references')
        if refs:
            for ref in refs.GetAddedOrExplicitItems():
                if str(ref.assetPath).endswith('.usdz'):
                    usdz_objects.append(prim)
                    break
    
    return usdz_objects

def analyze_usd_files(objects_file, floor_file):
    objects_stage = Usd.Stage.Open(objects_file)
    floor_stage = Usd.Stage.Open(floor_file)
    
    if not objects_stage:
        raise ValueError(f"Cannot open objects file: {objects_file}")
    if not floor_stage:
        raise ValueError(f"Cannot open floor file: {floor_file}")
    
    usdz_objects = get_usdz_objects(objects_stage)
    total_objects = len(usdz_objects)
    
    if total_objects == 0:
        print("Warning: No usdz objects found")
        return 0, 0, 0.0
    
    object_bboxes = [get_bbox(prim) for prim in usdz_objects]
    
    total_bbox_volume = 0.0
    for bbox in object_bboxes:
        total_bbox_volume += calculate_bbox_volume(bbox)
    
    floor_prim = None
    for prim in floor_stage.Traverse():
        if prim.GetTypeName() == "Mesh" and "floor" in prim.GetName().lower():
            floor_prim = prim
            break
    
    if not floor_prim:
        for prim in floor_stage.Traverse():
            if prim.GetTypeName() == "Mesh":
                floor_prim = prim
                break
    
    if not floor_prim:
        raise ValueError("No valid floor prim found in floor.usda")
    
    points_attr = floor_prim.GetAttribute("points")
    if not points_attr:
        raise ValueError("Floor prim has no points attribute")
    
    points = points_attr.Get()
    x_coords = [p[0] for p in points]
    y_coords = [p[2] for p in points]
    z_coords = [p[1] for p in points]
    
    x_min, x_max = min(x_coords), max(x_coords)
    y_min, y_max = min(y_coords), max(y_coords)
    floor_z = max(z_coords)
    floor_bounds = (x_min, x_max, y_min, y_max, floor_z)
    
    oob_count = 0
    for bbox in object_bboxes:
        if is_outside(bbox, floor_bounds):
            oob_count += 1
    
    total_intersection_volume = 0.0
    for i in range(total_objects):
        for j in range(i + 1, total_objects):
            intersection_vol = calculate_intersection_volume(object_bboxes[i], object_bboxes[j])
            total_intersection_volume += intersection_vol
    
    oor_ratio = 0.0
    if total_bbox_volume > 1e-9:
        oor_ratio = total_intersection_volume / total_bbox_volume
    
    return oob_count/total_objects, oor_ratio

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Check USD object BBOX: OOB count & OOR volume ratio")
    parser.add_argument("--scene_dir", type=str, default="infer_res/sft_FRONT3d_1.7b/balcony/2", help="Path to scene directory (contains object.usda and floor.usda)")
    args = parser.parse_args()

    object_path = os.path.join(args.scene_dir, "object.usda")
    floor_path = os.path.join(args.scene_dir, "floor.usda")
    oob_ratio, oor_ratio = analyze_usd_files(object_path, floor_path)
    
    print(f"OOB ratio (outside floor or below floor): {oob_ratio:.4f}")
    print(f"OOR ratio (intersection volume / total BBOX volume): {oor_ratio:.4f}")
