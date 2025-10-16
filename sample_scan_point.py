import argparse
import os
import numpy as np
from pxr import Usd, UsdGeom, Gf
import open3d as o3d
from open3d.geometry import TriangleMesh
import matplotlib.cm as cm


def read_point_cloud_from_txt(file_path):
    try:
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")
        if not file_path.endswith('.txt'):
            raise ValueError("Please provide a file in txt format")
        point_cloud = np.loadtxt(file_path)
        if point_cloud.ndim != 2 or point_cloud.shape[1] != 4:
            raise ValueError(f"Invalid point cloud format. Expected n*4 array, got {point_cloud.shape}")
        return point_cloud
    except Exception as e:
        print(f"Error reading point cloud file: {str(e)}")
        return None


def get_colors_for_labels(labels):
    unique_labels = np.unique(labels)
    num_labels = len(unique_labels)
    cmap = cm.get_cmap('tab10', num_labels)
    label_to_color = {}
    for i, label in enumerate(unique_labels):
        color = cmap(i)[:3]
        label_to_color[label] = color
    point_colors = np.array([label_to_color[label] for label in labels])
    return point_colors


def visualize_with_open3d(point_cloud, bboxes, title="Labeled Point Cloud with Bounding Boxes"):
    if point_cloud is None:
        print("Cannot visualize, point cloud data is empty")
        return
    coordinates = point_cloud[:, :3]
    labels = point_cloud[:, 3]
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(coordinates)
    colors = get_colors_for_labels(labels)
    pcd.colors = o3d.utility.Vector3dVector(colors)
    
    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name=title)
    vis.add_geometry(pcd)
    
    unique_labels = np.unique(labels)
    cmap = cm.get_cmap('tab10', len(unique_labels))
    label_color_map = {label: cmap(i)[:3] for i, label in enumerate(unique_labels)}
    
    for bbox in bboxes:
        label, min_x, min_y, min_z, max_x, max_y, max_z = bbox
        aabb = o3d.geometry.AxisAlignedBoundingBox(
            min_bound=[min_x, min_y, min_z],
            max_bound=[max_x, max_y, max_z]
        )
        aabb.color = label_color_map.get(label, [1, 0, 0])
        vis.add_geometry(aabb)
    
    opt = vis.get_render_option()
    opt.background_color = [1.0, 1.0, 1.0]
    opt.point_size = 3
    vis.run()
    vis.destroy_window()


def compute_mesh_surface_area(mesh):
    mesh.compute_triangle_normals()
    triangles = np.asarray(mesh.triangles)
    vertices = np.asarray(mesh.vertices)
    areas = []
    for tri in triangles:
        v0, v1, v2 = vertices[tri[0]], vertices[tri[1]], vertices[tri[2]]
        a = np.linalg.norm(v1 - v0)
        b = np.linalg.norm(v2 - v1)
        c = np.linalg.norm(v2 - v0)
        s = (a + b + c) / 2
        area = np.sqrt(s * (s - a) * (s - b) * (s - c))
        areas.append(area)
    return sum(areas)


def compute_bounding_box(points):
    min_x, min_y, min_z = np.min(points, axis=0)
    max_x, max_y, max_z = np.max(points, axis=0)
    return min_x, min_y, min_z, max_x, max_y, max_z


def main(input_folder, num_points, output_dir, vis):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    stage = Usd.Stage.CreateInMemory()
    usda_files = [f for f in os.listdir(input_folder) if f.endswith('.usda')]
    if not usda_files:
        print("No USDA files found in the input folder.")
        return

    meshes = []
    surface_areas = []
    bboxes = []
    for usda_file in usda_files:
        file_path = os.path.join(input_folder, usda_file)
        prim_name = os.path.splitext(usda_file)[0]
        ref_prim = stage.DefinePrim(f'/imported_{prim_name}', 'Xform')
        ref_prim.GetReferences().AddReference(file_path)

    all_points = []
    all_labels = []
    semantic_label = 0

    for prim in stage.Traverse():
        if prim.IsA(UsdGeom.Mesh):
            mesh = UsdGeom.Mesh(prim)

            xform_cache = UsdGeom.XformCache()
            world_transform = xform_cache.GetLocalToWorldTransform(prim)

            points = np.array(mesh.GetPointsAttr().Get(), dtype=np.float64)
            transformed_points = []
            for point in points:
                gf_point = Gf.Vec3d(point[0], point[1], point[2])
                transformed = world_transform.Transform(gf_point)
                transformed_points.append([transformed[0], transformed[1], transformed[2]])
            transformed_points = np.array(transformed_points)

            face_indices = np.array(mesh.GetFaceVertexIndicesAttr().Get(), dtype=np.int32)
            face_counts = np.array(mesh.GetFaceVertexCountsAttr().Get(), dtype=np.int32)

            o3d_mesh = TriangleMesh()
            o3d_mesh.vertices = o3d.utility.Vector3dVector(transformed_points)

            triangles = []
            idx = 0
            for count in face_counts:
                if count == 3:
                    triangles.append([face_indices[idx], face_indices[idx+1], face_indices[idx+2]])
                elif count == 4:
                    triangles.append([face_indices[idx], face_indices[idx+1], face_indices[idx+2]])
                    triangles.append([face_indices[idx], face_indices[idx+2], face_indices[idx+3]])
                idx += count
            o3d_mesh.triangles = o3d.utility.Vector3iVector(triangles)

            surface_area = compute_mesh_surface_area(o3d_mesh)
            meshes.append((o3d_mesh, semantic_label))
            surface_areas.append(surface_area)

            min_x, min_y, min_z, max_x, max_y, max_z = compute_bounding_box(transformed_points)
            bboxes.append((semantic_label, min_x, min_y, min_z, max_x, max_y, max_z))

            semantic_label += 1

    bbox_output_file = os.path.join(output_dir, 'bounding_boxes.txt')
    with open(bbox_output_file, 'w') as f:
        for bbox in bboxes:
            f.write(' '.join(map(str, bbox)) + '\n')

    total_area = sum(surface_areas)
    if total_area == 0:
        print("Total surface area is zero, cannot allocate points.")
        return
    point_allocations = [
        int(num_points * (area / total_area)) for area in surface_areas
    ]
    point_allocations = [max(1, n) for n in point_allocations]
    total_allocated = sum(point_allocations)
    if total_allocated > num_points:
        point_allocations = [
            int(n * (num_points / total_allocated)) for n in point_allocations
        ]
        point_allocations = [max(1, n) for n in point_allocations]
    elif total_allocated < num_points:
        remaining = num_points - total_allocated
        for i in range(remaining):
            point_allocations[i % len(point_allocations)] += 1

    transformed_bboxes = []
    for (label, orig_min_x, orig_min_y, orig_min_z, orig_max_x, orig_max_y, orig_max_z) in bboxes:
        x1 = -orig_min_x / 100.0
        x2 = -orig_max_x / 100.0
        new_min_x = min(x1, x2)
        new_max_x = max(x1, x2)
        
        z1 = orig_min_y / 100.0
        z2 = orig_max_y / 100.0
        new_min_z = min(z1, z2)
        new_max_z = max(z1, z2)
        
        y1 = orig_min_z / 100.0
        y2 = orig_max_z / 100.0
        new_min_y = min(y1, y2)
        new_max_y = max(y1, y2)
        
        transformed_bboxes.append((label, new_min_x, new_min_y, new_min_z, new_max_x, new_max_y, new_max_z))

    for (o3d_mesh, label), num_points_mesh in zip(meshes, point_allocations):
        try:
            pcd = o3d_mesh.sample_points_poisson_disk(number_of_points=num_points_mesh)
        except Exception as e:
            print(f"Poisson sampling failed for mesh {label}: {str(e)}, falling back to uniform sampling")
            pcd = o3d_mesh.sample_points_uniformly(number_of_points=num_points_mesh)

        points_np = np.asarray(pcd.points)
        points_np[:, [1, 2]] = points_np[:, [2, 1]]
        points_np[:, 0] = -points_np[:, 0]
        points_np = points_np / 100.0

        all_points.append(points_np)
        all_labels.append(np.full(points_np.shape[0], label, dtype=int))

    combined_points = np.vstack(all_points)
    combined_labels = np.hstack(all_labels)

    if len(combined_points) > num_points:
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(combined_points)
        indices = np.random.choice(len(combined_points), num_points, replace=False)
        combined_points = combined_points[indices]
        combined_labels = combined_labels[indices]

    labeled_point_cloud = np.column_stack((combined_points, combined_labels))

    output_file = os.path.join(output_dir, 'scene.txt')
    np.savetxt(output_file, labeled_point_cloud)

    if vis:
        point_cloud = read_point_cloud_from_txt(output_file)
        if point_cloud is not None:
            visualize_with_open3d(point_cloud, transformed_bboxes, title="Labeled Point Cloud with Bounding Boxes")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Sample labeled point clouds from USDA files with 3D bounding boxes')
    parser.add_argument('--input_folder', type=str, required=True, help='Path to input USDA folder')
    parser.add_argument('--num_points', default=8192, type=int, help='Total number of points to sample from all objects')
    parser.add_argument('--output_dir', type=str, required=True, help='Output directory for point clouds and bounding boxes')
    parser.add_argument('--vis', action='store_true', help='Visualize labeled point clouds with bounding boxes')
    args = parser.parse_args()
    main(
        input_folder=args.input_folder,
        num_points=args.num_points,
        output_dir=args.output_dir,
        vis=args.vis
    )
