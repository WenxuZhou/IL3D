import argparse
import sys
import os
import bpy
from math import radians


def clear_mv():
    for obj in bpy.data.objects:
        if obj.name not in ['Camera', 'Sun', 'Sun.001', 'Sun.002', 'Empty']:
            try:
                if obj.users_collection:
                    for coll in obj.users_collection:
                        coll.objects.unlink(obj)
                bpy.data.objects.remove(obj)
            except Exception as e:
                print(f"Error cleaning object {obj.name}: {e}")


def import_usda_files(folder_path):
    usda_files = [f for f in os.listdir(folder_path) if f.lower().endswith('.usda')]
    if not usda_files:
        print(f"No USDA files found in folder: {folder_path}")
        return False
        
    print(f"Found {len(usda_files)} USDA files to import")
    
    for usda_file in usda_files:
        file_path = os.path.join(folder_path, usda_file)
        try:
            bpy.ops.wm.usd_import(
                filepath=file_path,
                import_cameras=False,
                import_lights=False
            )
            print(f"Successfully imported: {usda_file}")
        except Exception as e:
            print(f"Failed to import {usda_file}: {e}")
    
    return True


def get_scene_center():
    min_x, min_y, min_z = float('inf'), float('inf'), float('inf')
    max_x, max_y, max_z = -float('inf'), -float('inf'), -float('inf')
    has_mesh = False
    
    for obj in bpy.context.scene.objects:
        if obj.type == 'MESH':
            has_mesh = True
            bbox = [obj.matrix_world @ v.co for v in obj.data.vertices]
            for v in bbox:
                min_x = min(min_x, v.x)
                max_x = max(max_x, v.x)
                min_y = min(min_y, v.y)
                max_y = max(max_y, v.y)
                min_z = min(min_z, v.z)
                max_z = max(max_z, v.z)
    
    if not has_mesh:
        return (0, 0, 0)
    return (
        (min_x + max_x) / 2,
        (min_y + max_y) / 2,
        (min_z + max_z) / 2
    )


def get_scene_radius(scene_center):
    max_distance = 0.0
    
    for obj in bpy.context.scene.objects:
        if obj.type == 'MESH':
            for v in obj.data.vertices:
                world_pos = obj.matrix_world @ v.co
                distance = (
                    (world_pos.x - scene_center[0])**2 +
                    (world_pos.y - scene_center[1])**2 +
                    (world_pos.z - scene_center[2])**2
                )**0.5
                max_distance = max(max_distance, distance)
    
    return max_distance


def render_function(model_folder, texture_file, scale, format, color_depth, view_type, views):
    if not os.path.isdir(model_folder):
        print(f"Error: Model folder does not exist or is not a directory -> {model_folder}")
        return None
    if texture_file and not os.path.exists(texture_file):
        print(f"Error: Texture file does not exist -> {texture_file}")
        return None
    
    save_root = os.path.abspath(model_folder)
    os.makedirs(save_root, exist_ok=True)
    print(f"All rendering results will be saved to: {save_root}")

    if not import_usda_files(model_folder):
        print("No valid USDA files imported, aborting rendering")
        return None

    if scale != 1.0:
        for obj in bpy.context.scene.objects:
            if obj.type == 'MESH':
                obj.scale = (scale, scale, scale)
        bpy.context.view_layer.update()
        print(f"Model scaled by: {scale}x")

    scene_center = get_scene_center()
    scene_radius = get_scene_radius(scene_center)
    
    max_z = -float('inf')
    has_mesh = False
    for obj in bpy.context.scene.objects:
        if obj.type == 'MESH':
            has_mesh = True
            for v in obj.data.vertices:
                world_z = (obj.matrix_world @ v.co).z
                if world_z > max_z:
                    max_z = world_z
    if not has_mesh:
        max_z = scene_center[2] + scene_radius * 0.5
    
    print(f"Scene center: {scene_center}, Scene radius: {scene_radius:.2f}, Scene max Z: {max_z:.2f}")

    scene = bpy.context.scene
    scene.render.engine = 'CYCLES'
    scene.cycles.samples = 20
    scene.cycles.device = 'GPU'
    
    scene.render.resolution_x = 512
    scene.render.resolution_y = 512
    scene.render.resolution_percentage = 100
    
    scene.render.film_transparent = True
    scene.render.image_settings.file_format = format
    scene.render.image_settings.color_depth = color_depth
    if format == 'PNG':
        scene.render.image_settings.compression = 15

    for obj in scene.objects:
        if obj.type == 'MESH':
            obj.cycles.shadow_visibility = False
            obj.cycles.camera_visibility = True
            obj.cycles.diffuse_visibility = True

    if 'World' in bpy.data.worlds:
        world = bpy.data.worlds['World']
        world.use_nodes = True
        bg_node = world.node_tree.nodes['Background']
        if view_type == 'indoor':
            bg_node.inputs[0].default_value = (0.1, 0.1, 0.1, 1.0)
            bg_node.inputs[1].default_value = 0.2
        else:
            bg_node.inputs[0].default_value = (0.75, 0.75, 0.75, 1.0)
            bg_node.inputs[1].default_value = 0.5

    def parent_camera_to_empty(camera):
        empty = bpy.data.objects.new("CameraEmpty", None)
        empty.location = scene_center
        scene.collection.objects.link(empty)
        camera.parent = empty
        
        track_constraint = camera.constraints.new(type='TRACK_TO')
        track_constraint.track_axis = 'TRACK_NEGATIVE_Z'
        track_constraint.up_axis = 'UP_Y'
        track_constraint.target = empty
        return empty

    cam = scene.objects.get('Camera')
    if not cam:
        cam_data = bpy.data.cameras.new("CameraData")
        cam = bpy.data.objects.new("Camera", cam_data)
        scene.collection.objects.link(cam)
    scene.camera = cam
    
    if view_type == 'outdoor':
        camera_distance = scene_radius * 2.0
        cam.location = (0, camera_distance, camera_distance * 0.3)
        light_distance = scene_radius * 3.0
        light_type = 'SUN'
        light_energy = 2.0
        light_positions = [
            (0, light_distance, light_distance * 0.5),
            (light_distance * 0.866, -light_distance * 0.5, light_distance * 0.5),
            (-light_distance * 0.866, -light_distance * 0.5, light_distance * 0.5)
        ]
    else:
        camera_distance = min(scene_center) * 0.5 if min(scene_center) != 0 else scene_radius * 1.5
        cam.location = (0, camera_distance, camera_distance * 0.1)
        light_type = 'POINT'
        light_energy = 100.0
        offset_z = scene_radius * 0.1
        light_positions = [
            (scene_center[0], scene_center[1], max_z - offset_z),
            (scene_center[0] - scene_radius * 0.2, scene_center[1], max_z - offset_z),
            (scene_center[0] + scene_radius * 0.2, scene_center[1], max_z - offset_z)
        ]
    
    print(f"Setting camera distance to: {camera_distance:.2f} (View type: {view_type})")
    cam.data.angle = radians(60)
    cam_empty = parent_camera_to_empty(cam)
    cam_empty.rotation_euler[2] = radians(330)

    for i, pos in enumerate(light_positions):
        light_data = bpy.data.lights.new(f"Light_{i}", type=light_type)
        light_data.energy = light_energy
        if light_type == 'POINT':
            light_data.shadow_soft_size = scene_radius * 0.1
        light = bpy.data.objects.new(f"Light_{i}", light_data)
        light.location = pos
        scene.collection.objects.link(light)
        
        track_constraint = light.constraints.new(type='TRACK_TO')
        track_constraint.track_axis = 'TRACK_NEGATIVE_Z'
        track_constraint.up_axis = 'UP_Y'
        track_constraint.target = cam_empty

    if texture_file:
        def create_texture_material(texture_path):
            mat = bpy.data.materials.new(name="TextureMaterial")
            mat.use_nodes = True
            
            nodes = mat.node_tree.nodes
            links = mat.node_tree.links
            for node in nodes:
                nodes.remove(node)
            
            tex_node = nodes.new(type='ShaderNodeTexImage')
            diffuse_node = nodes.new(type='ShaderNodeBsdfDiffuse')
            output_node = nodes.new(type='ShaderNodeOutputMaterial')
            
            try:
                tex_node.image = bpy.data.images.load(texture_path)
            except Exception as e:
                print(f"Failed to load texture: {e}")
                return None
            
            links.new(tex_node.outputs['Color'], diffuse_node.inputs['Color'])
            links.new(diffuse_node.outputs['BSDF'], output_node.inputs['Surface'])
            return mat

        texture_mat = create_texture_material(texture_file)
        if texture_mat:
            for obj in scene.objects:
                if obj.type == 'MESH':
                    if obj.data.materials:
                        obj.data.materials[0] = texture_mat
                    else:
                        obj.data.materials.append(texture_mat)
            print("Texture material applied successfully")
        else:
            print("Warning: Failed to create texture material, model will use default material")
    else:
        print("No texture file provided, will use USDA model's built-in material (if any)")

    stepsize = 360.0 / views
    print(f"Starting rendering of {views} views, rotating {stepsize:.1f} degrees per view")

    for i in range(views):
        current_angle = stepsize * i
        angle_str = f"{int(current_angle):03d}"
        scene.render.filepath = os.path.join(save_root, f"image_{angle_str}")
        bpy.ops.render.render(write_still=True)
        print(f"Completed rendering for view {angle_str} degrees")
        cam_empty.rotation_euler[2] += radians(stepsize)

    clear_mv()
    print(f"\nAll views rendered successfully! All files saved to: {save_root}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Blender 4.5.2 USDA Model Multi-view Color Rendering Script")
    parser.add_argument('--views', type=int, default=4, help="Number of rendering views (default: 12, evenly divided 360 degrees)")
    parser.add_argument('--output_folder', type=str, required=True, help="Output folder path (required)")
    parser.add_argument('--scale', type=float, default=1.0, help="Model scale factor (default: 1.0, no scaling)")
    parser.add_argument('--color_depth', type=str, default='8', help="Output image color depth (default: 8-bit, optional: 16/32-bit)")
    parser.add_argument('--format', type=str, default='PNG', choices=['PNG', 'OPEN_EXR', 'JPEG'], 
                        help="Output file format (default: PNG, optional: OPEN_EXR/JPEG)")
    parser.add_argument('--model_folder', type=str, default=None, help="USDA model folder path")
    parser.add_argument('--texture_file', type=str, default=None, help="External texture file path (optional, e.g., xxx.png)")
    parser.add_argument('--view_type', type=str, default='outdoor', choices=['indoor', 'outdoor'], 
                        help="Render view type (default: outdoor, optional: indoor)")

    if '--' in sys.argv:
        argv = sys.argv[sys.argv.index("--") + 1:]
    else:
        argv = []
    args = parser.parse_args(argv)
    
    path = args.output_folder
    rooms = ["bedroom", "livingroom", "kitchen", "bathroom", "diningroom", "balcony"]
    for room in rooms:
        ids = os.listdir(os.path.join(path, room)) if os.path.exists(os.path.join(path, room)) else []
        for id in ids:
            current_model_folder = os.path.join(path, room, id)
            if not os.path.isdir(current_model_folder):
                print(f"Error: The provided model folder does not exist or is not a directory: {current_model_folder}")
                sys.exit(1)

            usda_files = [f for f in os.listdir(current_model_folder) if f.lower().endswith('.usda')]
            if not usda_files:
                print(f"Warning: No USDA files found in the specified folder: {current_model_folder}")

            for obj_name in ['Cube', 'Light']:
                obj = bpy.data.objects.get(obj_name)
                if obj:
                    if obj.users_collection:
                        for coll in obj.users_collection:
                            coll.objects.unlink(obj)
                    bpy.data.objects.remove(obj)
                    print(f"Deleted Blender default object: {obj_name}")

            render_function(
                model_folder=current_model_folder,
                texture_file=args.texture_file,
                scale=args.scale,
                format=args.format,
                color_depth=args.color_depth,
                view_type=args.view_type,
                views=args.views
            )
