from orca_gym.scene.orca_gym_scene import OrcaGymScene
from orca_gym.scene.orca_gym_scene import Actor
import time
from orca_gym.scene.orca_gym_scene_runtime import OrcaGymSceneRuntime
from orca_gym.scripts.run_sim_loop import run_simulation
import numpy as np
import orca_gym.utils.rotations as rotations
from orca_gym.scripts.run_sim_loop import register_env
import gymnasium as gym
import sys
import os


def generate_camera_coordinates(center, distance, num_cameras):
    if num_cameras <= 0 or not isinstance(num_cameras, int):
        raise ValueError("Number of cameras must be a positive integer")
    if distance <= 0:
        raise ValueError("Distance must be a positive number")

    angle_from_ground = np.radians(15)
    
    radius = distance * np.cos(angle_from_ground)
    camera_z = center[2] + distance * np.sin(angle_from_ground)
    
    cameras = []
    for i in range(num_cameras):
        theta = 2 * np.pi * i / num_cameras
        x = center[0] + radius * np.cos(theta)
        y = center[1] + radius * np.sin(theta)
        z = camera_z
        
        pitch = 0
        yaw = theta + np.pi/2
        roll = 0
        cameras.append([[x, y, z], [roll, pitch, yaw]])
    
    return cameras

CAMERA_COORDINATES = generate_camera_coordinates([1.5, -3.5, 0], 10, 8)

def publish_scene(actor: dict) -> None:
    print("=============> publish scene ...")
    scene = OrcaGymScene("localhost:50051")

    scene.add_actor(Actor(
        name=actor["name"],
        spawnable_name=actor["spawnable_name"],
        position=actor["position"],
        rotation=actor["rotation"],
        scale=actor["scale"],
    ))
    print(f"    =============> Add actor {actor['name']} ...")
    time.sleep(0.1)

    for i, camera_coordinate in enumerate(CAMERA_COORDINATES):
        camera_name = f"camera_{i}"
        camera_actor = Actor(
            name=camera_name,
            spawnable_name="cameraviewport",
            position=np.array(camera_coordinate[0]),
            rotation=rotations.euler2quat(np.array(camera_coordinate[1])),
            scale=1.0,
        )
        scene.add_actor(camera_actor)
        print(f"    =============> Add actor {camera_name} ...")
        time.sleep(0.1)

    scene.publish_scene()
    time.sleep(1)

    for i, camera_coordinate in enumerate(CAMERA_COORDINATES):
        camera_name = f"camera_{i}"
        scene.make_camera_viewport_active(camera_name, "CameraViewport")
        print(f"    =============> Make camera viewport active {camera_name} ...")
        time.sleep(0.1)

    print("=============> publish scene done.")
    return scene

def clear_scene() -> None:
    print("=============> clear scene ...")
    scene = OrcaGymScene("localhost:50051")
    scene.publish_scene()
    time.sleep(1)
    scene.close()
    time.sleep(1)
    print("=============> clear scene done.")

if __name__ == "__main__":
    actors = {
        "demo": {
            "name": "demo",
            "spawnable_name": "demo_usda",
            "position": [0, 0, 0.0],
            "rotation": [0, 0, 0, 1],
            "scale": 0.01,
        },
    }

    for actor in actors.values():
        clear_scene()

        print("add actor: ", actor["name"])
        scene = publish_scene(actor=actor)

        orcagym_addr = "localhost:50051"
        agent_name = "NoRobot"
        env_name = "SimulationLoop"

        env_index = 0
        env_id, kwargs = register_env(orcagym_addr, 
                                      env_name, 
                                      env_index, 
                                      agent_name, 
                                      sys.maxsize)
        print("Registered environment: ", env_id)

        env = gym.make(env_id)        
        print("Starting simulation...")
        time.sleep(1)

        scene_runtime=OrcaGymSceneRuntime(scene)
        if scene_runtime is not None:
            if hasattr(env, "set_scene_runtime"):
                print("Setting scene runtime...")
                env.set_scene_runtime(scene_runtime)
            elif hasattr(env.unwrapped, "set_scene_runtime"):
                print("Setting scene runtime...")
                env.unwrapped.set_scene_runtime(scene_runtime)
        time.sleep(1)

        save_png_dir = os.path.join(os.path.dirname(__file__), "OrcaLabPNG")
        os.makedirs(save_png_dir, exist_ok=True)
        if hasattr(env, "get_frame_png"):
            env.get_frame_png(save_png_dir)
        elif hasattr(env.unwrapped, "get_frame_png"):
            env.unwrapped.get_frame_png(save_png_dir)
        else:
            print("get_frame_png is not supported")
        time.sleep(1)

        env.close()
        print("=============> Close environment.")
        time.sleep(1)
        print("=============> Close environment done.")

        scene.close()
        print("=============> Close scene.")
        time.sleep(1)
        print("=============> Close scene done.")