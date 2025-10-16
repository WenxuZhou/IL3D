# Usage Instructions

1. OrcaStudio Engine

Follow the OrcaStudio documentation to set up the environment and launch the OrcaLab engine.  

2. Verify that the environment is correctly configured and that ray-tracing runs without issues.

```bash
   python test_dome_light_removal.py
```

3. Render multi-view results of the scene.

```bash
   python scene_render.py
```

You can modify the parameters of `generate_camera_coordinates()` to adjust the rendering settings.

4. Pack your own USD scene.

We primarily use the `OrcaStudio` engine to import scenes and process them into PAK files that can be used in physics simulation software. Please refer to for specific usage http://www.orca3d.cn/ .
