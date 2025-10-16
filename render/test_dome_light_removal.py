#!/usr/bin/env python3
"""
测试脚本：验证 dome light 删除功能
"""

import os
import sys
import tempfile
import shutil
from pathlib import Path

from pxr import Usd, UsdLux, UsdGeom
from orca_gym.tools.assets.usdz_to_xml import remove_dome_lights_from_stage

def create_test_usdc_with_dome_light():
    """创建一个包含 dome light 的测试 USD 文件"""
    # 创建临时文件
    temp_file = tempfile.NamedTemporaryFile(suffix='.usdc', delete=False)
    temp_file.close()
    
    # 创建新的 stage
    stage = Usd.Stage.CreateNew(temp_file.name)
    
    # 添加一个 dome light
    dome_light = UsdLux.DomeLight.Define(stage, "/DomeLight")
    dome_light.CreateIntensityAttr(1.0)
    dome_light.CreateColorAttr((1.0, 1.0, 1.0))
    
    # 添加一个简单的几何体作为参考
    sphere = UsdGeom.Sphere.Define(stage, "/Sphere")
    sphere.CreateRadiusAttr(1.0)
    
    # 保存 stage
    stage.GetRootLayer().Save()
    
    return temp_file.name

def test_dome_light_removal():
    """测试 dome light 删除功能"""
    print("=== 测试 Dome Light 删除功能 ===")
    
    # 创建测试文件
    test_file = create_test_usdc_with_dome_light()
    print(f"创建测试文件: {test_file}")
    
    try:
        # 打开文件并检查初始状态
        stage = Usd.Stage.Open(test_file)
        initial_prims = list(stage.Traverse())
        print(f"初始 prims 数量: {len(initial_prims)}")
        
        # 检查是否有 dome light
        dome_lights_before = [prim for prim in initial_prims if prim.GetTypeName() == 'DomeLight']
        print(f"删除前 dome lights 数量: {len(dome_lights_before)}")
        for light in dome_lights_before:
            print(f"  - {light.GetPath()}")
        
        # 执行删除操作
        removed_lights = remove_dome_lights_from_stage(stage)
        print(f"删除的 dome lights: {removed_lights}")
        
        # 重新打开文件检查结果
        stage_after = Usd.Stage.Open(test_file)
        final_prims = list(stage_after.Traverse())
        print(f"删除后 prims 数量: {len(final_prims)}")
        
        # 检查是否还有 dome light
        dome_lights_after = [prim for prim in final_prims if prim.GetTypeName() == 'DomeLight']
        print(f"删除后 dome lights 数量: {len(dome_lights_after)}")
        
        # 验证结果
        if len(dome_lights_after) == 0 and len(removed_lights) == len(dome_lights_before):
            print("✅ 测试通过：所有 dome lights 已成功删除")
            return True
        else:
            print("❌ 测试失败：dome lights 删除不完整")
            return False
            
    except Exception as e:
        print(f"❌ 测试过程中出现错误: {e}")
        return False
    finally:
        # 清理测试文件
        if os.path.exists(test_file):
            os.unlink(test_file)
            print(f"清理测试文件: {test_file}")

if __name__ == "__main__":
    success = test_dome_light_removal()
    sys.exit(0 if success else 1)
