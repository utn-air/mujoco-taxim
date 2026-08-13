# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import logging
import os

import mujoco as mj

from mujoco.viewer import launch
import glfw
import threading
from math import pi
import numpy as np 

# Taxim imports 
from TaximSensor import TaximSensor

np.set_printoptions(precision=3, suppress=True)
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
CONTROL_INCREMENT = 0.0005  # Amount to move joints per keypress

SCENE_ROOT = os.path.join(CURRENT_DIR, "xml")
SCENE_FILE = os.path.join(SCENE_ROOT, "touch_playground_rot.xml")
TACTO_DIR = os.path.join(CURRENT_DIR, "..", "tacto", "assets")
CONFIG_DIR = os.path.join(CURRENT_DIR, "..", "tacto", "cfg")
CAN_XYZ_LIM = 0.012

qpos_holder = {
    "xx": 0.0,
    "yy": 0.0,
    "zz": 0.0,
}


def key_callback(window, key, scancode, action, mods):
    """
    GLFW key callback to interactively control joint positions.
    """
    if action != glfw.PRESS and action != glfw.REPEAT:
        return  # Ignore key releases

    elif key == glfw.KEY_R:
        qpos_holder["xx"] = min(qpos_holder["xx"] + CONTROL_INCREMENT, CAN_XYZ_LIM)
    elif key == glfw.KEY_F:
        qpos_holder["xx"] = max(qpos_holder["xx"] - CONTROL_INCREMENT, -CAN_XYZ_LIM)
    elif key == glfw.KEY_W:
        qpos_holder["yy"] = min(qpos_holder["yy"] + CONTROL_INCREMENT, CAN_XYZ_LIM)
    elif key == glfw.KEY_S:
        qpos_holder["yy"] = max(qpos_holder["yy"] - CONTROL_INCREMENT, -CAN_XYZ_LIM)
    elif key == glfw.KEY_D:
        qpos_holder["zz"] = min(qpos_holder["zz"] + CONTROL_INCREMENT, CAN_XYZ_LIM)
    elif key == glfw.KEY_A:
        qpos_holder["zz"] = max(qpos_holder["zz"] - CONTROL_INCREMENT, -CAN_XYZ_LIM)
    elif key == glfw.KEY_J:
        data.qpos[model.joint("master_joint_x").qposadr] += (pi/20)
    elif key == glfw.KEY_K:
        data.qpos[model.joint("master_joint_y").qposadr] += (pi/20)
    elif key == glfw.KEY_L:
        data.qpos[model.joint("master_joint_z").qposadr] += (pi/20)


log = logging.getLogger(__name__)

def main():
    
    # For allowing keyboard input in the viewer
    global model, data, camera, scene
    
    #--------------#
    # MuJoCo Setup #
    #--------------#
    model = mj.MjModel.from_xml_path(
        str(SCENE_FILE)
    )
    data = mj.MjData(model)
    model.opt.timestep = 0.001
    mj.mj_step(model, data) # step to initialize object poses
    

    #--------------------#
    # Taxim Sensor Setup #
    #--------------------#

    # Sensor initialization is very simple. 
    # You can choose between "digit" and "gelsight_r1.5" sensor types.
    # You can specify a background image saved as npz where the image is stored under the key "bg", in shape [N, H, W, C].
    # If it's not provided, the sensor will use a default background appropriate to the sensor type.
    # bg_index specifies which background image to use in the provided/default npz file.
    # Resize does what it says, in the format (new_h, new_w); output image will be scaled to this.
    # preprocess_bg applies a gaussian blur to the background image.
    sim = TaximSensor(sensor_type="digit", gelmap_file="/home/sbien/Documents/Development/V2T/TactoSampler/taxim_files/gelmap_alt.npy", bg_file=None, bg_index=0, resize=None, preprocess_bg=True)

    # For the sensor to work, the desired object in mujoco needs to be added.
    sim.add_geom_mujoco("can_geom", model=model, data=data, mesh_name="can_mesh")
    # Additionally, the site that acts as the surface of the sensor needs to be added.
    # The site should align with the surface of the sensor, as Taxim determines which part to render using the site's xy-plane.
    sim.add_camera_mujoco("left_tacto_pad", model, data)
    sim.set_sensor_pad_geom("finger_1_left_0")
    
    #----------------------------#
    # MuJoCo Visualization Setup #
    #----------------------------#
    # Create GLFW window
    if not glfw.init():
       raise Exception("Failed to initialize GLFW")
    window = glfw.create_window(1200, 800, "MuJoCo Interactive Viewer", None, None)
    if not window:
        glfw.terminate()
        raise RuntimeError("GLFW window creation failed")

    glfw.make_context_current(window)
    glfw.set_key_callback(window, key_callback)
    options = mj.MjvOption()
    scene = mj.MjvScene(model, maxgeom=1000)
    context = mj.MjrContext(model, mj.mjtFontScale.mjFONTSCALE_150)
    camera = mj.MjvCamera()
    camera.type = mj.mjtCamera.mjCAMERA_TRACKING
    camera.trackbodyid = mj.mj_name2id(model, mj.mjtObj.mjOBJ_BODY, "master")

    while not glfw.window_should_close(window):
        # step simulation of mujoco model
        data.qpos[model.joint("xx").qposadr] = qpos_holder['xx']
        data.qpos[model.joint("yy").qposadr] = qpos_holder['yy']
        data.qpos[model.joint("zz").qposadr] = qpos_holder['zz']
        mj.mj_step(model, data)
        # Optionally, you can cycle through the background images.
        # sim.change_bg((sim.bg_index + 1) % sim.bg_len)
        sim.render_taxim(model, 
                         data, 
                         shadow=True, # add shadow to rendered image?
                         get_depth=True, # Return depth image?
                         img_noise_sigma=5, # Gaussian noise to add to the rgb image 
                         pcn_add_noise=False, # Add noise to the returned point cloud-normal?
                         visualize=True, # visualize the render? 
                         cycle_bg=True) # cycle through the background image after every render?

        # Render the scene
        viewport_width, viewport_height = glfw.get_framebuffer_size(window)
        mj.mjv_updateScene(
            model, data, options, None, camera, mj.mjtCatBit.mjCAT_ALL.value, scene
        )
        mj.mjr_render(mj.MjrRect(0, 0, viewport_width, viewport_height), scene, context)

        glfw.swap_buffers(window)
        glfw.poll_events()

if __name__ == "__main__":
    # hydra.initialize_config_dir(CONFIG_DIR, version_base=None)
    # script_dir = CURRENT_DIR
    # cfg = hydra.compose("digit.yaml", overrides=[f"base_dir={script_dir}"])
    main()
