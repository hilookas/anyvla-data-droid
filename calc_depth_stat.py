import pickle
from tqdm import tqdm

with open("list.pkl", "rb") as f:
    mylist = pickle.load(f)

droid_base_path = "/mnt/20T/droid_raw_1.0.1"

import cv2
import numpy as np
import pyzed.sl as sl

def load_svo(svo_path, frame_max=None, frame_step=1):
    # Create a ZED camera object
    zed = sl.Camera()
    
    # Set configuration parameters
    init_params = sl.InitParameters()
    init_params.set_from_svo_file(svo_path)
    
    # Open the camera
    err = zed.open(init_params)
    if err != sl.ERROR_CODE.SUCCESS:
        print(f"Error {err}: Failed to open SVO file")
        return
    
    # Print camera info
    cam_info = zed.get_camera_information()
    fps = round(cam_info.camera_configuration.fps)
    height = round(cam_info.camera_configuration.resolution.height)
    width = round(cam_info.camera_configuration.resolution.width)
    print("ZED Model                 : {0}".format(cam_info.camera_model))
    print("ZED Serial Number         : {0}".format(cam_info.serial_number))
    print("ZED Camera Firmware       : {0}/{1}".format(cam_info.camera_configuration.firmware_version, cam_info.sensors_configuration.firmware_version))
    print("ZED Camera Resolution     : {0}x{1}".format(width, height))
    print("ZED Camera FPS            : {0}".format(fps))
    
    # Get intrinsics
    left_cam_calibration_params = zed.get_camera_information().camera_configuration.calibration_parameters.left_cam
    fx = left_cam_calibration_params.fx
    fy = left_cam_calibration_params.fy
    cx = left_cam_calibration_params.cx
    cy = left_cam_calibration_params.cy
    intrinsics = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])
    depth_scale = 1000
    
    yield intrinsics, depth_scale, fps, height, width
    
    # Create image
    image = sl.Mat()
    depth = sl.Mat()
    # point_cloud = sl.Mat()
    
    # Read frames
    frame_idx = 0
    while zed.grab() == sl.ERROR_CODE.SUCCESS:
        if frame_idx % frame_step == 0:
            zed.retrieve_image(image, sl.VIEW.LEFT) # Retrieve image
            zed.retrieve_measure(depth, sl.MEASURE.DEPTH) # Retrieve depth
            # zed.retrieve_measure(point_cloud, sl.MEASURE.XYZRGBA)
            
            # zed.retrieve_image(image_r, sl.VIEW.RIGHT) # Retrieve right image
            # zed.retrieve_measure(depth_r, sl.MEASURE.DEPTH_RIGHT) # Retrieve right depth
            # zed.retrieve_measure(point_cloud_r, sl.MEASURE.XYZRGBA_RIGHT)

            # Convert to numpy array
            image_array = image.get_data().copy() # 不加copy好像会导致内存泄漏
            depth_array = depth.get_data().copy()
            # point_cloud_array = point_cloud.get_data().copy()
            
            a = yield frame_idx, image_array, depth_array
            
            if frame_max is not None and frame_idx == frame_max:
                break
        
        frame_idx += 1
    
    # Close the camera
    zed.close()
    
    return

import random

mylist = random.sample(mylist, 100)

depth_arrays = []

for (typ, episode_meta_path, task_instruction, h5_path, Tbase2cam, svo_path) in tqdm(mylist):
    print(episode_meta_path)

    g = load_svo(droid_base_path + "/" + svo_path, 0)

    intrinsics, depth_scale, fps, height, width = next(g)

    assert fps == 60 or fps == 59 or fps == 58 # maybe 59
    assert height == 720
    assert width == 1280

    for frame_idx, image_array, depth_array in g:
        break
    
    depth_arrays.append(depth_array)
    
    # depth_arrays.append((depth_array * 4).astype(np.uint16)) # depth_scale = 4000
    
depth_arrays = np.array(depth_arrays)

depth_arrays = depth_arrays[(depth_arrays > 0) & (depth_arrays < 100000)]

print(f"mean: {np.mean(depth_arrays)} std: {np.std(depth_arrays)}")

# 10 samples
# mean: 1160.43701171875 std: 1332.7889404296875
# mean: 1188.3355712890625 std: 1006.9229736328125

# 100 samples
# mean: 1398.9407958984375 std: 1614.354248046875
# mean: 1308.0906982421875 std: 1546.66943359375
