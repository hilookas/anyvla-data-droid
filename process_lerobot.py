
import pickle
import h5py
from utils import T_from_xyzrpy, xyzrpy_from_T, normalized_uvd_axis_angle_from_T, T_from_normalized_uvd_axis_angle
import numpy as np
import cv2
import numpy as np
import pyzed.sl as sl
import shutil
from lerobot.common.datasets.lerobot_dataset import HF_LEROBOT_HOME
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
import pytransform3d.rotations as pr
from tqdm import tqdm

REPO_NAME = "lookas/droid3d"  # Name of the output dataset, also used for the Hugging Face Hub

droid_base_path = "/mnt/20T/droid_raw_1.0.1"

Teef2tip = np.array([
    [0, 0, 1, 0],
    [0, -1, 0, 0],
    [1, 0, 0, 0.150],
    [0, 0, 0, 1]
])

depth_mean = 1398.9407958984375
depth_std = 1614.354248046875

depth_mean_256 = 127.5
depth_std_256 = 74.18226046

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
    # print("ZED Model                 : {0}".format(cam_info.camera_model))
    # print("ZED Serial Number         : {0}".format(cam_info.serial_number))
    # print("ZED Camera Firmware       : {0}/{1}".format(cam_info.camera_configuration.firmware_version, cam_info.sensors_configuration.firmware_version))
    # print("ZED Camera Resolution     : {0}x{1}".format(width, height))
    # print("ZED Camera FPS            : {0}".format(fps))
    
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

def main():
    # Clean up any existing dataset in the output directory
    output_path = HF_LEROBOT_HOME / REPO_NAME
    if output_path.exists():
        shutil.rmtree(output_path)

    # Create LeRobot dataset, define features to store
    # OpenPi assumes that proprio is stored in `state` and actions in `action`
    # LeRobot assumes that dtype of image data is `image`
    dataset = LeRobotDataset.create(
        repo_id=REPO_NAME,
        robot_type="eef",
        fps=10,
        features={
            "image": {
                "dtype": "image",
                "shape": (224, 224, 3),
                "names": ["height", "width", "channel"],
            },
            "depth": {
                "dtype": "image",
                "shape": (224, 224, 3),
                "names": ["height", "width", "channel"],
            },
            "state": {
                "dtype": "float32",
                "shape": (14,),
                "names": ["state"],
            },
            "actions": {
                "dtype": "float32",
                "shape": (14,),
                "names": ["actions"],
            },
        },
        image_writer_threads=10,
        image_writer_processes=5,
    )

    with open("list.pkl", "rb") as f:
        mylist = pickle.load(f)

    # Iterate over all episodes
    for typ, episode_meta_path, task_instruction, h5_path, Tbase2cam, svo_path in tqdm(mylist):
        try:
            # Process eef pose
            cartesian_positions = None
            gripper_positions = None

            with h5py.File(droid_base_path + "/" + h5_path, "r") as f:
                cartesian_positions = np.array(f["observation"]["robot_state"]["cartesian_position"])
                gripper_positions = np.array(f["observation"]["robot_state"]["gripper_position"])

            Teef2cams = []
            for cartesian_position in cartesian_positions:
                Ttip2base = T_from_xyzrpy(cartesian_position)
                Ttip2cam = Tbase2cam @ Ttip2base
                Teef2cam = Ttip2cam @ Teef2tip
                Teef2cams.append(Teef2cam)

            g = load_svo(droid_base_path + "/" + svo_path, frame_step=6) # 6 * 10 = 60

            intrinsics, depth_scale, fps, height, width = next(g)

            assert fps > 50
            assert height == 720
            assert width == 1280

            for frame_idx, image_array, depth_array in g:
                # TODO depth 精度测试
                # TODO image 大小测试

                action_frame_idx = frame_idx + round(60 * 0.2) # following latency 200ms
                
                if action_frame_idx >= len(Teef2cams):
                    continue

                state = np.array([*normalized_uvd_axis_angle_from_T(Teef2cams[frame_idx], intrinsics, depth_scale, height, width), gripper_positions[frame_idx], *xyzrpy_from_T(Teef2cams[frame_idx]), gripper_positions[frame_idx]], dtype=np.float32)
                action = np.array([*normalized_uvd_axis_angle_from_T(Teef2cams[action_frame_idx], intrinsics, depth_scale, height, width), gripper_positions[action_frame_idx], *xyzrpy_from_T(Teef2cams[action_frame_idx]), gripper_positions[action_frame_idx]], dtype=np.float32)

                image_array_rgb = cv2.cvtColor(np.asarray(image_array), cv2.COLOR_BGRA2RGB)
                image_array_rgb = cv2.resize(image_array_rgb, (224, 224), interpolation=cv2.INTER_AREA)

                depth_array = (np.asarray(depth_array) - depth_mean) / depth_std
                depth_array = depth_array * depth_std_256 + depth_mean_256
                depth_array = cv2.resize(depth_array, (224, 224), interpolation=cv2.INTER_NEAREST)
                # Clip depth values to valid range before casting to uint8
                depth_array = np.clip(depth_array, 0, 255).astype(np.uint8)
                depth_array = depth_array[:, :, np.newaxis].repeat(3, axis=2)
                
                # cv2.imwrite("image.png", cv2.cvtColor(image_array_rgb, cv2.COLOR_RGB2BGR))
                # cv2.imwrite("depth.png", depth_array)
                
                # import ipdb; ipdb.set_trace()
                
                dataset.add_frame(
                    {
                        "image": image_array_rgb, # rgb image
                        "depth": depth_array,
                        "state": state,
                        "actions": action,
                        "task": task_instruction,  # see: https://github.com/Physical-Intelligence/openpi/issues/353
                    }
                )
            dataset.save_episode()
        except Exception as e:
            import traceback; print(traceback.format_exc())
            print(e)
            import json
            with open("error_log.jsonl", "a", encoding="utf-8") as f:
                f.write(json.dumps({
                    "episode_meta_path": episode_meta_path,
                    "traceback": traceback.format_exc(),
                    "error": str(e),
                }, ensure_ascii=False) + "\n")
            # import ipdb; ipdb.set_trace()

if __name__ == "__main__":
    main()