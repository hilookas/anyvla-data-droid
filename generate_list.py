import json
import glob
import pytransform3d.transformations as pt
from utils import xyz_from_uvd, uvd_from_xyz, T_from_xyzrpy, xyzrpy_from_T
from tqdm import tqdm
import pickle

droid3d_anno_path = "droid"

with open(droid3d_anno_path + "/cam2base_extrinsics.json") as f:
    cam2base_extrinsics = json.load(f)

with open(droid3d_anno_path + "/cam2base_extrinsic_superset.json") as f:
    cam2base_extrinsic_superset = json.load(f)

with open(droid3d_anno_path + "/droid_language_annotations.json") as f:
    droid_language_annotations = json.load(f)

print(len(cam2base_extrinsic_superset))

droid_base_path = "/mnt/20T/droid_raw_1.0.1"

episode_meta_paths = glob.glob("*/success/*/*/metadata_*.json", root_dir=droid_base_path)
print(len(episode_meta_paths))

def wo_prefix(p):
    return "/".join(p.split("/")[3:])

mylist = []

for episode_meta_path in tqdm(episode_meta_paths):
    try:
        with open(droid_base_path + "/" + episode_meta_path, "r") as f:
            metadata = json.load(f)
            
        uuid = metadata["uuid"]
        
        if uuid in droid_language_annotations:
            task_instruction = metadata["current_task"] + " | " + droid_language_annotations[uuid]['language_instruction1'] + " | " + droid_language_annotations[uuid]['language_instruction2'] + " | " + droid_language_annotations[uuid]['language_instruction3']
        else:
            task_instruction = metadata["current_task"]
            print(f"uuid {uuid} not in droid_language_annotations")
        
        if uuid in cam2base_extrinsic_superset:
            prefix_path = "/".join(episode_meta_path.split("/")[:-1])
            
            h5_path = prefix_path + "/" + wo_prefix(metadata["hdf5_path"])
            
            Tbase2cam = pt.invert_transform(T_from_xyzrpy(cam2base_extrinsic_superset[uuid][metadata["ext1_cam_serial"]]))
            svo_path = prefix_path + "/" + wo_prefix(metadata["ext1_svo_path"])

            Tbase2cam2 = pt.invert_transform(T_from_xyzrpy(cam2base_extrinsic_superset[uuid][metadata["ext2_cam_serial"]]))
            svo_path2 = prefix_path + "/" + wo_prefix(metadata["ext2_svo_path"])
            
            mylist.append((
                "superset_ext1",
                episode_meta_path,
                task_instruction,
                h5_path,
                Tbase2cam,
                svo_path,
            ))
            
            mylist.append((
                "superset_ext2",
                episode_meta_path,
                task_instruction,
                h5_path,
                Tbase2cam2,
                svo_path2,
            ))
        elif uuid in cam2base_extrinsics:
            prefix_path = "/".join(episode_meta_path.split("/")[:-1])
                
            h5_path = prefix_path + "/" + wo_prefix(metadata["hdf5_path"])
            
            if metadata["ext1_cam_serial"] in cam2base_extrinsics[uuid]:
                Tbase2cam = pt.invert_transform(T_from_xyzrpy(cam2base_extrinsics[uuid][metadata["ext1_cam_serial"]]))
                svo_path = prefix_path + "/" + wo_prefix(metadata["ext1_svo_path"])
                
                mylist.append((
                    "ext1",
                    episode_meta_path,
                    task_instruction,
                    h5_path,
                    Tbase2cam,
                    svo_path,
                ))
                
            if metadata["ext2_cam_serial"] in cam2base_extrinsics[uuid]:
                Tbase2cam2 = pt.invert_transform(T_from_xyzrpy(cam2base_extrinsics[uuid][metadata["ext2_cam_serial"]]))
                svo_path2 = prefix_path + "/" + wo_prefix(metadata["ext2_svo_path"])
                
                mylist.append((
                    "ext2",
                    episode_meta_path,
                    task_instruction,
                    h5_path,
                    Tbase2cam2,
                    svo_path2,
                ))
        else:
            print(f"uuid {uuid} not in cam2base_extrinsics or cam2base_extrinsic_superset")
            continue
    except Exception as e:
        import traceback; print(traceback.format_exc())
        print(e)
        import ipdb; ipdb.set_trace()

with open("list.pkl", "wb") as f:
    pickle.dump(mylist, f)