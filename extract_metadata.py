import glob
import json
import os
import tqdm

DATA_PATH = "/nfs/kun2/datasets/r2d2/r2d2-data-full"


metadata_paths = glob.glob(os.path.join(DATA_PATH, "**/metadata_*.json"), recursive=True)
print(len(metadata_paths))

r2d2_metadata = {}
for meta_filepath in tqdm.tqdm(metadata_paths):
    with open(meta_filepath, "r") as F:
        metadata = json.load(F)
    r2d2_metadata[metadata["hdf5_path"]] = {
        "scene_id": metadata["scene_id"],
        "lab": metadata["lab"],
        "building": metadata["building"],
        "wrist_cam_extrinsics": metadata["wrist_cam_extrinsics"],
        "ext1_cam_extrinsics": metadata["ext1_cam_extrinsics"],
        "ext2_cam_extrinsics": metadata["ext2_cam_extrinsics"],
    }

with open("r2d2_metadata.json", "w") as F:
    json.dump(r2d2_metadata, F, indent=2)
