```bash
conda create -n torch python=3.12

bash ZED_SDK_Ubuntu22_cuda12.1_v4.2.5.zstd.run
pip install open3d numpy opencv-python pytransform3d

pip install -e ./Helper3D
pip install "h5py<3.11"
pip install "opencv-python<4.9"
pip install pytorch3d
pip install -U "numpy<2"

# add pyproject.toml following https://github.com/facebookresearch/pytorch3d/issues/1419
python lerobot/scripts/visualize_dataset.py --repo-id lookas/droid3d_test --episode-index 0
```