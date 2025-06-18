conda create -n torch python=3.12

bash ZED_SDK_Ubuntu22_cuda12.1_v4.2.5.zstd.run
pip install open3d numpy opencv-python pytransform3d

pip install -e ./Helper3D
pip install "h5py<3.11"
pip install "opencv-python<4.10"