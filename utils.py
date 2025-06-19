import pytransform3d.rotations as pr
import numpy as np

def xyz_from_uvd(uvd, intrinsic_matrix, depth_scale):
    u, v, d = uvd
    # https://www.open3d.org/docs/0.6.0/python_api/open3d.geometry.create_point_cloud_from_rgbd_image.html
    cx = intrinsic_matrix[0, 2]
    cy = intrinsic_matrix[1, 2]
    fx = intrinsic_matrix[0, 0]
    fy = intrinsic_matrix[1, 1]
    z = d / depth_scale
    x = (u - cx) * z / fx
    y = (v - cy) * z / fy
    return (x, y, z)

def uvd_from_xyz(xyz, intrinsic_matrix, depth_scale):
    x, y, z = xyz
    cx = intrinsic_matrix[0, 2]
    cy = intrinsic_matrix[1, 2]
    fx = intrinsic_matrix[0, 0]
    fy = intrinsic_matrix[1, 1]
    u = (x * fx / z) + cx
    v = (y * fy / z) + cy
    d = z * depth_scale
    return (u, v, d)

def T_from_xyzrpy(xyzrpy):
    T = np.eye(4)
    T[:3,3] = xyzrpy[:3]
    T[:3,:3] = pr.matrix_from_euler(xyzrpy[3:], 0, 1, 2, True)
    return T

def xyzrpy_from_T(T):
    xyzrpy = np.zeros(6)
    xyzrpy[:3] = T[:3,3]
    xyzrpy[3:] = pr.euler_from_matrix(T[:3,:3], 0, 1, 2, True)
    return xyzrpy

def normalized_uvd_axis_angle_from_T(T, intrinsics, depth_scale, height, width):
    xyz = T[:3, 3]
    
    axis_angle = pr.compact_axis_angle_from_matrix(T[:3,:3])
    
    uvd = uvd_from_xyz(xyz, intrinsics, depth_scale)

    return uvd[0] / width, uvd[1] / height, uvd[2] / depth_scale / 2.8, *axis_angle

def T_from_normalized_uvd_axis_angle(pose, intrinsics, depth_scale, height, width):
    xyz = xyz_from_uvd((pose[0] * width, pose[1] * height, pose[2] * depth_scale * 2.8), intrinsics, depth_scale)
    
    R = pr.matrix_from_compact_axis_angle(pose[3:])
    
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = xyz
    
    return T