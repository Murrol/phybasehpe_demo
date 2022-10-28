import numpy as np
import pyrender
import smplx
import torch
import trimesh
import matplotlib.pyplot as plt
import torch.nn.functional as F
import os
import cv2
os.environ["PYOPENGL_PLATFORM"] = "egl"
def quat_pos(x):
    """
    make all the real part of the quaternion positive
    """
    q = x
    z = (q[..., 3:] < 0).float()
    q = (1 - 2 * z) * q
    return q

def quat_abs(x):
    """
    quaternion norm (unit quaternion represents a 3D rotation, which has norm of 1)
    """
    x = x.norm(p=2, dim=-1)
    return x

def quat_unit(x):
    """
    normalized quaternion with norm of 1
    """
    norm = quat_abs(x).unsqueeze(-1)
    return x / (norm.clamp(min=1e-9))

def quat_normalize(q):
    """
    Construct 3D rotation from quaternion (the quaternion needs not to be normalized).
    """
    q = quat_unit(quat_pos(q))  # normalized to positive and unit quaternion
    return q


def my_quat_rotate(q, v):
    shape = q.shape
    q_w = q[:, -1]
    q_vec = q[:, :3]
    a = v * (2.0 * q_w ** 2 - 1.0).unsqueeze(-1)
    b = torch.cross(q_vec, v, dim=-1) * q_w.unsqueeze(-1) * 2.0
    c = q_vec * \
        torch.bmm(q_vec.view(shape[0], 1, 3), v.view(
            shape[0], 3, 1)).squeeze(-1) * 2.0
    return a + b + c

def quat_to_tan_norm_yup(q):
    # type: (Tensor) -> Tensor
    # represents a rotation using the tangent and normal vectors
    ref_tan = torch.zeros_like(q[..., 0:3])
    ref_tan[..., 2] = 1
    tan = my_quat_rotate(q, ref_tan)
    
    ref_norm = torch.zeros_like(q[..., 0:3])
    ref_norm[..., 1] = 1
    norm = my_quat_rotate(q, ref_norm)
    
    norm_tan = torch.cat([tan, norm], dim=len(tan.shape) - 1)
    return norm_tan

def quat_mul_vec(q, x):
    """
    Performs multiplication of an array of 3D vectors by an array of quaternions (rotation).

    :param q: tensor of quaternions of shape (..., Nb of joints, 4)
    :param x: tensor of vectors of shape (..., Nb of joints, 3)
    :return: the resulting array of rotated vectors
    """
    t = 2.0 * torch.cross(q[..., :3], x)
    res = x + q[..., 3][..., None] * t + torch.cross(q[..., :3], t)

    return res

def quat_between(x, y):
    """
    Quaternion rotations between two 3D-vector arrays

    :param x: tensor of 3D vectors
    :param y: tensor of 3D vetcors
    :return: tensor of quaternions
    """
    # print(x.shape, y.shape)
    x = x.type(y.dtype)
    x, y = torch.broadcast_tensors(x, y)
    res = torch.cat([
        torch.cross(x, y),
        torch.sqrt(torch.sum(x * x, dim=-1) * torch.sum(y * y, dim=-1)).unsqueeze(-1) +
        torch.sum(x * y, dim=-1).unsqueeze(-1)
        ], dim=-1)
    return res

def normalize_angle(x):
    return torch.atan2(torch.sin(x), torch.cos(x))

def quat_to_angle_axis(q):
    # type: (Tensor) -> Tuple[Tensor, Tensor]
    # computes axis-angle representation from quaternion q
    # q must be normalized
    min_theta = 1e-5
    qx, qy, qz, qw = 0, 1, 2, 3

    sin_theta = torch.sqrt(1 - q[..., qw] * q[..., qw])
    angle = 2 * torch.acos(q[..., qw])
    angle = normalize_angle(angle)
    sin_theta_expand = sin_theta.unsqueeze(-1)
    axis = q[..., qx:qw] / sin_theta_expand

    mask = sin_theta > min_theta
    default_axis = torch.zeros_like(axis)
    default_axis[..., -1] = 1

    angle = torch.where(mask, angle, torch.zeros_like(angle))
    mask_expand = mask.unsqueeze(-1)
    axis = torch.where(mask_expand, axis, default_axis)
    return angle, axis

def angle_axis_to_exp_map(angle, axis):
    # type: (Tensor, Tensor) -> Tensor
    # compute exponential map from axis-angle
    angle_expand = angle.unsqueeze(-1)
    exp_map = angle_expand * axis
    return exp_map

def quat_to_exp_map(q):
    # type: (Tensor) -> Tensor
    # compute exponential map from quaternion
    # q must be normalized
    angle, axis = quat_to_angle_axis(q)
    exp_map = angle_axis_to_exp_map(angle, axis)
    return exp_map

'''
body_model = 'smpl'
body_model_path = '/home/datassd/yuxuan/smpl_model/models'
obs = np.load('obs/2022-09-18_15-47-47.npy')
obs = obs[None]
obs = torch.from_numpy(obs)

# [(root_h, root_rot, root_vel, root_ang_vel), dof_pos, dof_vel, key_body_pos 3*4]
# localRootObs: False
# print(obs.shape) #13 + 63*2 + 63 + 12 = 214
root_rot_obs = obs[..., 3: 9]
joint_rot_obs = obs[..., 13: 13+126]

root_rot_obs = root_rot_obs.unsqueeze(-2)
root_tan = root_rot_obs[..., 0:3]
ref_tan = torch.zeros_like(root_tan)
ref_tan[..., 2] = 1.

root_quat = quat_between(ref_tan, root_tan)
root_rot = quat_to_exp_map(root_quat)
# print(joint_rot_obs.shape[:-1])
joint_tan = joint_rot_obs.reshape(list(joint_rot_obs.shape[:-1]) + [21, 2, 3])[..., 0, :]
# print(joint_tan.shape)
ref_tan = torch.zeros_like(joint_tan)
ref_tan[..., 2] = 1.
joint_quat = quat_between(ref_tan, joint_tan)
joint_rot = torch.zeros(list(joint_tan.shape[:-2]) + [23, 3])
joint_rot[..., :-2, :] = quat_to_exp_map(joint_quat)
# print(joint_rot.shape)

###############exp
qt = quat_normalize(torch.Tensor([[0.3, -0.5, 0.7, 0.5]]))
tan_norm = quat_to_tan_norm_yup(qt)
ref_tan = torch.Tensor([[0, 0, 1]])
ref_norm = torch.Tensor([[0, 1, 0]])
qttan = quat_between(ref_tan, tan_norm[..., :3])
qtnorm = quat_between(ref_norm, tan_norm[..., 3:])
# print(qt, qttan, qtnorm)

vec1 = torch.FloatTensor([[2,1,1]])
vec2 = torch.FloatTensor([[1,-1,2]])
print(quat_between(vec1, vec2))
print(quat_normalize(quat_between(vec1, vec2)))

print(my_quat_rotate(quat_normalize(quat_between(vec2, vec1)), vec2))

print(qt, quat_normalize(quat_between(quat_mul_vec(qt, ref_tan), ref_tan)))

body_model = smplx.create(model_path=body_model_path, model_type=body_model)
faces = body_model.faces
# print(root_quat)
# print(joint_quat)
vertices = body_model(global_orient=root_rot, body_pose=joint_rot).vertices[0].detach().numpy()
# vertices = body_model().vertices[0].detach().numpy()

original_mesh = trimesh.Trimesh(vertices=vertices, faces=faces)
original_mesh.export('obsvistest.ply')
mesh = pyrender.Mesh.from_trimesh(original_mesh)
scene = pyrender.Scene(bg_color=[0, 0, 0, 0], ambient_light=(0.3, 0.3, 0.3))
# scene = pyrender.Scene()
scene.add(mesh, 'mesh')

# add camera pose
camera_pose = np.array([[1, 0, 0, 0],
                        [0, 1, 0, 0],
                        [0, 0, 1, 3],
                        [0, 0, 0, 1]])
# use this to make it to center
camera = pyrender.camera.PerspectiveCamera(yfov=1)
scene.add(camera, pose=camera_pose)

# Get the lights from the viewer
light = pyrender.SpotLight(color=np.ones(3), intensity=3.0, innerConeAngle=np.pi/3.0, outerConeAngle=np.pi/3.0)
scene.add(light, pose=camera_pose)

# offscreen render
r = pyrender.OffscreenRenderer(viewport_width=512, viewport_height=512)
color, depth = r.render(scene, flags=pyrender.RenderFlags.RGBA)
plt.figure(figsize=(8, 8))
plt.imshow(color[:, :, 0:3])
plt.show()
'''

DOF_BODY_IDS = [3, 6, 9, 13, 16, 18, 20, 12, 15, 14, 17, 19, 21, 2, 5, 8, 11, 1, 4, 7, 10]
body_model = 'smpl'
body_model_path = '/home/datassd/yuxuan/smpl_model/models'
obs = np.load('obs/new/2022-09-18_23-47-48.npy')
obs = obs[None]
obs = torch.from_numpy(obs)
print(obs.shape)


rot = torch.zeros(list(obs.shape[:-2]) + [24, 3])
rot[..., DOF_BODY_IDS, :] = obs[..., 4:].view(list(obs.shape[:-2]) + [21, 3])
rot[..., 0, :] = quat_to_exp_map(obs[:, 0, :4])


body_model = smplx.create(model_path=body_model_path, model_type=body_model)
faces = body_model.faces
print(rot.shape)

vertices = body_model(global_orient=rot[..., :1, :], body_pose=rot[..., 1:, :]).vertices[0].detach().numpy()
# vertices = body_model().vertices[0].detach().numpy()

original_mesh = trimesh.Trimesh(vertices=vertices, faces=faces)
original_mesh.export('obsvistest.ply')
mesh = pyrender.Mesh.from_trimesh(original_mesh)
scene = pyrender.Scene(bg_color=[0, 0, 0, 0], ambient_light=(0.3, 0.3, 0.3))
# scene = pyrender.Scene()
scene.add(mesh, 'mesh')

# add camera pose
camera_pose = np.array([[1, 0, 0, 0],
                        [0, 1, 0, 0],
                        [0, 0, 1, 3],
                        [0, 0, 0, 1]])
# use this to make it to center
camera = pyrender.camera.PerspectiveCamera(yfov=1)
scene.add(camera, pose=camera_pose)

# Get the lights from the viewer
light = pyrender.SpotLight(color=np.ones(3), intensity=3.0, innerConeAngle=np.pi/3.0, outerConeAngle=np.pi/3.0)
scene.add(light, pose=camera_pose)

# offscreen render
r = pyrender.OffscreenRenderer(viewport_width=512, viewport_height=512)
color, depth = r.render(scene, flags=pyrender.RenderFlags.RGBA)
plt.figure(figsize=(8, 8))
plt.imshow(color[:, :, 0:3])
plt.show()
cv2.imwrite('obsvistest.png', color[:, :, 0:3])