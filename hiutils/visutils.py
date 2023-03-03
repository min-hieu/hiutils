import mitsuba as mi
mi.set_variant('llvm_ad_rgb')

from mitsuba import ScalarTransform4f as T
# from mitsuba import ScalarTransform4f, Transform4f
import torch
import numpy as np
from typing import Union
import drjit as dr
import trimesh
import pathlib
import matplotlib.pyplot as plt
from pathlib import Path

def as_numpy(ndarray):
    if isinstance(ndarray, torch.Tensor):
        return ndarray.detach().cpu().numpy()
    elif isinstance(ndarray, np.ndarray):
        return ndarray
    elif isinstance(ndarray, list) or isinstance(ndarray, tuple):
        return np.array(ndarray)
    else:
        raise ValueError("Input should be either torch.Tensor or np.ndarray")


def np2th(ndarray):
    if isinstance(ndarray, torch.Tensor):
        return ndarray.detach().cpu()
    elif isinstance(ndarray, np.ndarray):
        return torch.tensor(ndarray).float()
    else:
        raise ValueError("Input should be either torch.Tensor or np.ndarray")

def get_color(c):
    if c == "chair":
        col = [96, 153, 102]
    if c == "airplane":
        col = [176, 139, 187]
    return np.array(col) / 255

def to_mitsuba_coord(pc):
    rot_mat = np.array([[1, 0, 0], [0,0,-1], [0,1,0]])
    return (pc * 2) @ rot_mat.T

def normalize_points(p, method: str="sphere"):
    if method == "sphere":
        return _to_unit_sphere(p).numpy()
    elif method == "cube":
        return _to_unit_cube(p).numpy()
    else:
        raise AssertionError

def _to_unit_sphere(pc: Union[np.ndarray, torch.Tensor]):
    """
    pc: [B,N,3] or [N,3]
    """
    pc = np2th(pc)
    shapes = pc.shape
    N = shapes[-2]
    pc = pc.reshape(-1, N, 3)
    m = pc.mean(1, keepdim=True)
    pc = pc - m
    s = torch.max(torch.sqrt(torch.sum(pc**2, -1, keepdim=True)), 1, keepdim=True)[0]
    pc = pc / s
    pc = pc.reshape(shapes)
    return pc

def _to_unit_cube(pc: Union[np.ndarray, torch.Tensor]):
    """
    pc: [B,N,3] or [N,3]
    """
    pc = np2th(pc)
    shapes = pc.shape
    N = shapes[-2]
    pc = pc.reshape(-1,N,3)
    max_vals = pc.max(1, keepdim=True)[0] #[B,1,3]
    min_vals = pc.min(1,keepdim=True)[0] #[B,1,3]
    max_range = (max_vals - min_vals).max(-1)[0] / 2 #[B,1]
    center = (max_vals + min_vals) / 2 #[B,1,3]

    pc = pc - center
    pc = pc / max_range[..., None]
    pc = pc.reshape(shapes)
    return pc

def get_unit_cube_transform(pc):
    pc = np2th(pc)
    shapes = pc.shape
    N = shapes[-2]
    pc = pc.reshape(-1,N,3)
    max_vals = pc.max(1, keepdim=True)[0] #[B,1,3]
    min_vals = pc.min(1,keepdim=True)[0] #[B,1,3]
    max_range = (max_vals - min_vals).max(-1)[0] / 2 #[B,1]
    center = (max_vals + min_vals) / 2 #[B,1,3]
    return center, max_range

def clean_cache():
    gc.collect()
    dr.eval()
    dr.sync_thread()
    dr.flush_malloc_cache()
    dr.malloc_clear_statistics()

def get_sensor(r, phi, theta, res):
    # Apply two rotations to convert from spherical coordinates to world 3D coordinates.
    origin = T.rotate([0, 0, 1], phi).rotate([0, 1, 0], theta) @ mi.ScalarPoint3f([0, 0, r])

    if type(res) == int: res = (res, res)

    return mi.load_dict({
        'type': 'perspective',
        'fov': 39.3077,
        'to_world': T.look_at(
            origin=origin,
            target=[0, 0, 0],
            up=[0, 0, 1]
        ),
        'sampler': {
            'type': 'independent',
            'sample_count': 16
        },
        'film': {
            'type': 'hdrfilm',
            'width': res[0],
            'height': res[1],
            'rfilter': {
                'type': 'box',
            },
            'pixel_format': 'rgb',
        },
    })

def get_scene_dict(scene_type="default", floor=True):
    default_scene = {
        'type': 'scene',
        'integrator': {'type': 'path'},
        'light': {'type': 'constant', 'radiance': 1.0},
    }

    if scene_type == "default":
        out_scene = default_scene
    else:
        out_scene = default_scene

    if floor:
        out_scene["floor"] = {
            'type': 'rectangle',
            'to_world': T.translate([0, 0, -2]).scale(100),
            'bsdf': {
                'type': 'diffuse',
                'reflectance': {'type': 'rgb', 'value': 1.},
            },
        }

    return out_scene

def render_pointcloud(pc, color=0.6, normalize='cube',
                      transform=lambda x: x,
                      camR=10, camPhi=45, camTheta=60, camRes=(512,512),
                      **scene_kwargs):
    pc = as_numpy(pc)
    scene_dict = get_scene_dict(**scene_kwargs)

    if normalize: pc = normalize_points(pc, normalize)

    for i, pos in enumerate(to_mitsuba_coord(transform(pc))):
        scene_dict[f'point_{i}'] = {
            'type': 'sphere',
            'to_world': mi.Transform4f.translate(pos).scale(0.05),
            'bsdf': {
                'type': 'diffuse',
                'reflectance': {'type': 'rgb', 'value': color},
            }
        }

    scene = mi.load_dict(scene_dict)
    sensor = get_sensor(camR, camPhi, camTheta, camRes)

    return mi.render(scene, spp=1000, sensor=sensor)

def as_mesh(scene_or_mesh):
    if isinstance(scene_or_mesh, trimesh.Scene):
        if len(scene_or_mesh.geometry) == 0:
            mesh = None  # empty scene
        else:
            # we lose texture information here
            mesh = trimesh.util.concatenate(
                tuple(trimesh.Trimesh(vertices=g.vertices, faces=g.faces)
                    for g in scene_or_mesh.geometry.values()))
    else:
        mesh = scene_or_mesh
        assert(isinstance(mesh, trimesh.Trimesh))
    return mesh

def vf2mimesh(v, f, color):
    bsdf = mi.load_dict({'type': 'diffuse',
                         'reflectance': {'type': 'rgb', 'value': color}})
    props = mi.Properties()
    props["mesh_bsdf"] = bsdf
    mimesh = mi.Mesh(
        "mymesh",
        vertex_count=v.shape[0],
        face_count=f.shape[0],
        has_vertex_normals=False,
        has_vertex_texcoords=False,
        props=props
    )
    mesh_params = mi.traverse(mimesh)
    mesh_params["vertex_positions"] = np.ravel(v)
    mesh_params["faces"] = np.ravel(f)
    mesh_params.update()

    return mimesh

def read_obj(name: str):
    verts = []
    faces = []
    with open(name, "r") as f:
        lines = [line.rstrip() for line in f]

        for line in lines:
            if line.startswith("v "):
                verts.append(np.float32(line.split()[1:4]))
            elif line.startswith("f "):
                faces.append(np.int32([item.split("/")[0] for item in line.split()[1:4]]))

        v = np.vstack(verts)
        f = np.vstack(faces) - 1
        return v, f

def load_mesh(mesh_path):
    mesh = as_mesh(trimesh.exchange.load.load(mesh_path))
    mesh = trimesh.exchange.load.load(mesh_path)
    # v, f = read_obj(mesh_path)
    # trimesh.Trimesh(v, f)
    trimesh.repair.fix_normals(mesh)
    return np.array(mesh.vertices), np.array(mesh.faces)

def render_gaussian(gaussians, color=0.6, cmap=None,
                    transform=lambda x: x,
                    camR=10, camPhi=45, camTheta=60, camRes=(512,512),
                    **scene_kwargs):

    scene_dict = get_scene_dict(**scene_kwargs)
    cmap = plt.get_cmap("plasma")

    gaussians = as_numpy(gaussians)
    N = gaussians.shape[0]
    v_list = []
    f_list = []
    for i, g in enumerate(gaussians):
        mu, eivec, eival = g[:3], g[3:12], g[13:]

        R = eivec.reshape(3, 3).T
        S = np.sqrt(np.clip(eival, 1e-4, 99999)) * 0.5

        v, f = load_mesh(Path(__file__).parent / "objs/sphere.obj")
        v = mu + ((v * S) @ R.T)
        v = to_mitsuba_coord(v) * 1.2
        mesh = trimesh.Trimesh(vertices=v, faces=f)
        mesh.fix_normals()
        v, f = np.array(mesh.vertices), np.array(mesh.faces)
        scene_dict[f'point_{i}'] = vf2mimesh(v, f, 0.45)

    scene = mi.load_dict(scene_dict)
    sensor = get_sensor(camR, camPhi, camTheta, camRes)
    return mi.render(scene, spp=1000, sensor=sensor)

def render_mesh(mesh, color=0.6, normalize='cube',
                transform=lambda x: x,
                camR=10, camPhi=45, camTheta=60, camRes=(512,512),
                **scene_kwargs):
    scene_dict = get_scene_dict(**scene_kwargs)

    if type(mesh) == pathlib.PosixPath or type(mesh) == str:
        v, f = load_mesh(mesh)
    elif type(mesh) == dict:
        v, f = mesh['vert'], mesh['face']
    elif type(mesh) == trimesh.Trimesh:
        v, f = np.array(mesh.vertices), np.array(mesh.faces)
    elif type(mesh) == tuple:
        v, f = mesh
    else:
        raise Exception('Invalid Mesh Type')

    if normalize: v = normalize_points(v, normalize)
    v = to_mitsuba_coord(transform(v))

    assert(type(v) == np.ndarray and type(f) == np.ndarray)
    scene_dict['mesh'] = vf2mimesh(v, f, color)
    scene = mi.load_dict(scene_dict)
    sensor = get_sensor(camR, camPhi, camTheta, camRes)

    return mi.render(scene, spp=1000, sensor=sensor)

def write_img(img, path):
    '''
    input:
        img - mitsuba tensorXf or List[tensorXf]
        path - Path or str or List[Path or str]. len(path) == len(img) if they
            are both List
    output: None. write img to path
    '''
    if type(img) == list:
        assert(len(img) == len(path))
        for i, img in enumerate(img):
            mi.util.write_bitmap(f"{path[i]}", img)
    else:
        mi.util.write_bitmap(f"{path}", img)
