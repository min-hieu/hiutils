import mitsuba as mi
mi.set_variant('llvm_ad_rgb')

from mitsuba import ScalarTransform4f as T
import torch
import numpy as np
from typing import Union
import drjit as dr
import trimesh
import pathlib

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
    pc = np2th(pc).numpy()
    scene_dict = get_scene_dict(**scene_kwargs)
    if normalize is not None: pc = normalize_points(pc, normalize)
    mit_pc = to_mitsuba_coord(pc) - np.array([-0.25,-0.25,0])

    for i, pos in enumerate(mit_pc):
        scene_dict[f'point_{i}'] = {
            'type': 'sphere',
            'to_world': T.translate(pos).scale(0.05),
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

def dict2mesh(mesh_dict, color, normalize):
    v = np.array(mesh_dict['vert'])
    if normalize: v = to_unit_cube(mesh.vertices)
    v = to_mitsuba_coord(v)
    f = np.array(mesh_dict['face'])
    mimesh = mi.Mesh(
        "mymesh",
        vertex_count=v.shape[0],
        face_count=f.shape[0],
        has_vertex_normals=False,
        has_vertex_texcoords=False,
    )
    mesh_params = mi.traverse(mimesh)
    mesh_params["vertex_positions"] = np.ravel(v)
    mesh_params["faces"] = np.ravel(f)
    mesh_params["bsdf.reflectance.value"] = color
    mesh_params.update()

    return mimesh

def load_mesh(mesh_path, color, normalize):
    mesh = as_mesh(trimesh.exchange.load.load(mesh_path))
    trimesh.repair.fix_normals(mesh)
    return mesh.vertices, mesh.faces

def render_mesh(mesh, color=0.6, normalize='cube',
                transform=lambda x: x,
                camR=10, camPhi=45, camTheta=60, camRes=(512,512),
                **scene_kwargs):
    scene_dict = get_scene_dict(**scene_kwargs)

    if type(mesh) == pathlib.PosixPath or type(mesh) == str:
        v, f = load_mesh(mesh, color, normalize)
    elif type(mesh) == dict:
        v, f = mesh['vert'], mesh['face']
    elif type(mesh) == trimesh.Trimesh:
        v, f = mesh.vertices, mesh.faces
    else:
        raise Exception('Invalid Mesh Type')

    if normalize is not None:
        v = normalize_points(transform(v), normalize)
    else:
        v = transform(v)

    mesh = {'vert': v,'face': f}

    scene_dict['mesh'] = dict2mesh(mesh, color, normalize)
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
