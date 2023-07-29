import mitsuba as mi
mi.set_variant('scalar_rgb')

import itertools
from hiutils.miutils import load_mesh, normalize_points, render_mesh_bbox, write_img
import numpy as np
from pathlib import Path

def rotate_pc(pc):
    rot_mat = np.array([[0, 0, 1], [0,1,0], [-1,0,0]])
    return pc @ rot_mat.T

def get_bbox(pc, label, l):
    mask_pc = pc[label == l]
    minv, maxv = mask_pc.min(0), mask_pc.max(0)
    return minv-0.1, maxv+0.1

pc_file = Path(f"path/to/pc")
mesh = Path(f"path/to/pc")
pc_norm_label = np.loadtxt(pc_file)

pc, label = pc_norm_label[:, :3], pc_norm_label[:, -1]
pc = normalize_points(rotate_pc(pc), 'cube')
bbox = [get_bbox(pc, label, l) for l in np.unique(label)][0]

write_img(render_mesh_bbox(mesh, bbox), "./mesh_w_bbox.png")
