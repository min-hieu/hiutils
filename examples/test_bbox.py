import itertools
from hiutils.visutils import load_mesh, normalize_points, render_mesh_bbox, write_img
import numpy as np
from pathlib import Path

def rotate_pc(pc):
    rot_mat = np.array([[0, 0, 1], [0,1,0], [-1,0,0]])
    return pc @ rot_mat.T

def get_bbox(pc, label, l):
    mask_pc = pc[label == l]
    minv, maxv = mask_pc.min(0), mask_pc.max(0)
    return minv-0.1, maxv+0.1

shapeid = 'ff529b9ad2d5c6abf7e98086e1ca9511'
pc_file = Path(f"/home/blackhole/shared/datasets/ShapeNet-Seg/03001627/{shapeid}.txt")
mesh = Path(f"/home/blackhole/juil/docker_home/datasets/ShapenetCore.v2-WT/chairs/{shapeid}.obj")
pc_norm_label = np.loadtxt(pc_file)

pc, label = pc_norm_label[:, :3], pc_norm_label[:, -1]
pc = normalize_points(rotate_pc(pc), 'cube')
bbox = [get_bbox(pc, label, l) for l in np.unique(label)][0]

write_img(render_mesh_bbox(mesh, bbox), "./mesh_w_bbox.png")
