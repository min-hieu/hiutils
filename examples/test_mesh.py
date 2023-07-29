import mitsuba as mi
mi.set_variant('scalar_rgb')

import itertools
from hiutils.miutils import load_mesh, render_mesh, write_img
from pathlib import Path
import numpy as np

mesh = Path(f"./bunny.obj")

rot = lambda t: np.array([
    [np.cos(t), 0., np.sin(t)],
    [0.,  1.,  0.],
    [-np.sin(t),  0., np.cos(t)]
])
col = np.array([176, 139, 187]) / 255

def tran(x):
    return x @ rot(np.pi).T

print("[*] Rendering...")
write_img(render_mesh(mesh, transform=tran, color=col), "./out_mesh.png")
print("[!] Render Completed!")
