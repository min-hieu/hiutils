import torch
import numpy as np
import nvdiffrast.torch as dr

def projection(x=0.1, n=1.0, f=6.0, device='cpu'):
    return torch.tensor([[1/x,    0,            0,              0],
                         [  0,  1/x,            0,              0],
                         [  0,    0, -(f+n)/(f-n), -(2*f*n)/(f-n)],
                         [  0,    0,           -1,              0]],
                        dtype=torch.float32, device=device)

def translate(x, y, z, device='cpu'):
    return torch.tensor([[1, 0, 0, x],
                         [0, 1, 0, y],
                         [0, 0, 1, z],
                         [0, 0, 0, 1]],
                        dtype=torch.float32, device=device)

def rot_x(a, device='cpu'):
    s, c = np.sin(a), np.cos(a)
    return torch.tensor([[1,  0, 0, 0],
                         [0,  c, s, 0],
                         [0, -s, c, 0],
                         [0,  0, 0, 1]],
                        dtype=torch.float32, device=device)

def rot_y(a, device='cpu'):
    s, c = np.sin(a), np.cos(a)
    return torch.tensor([[ c, 0, s, 0],
                         [ 0, 1, 0, 0],
                         [-s, 0, c, 0],
                         [ 0, 0, 0, 1]],
                        dtype=torch.float32, device=device)

def look_at(eye, tgt, up, return3x4=False):
    up = up[None, ...] # B, 3
    f = normalize(eye - tgt)
    s = normalize(torch.linalg.cross(up, f))
    u = normalize(torch.linalg.cross(f, s))
    sufT = torch.stack((s,u,f), dim=-1).transpose(1,2) # B, 3, 3 <- transposed
    t = -eye.unsqueeze(1) @ sufT # dot of eye and s,u,f
    lookat3x4 = torch.cat((sufT, t), dim=-2).transpose(-1,-2)
    if return3x4:
        return lookat3x4
    else:
        bottom_row = torch.tensor([[[0,0,0,1]]]).repeat(lookat3x4.shape[0],1,1)
        return torch.cat((lookat3x4,bottom_row), dim=1)

def vec3(*v, device='cpu'):
    if len(v) == 1:
        return torch.tensor([v[0],v[0],v[0]], dtype=torch.float32, device=device)
    else:
        return torch.cat([torch.atleast_1d(torch.tensor(i)) for i in v], dim=-1).float().to(device)

def vec(*v, size=0, device='cpu'):
    return torch.tensor(v, dtype=torch.float32, device=device)

def length(v, dim=-1):
    return (torch.sum(v**2, dim, keepdim=True)+1e-8)**0.5

def normalize(v, dim=-1):
    return v / (torch.sum(v**2, dim, keepdim=True)+1e-8)**0.5

def dot(a,b):
    return torch.sum(a*b,-1,keepdim=True)

def inv(M):
    if torch.is_tensor(M):
        device = M.device
        return torch.tensor(np.linalg.inv(M.cpu().numpy())).float().to(device)
    else:
        return torch.tensor(np.linalg.inv(M))

def transform_pos(mtx, pos, device='cpu'):
    t_mtx = torch.from_numpy(mtx).to(device) if isinstance(mtx, np.ndarray) else mtx
    # (x,y,z) -> (x,y,z,1)
    posw = torch.cat([pos, torch.ones([pos.shape[0], 1]).to(device)], axis=1)
    return torch.matmul(posw, t_mtx.t())[None, ...]

def transform_norm(mtx, norm, device='cpu'):
    norm_shape = norm.shape
    norm = norm.reshape(-1,3)
    t_mtx = torch.from_numpy(mtx).to(device) if isinstance(mtx, np.ndarray) else mtx
    # (x,y,z) -> (x,y,z,0)
    normw = torch.cat([norm, torch.zeros([norm.shape[0], 1]).to(device)], axis=1)
    return torch.matmul(normw, t_mtx.t())[...,:3].reshape(*norm_shape)

def transform_pos_batch(M, p, device='cpu'):
    pw = torch.cat([p, torch.ones(*p.shape[:-1], 1).to(device)], axis=-1)
    return (M @ pw.transpose(-1,-2)).transpose(-1,-2)

def transform_norm_batch(M, n, device='cpu'):
    nw = torch.cat([n, torch.zeros(*n.shape[:-1], 1).to(device)], axis=-1)
    return (M @ nw.transpose(-1,-2)).transpose(-1,-2)[...,:3]

def save_img(col, out_path="out.png"):
    col = col[0].detach().cpu().numpy()[::-1]
    img = Image.fromarray(np.clip(np.rint(col*255.0), 0, 255).astype(np.uint8))
    img.save(out_path)

def save_img_batch(col, out_path_list=None):
    assert len(col) == len(out_path_list)
    for i in range(len(col)):
        tmp = col[i].detach().cpu().numpy()[::-1]
        img = Image.fromarray(np.clip(np.rint(tmp*255.0), 0, 255).astype(np.uint8))
        img.save(out_path_list[i])

def _to_unit_cube(p):
    centroid = np.mean(p, axis=0)
    p[:,0]-=centroid[0]
    p[:,1]-=centroid[1]
    p[:,2]-=centroid[2]
    d = max(np.sum(np.abs(p)**2,axis=-1)**(1./2))
    p /= d

    return p

def load_mesh(path, device='cpu', c=1.0, face_normals=False):
    mesh = trimesh.load(path)
    v    = _to_unit_cube(mesh.vertices)
    v    = torch.from_numpy(v.astype(np.float32)).to(device)
    f    = torch.from_numpy(mesh.faces.astype(np.int32)).to(device)
    vn   = torch.from_numpy(mesh.vertex_normals.astype(np.float32)).to(device)
    fn   = torch.from_numpy(mesh.face_normals.astype(np.float32)).to(device)
    c    = torch.from_numpy(mesh.visual.vertex_colors.astype(np.uint8)).to(device)
    c    = c[...,:3].float() / 255.0
    return v, f, vn, fn, c

def Fibonacci_grid_sample(num, radius):
    # https://www.jianshu.com/p/8ffa122d2c15
    points = [[0, 0, 0] for _ in range(num)]
    phi = 0.618
    for n in range(num):
        z = (2 * n - 1) / num - 1
        x = np.sqrt(np.abs(1 - z * z)) * np.cos(2 * np.pi * n * phi)
        y = np.sqrt(np.abs(1 - z * z)) * np.sin(2 * np.pi * n * phi)
        points[n][0] = x * radius
        points[n][1] = y * radius
        points[n][2] = z * radius

    points = np.array(points)
    return points

def sphere_angle_sample(num, radius):
    points = []
    for azim in np.linspace(-180, 180, num):
        elev = 60
        razim = np.pi * azim / 180
        relev = np.pi * elev / 180

        center = [0, 0, 0]
        xp = center[0] + np.cos(razim) * np.cos(relev) * radius
        yp = center[1] + np.sin(razim) * np.cos(relev) * radius
        zp = center[2] + np.sin(relev) * radius
        points.append([xp, yp, zp])
    points = np.array(points)
    return points

@torch.no_grad()
def render_flat(
    mvp,
    pos, pos_idx,
    col, col_idx=None,
    res=64, device='cpu',
):
    glctx   = dr.RasterizeCudaContext(device)
    col_idx = pos_idx if col_idx is None else col_idx

    pos_clip    = transform_pos(mvp, pos, device)
    rast_out, _ = dr.rasterize(glctx, pos_clip, pos_idx, resolution=[res, res], grad_db=False)
    color, _    = dr.interpolate(col[None, ...], rast_out, col_idx)
    color       = dr.antialias(color, rast_out, pos_clip, pos_idx)

    return color

@torch.no_grad()
def render_flat_batch(
    mvp,
    pos, pos_idx,
    col,
    res=64, device='cpu',
):
    glctx   = dr.RasterizeCudaContext(device)

    pos_clip    = transform_pos_batch(mvp, pos, device).contiguous()
    rast_out, _ = dr.rasterize(glctx, pos_clip, pos_idx, resolution=[res, res], grad_db=False)
    color, _    = dr.interpolate(col, rast_out, pos_idx)
    color       = dr.antialias(color, rast_out, pos_clip, pos_idx)

    return color

@torch.no_grad()
def render_diffuse(
    mvp, nmvp,
    pos, pos_idx, normals,
    col, col_idx=None,
    res=64, device='cpu',
    lights=[torch.tensor([[0,0,1]]).float(),(1.0)], bg_color=1.0,
    interpolate_normals=False,
):
    glctx   = dr.RasterizeCudaContext(device)
    col_idx = pos_idx if col_idx is None else col_idx

    Kd = vec3(0.4).to(device) # diffuse
    Ka = vec3(0.15).to(device) # ambient

    # Rasterize
    pos_clip    = transform_pos(mvp, pos, device)
    rast_out, _ = dr.rasterize(glctx, pos_clip, pos_idx, [res, res], grad_db=False)

    # Process Normals
    normals  = normalize(transform_norm(nmvp, normals, device)).contiguous()
    # norms, _ = dr.interpolate(normals, rast_out, pos_idx)
    # norms    = normalize(norms)
    norms = normals[rast_out[...,-1].int().clip(1) - 1]
    print(norms.shape, rast_out[...,-1].shape)
    norms[rast_out[...,-1].int() == 0] = 0

    # Phong shading
    color = Ka * torch.ones_like(norms)

    ldirs = normalize(transform_norm(nmvp, lights[0].to(device), device))
    for i, intensity in enumerate(lights[1]):
        ndotl = dot(ldirs[i], norms).clip(min=0)
        color += Kd * ndotl * intensity

    color = torch.where(rast_out[..., -1:] == 0, bg_color, color)
    color = dr.antialias(color, rast_out, pos_clip, pos_idx)

    return color

@torch.no_grad()
def render_diffuse_batch(
    mvp, nmvp,
    pos, pos_idx, normals,
    col, col_idx=None,
    res=64, device='cpu',
    lights=[torch.tensor([[0,0,1]]).float(),(1.0)], bg_color=1.0,
    interpolate_normals=False,
):
    glctx   = dr.RasterizeCudaContext(device)
    col_idx = pos_idx if col_idx is None else col_idx

    Kd = vec3(0.4).to(device) # diffuse
    Ka = vec3(0.15).to(device) # ambient

    # Rasterize
    pos_clip    = transform_pos_batch(mvp, pos, device).contiguous()
    rast_out, _ = dr.rasterize(glctx, pos_clip, pos_idx, [res, res], grad_db=False)

    # Process Normals
    normals = normalize(transform_norm_batch(nmvp, normals, device)).contiguous()
    norms   = torch.zeros(*rast_out.shape[:-1], 3).to(device)
    indices = rast_out[...,-1].int().clip(1) - 1

    for i in range(rast_out.shape[0]):
        norms[i] = normals[i][indices[i]]
    norms[rast_out[...,-1].int() == 0] = 0

    # Phong shading
    color = Ka * torch.ones_like(norms)

    batch_ldirs = lights[0][None, ...].repeat(nmvp.shape[0],1,1,).to(device)
    ldirs = normalize(transform_norm_batch(nmvp, batch_ldirs, device))
    for i, intensity in enumerate(lights[1]):
        ndotl = dot(ldirs[:, None, None, i, :], norms).clip(min=0)
        color += Kd * ndotl * intensity

    color = torch.where(rast_out[..., -1:] == 0, bg_color, color)
    color = dr.antialias(color, rast_out, pos_clip, pos_idx)

    return color

@torch.no_grad()
def render_phong(
    mvp, nmvp, cpos,
    pos, pos_idx, normals,
    col, col_idx=None,
    res=64, device='cpu',
    ldir=None, bg_color=1.0,
):
    ###############################
    # THIS FUNCTION DOES NOT WORK #
    ###############################

    glctx   = dr.RasterizeCudaContext(device)
    ldir    = vec(0,0,1).to(device) if ldir is None else ldir.to(device)
    zero    = vec(0).to(device)
    col_idx = pos_idx if col_idx is None else col_idx

    alpha = 3.0

    Kd = vec3(0.4).to(device) # diffuse
    Ks = vec3(3.0).to(device) # specular
    Ka = vec3(0.1).to(device) # ambient

    # Rasterize
    pos_clip    = transform_pos(mvp, pos, device)
    rast_out, _ = dr.rasterize(glctx, pos_clip, pos_idx, [res, res], grad_db=False)

    # Process Normals
    normals  = normalize(transform_norm(nmvp, normals, device)).contiguous()
    norms, _ = dr.interpolate(normals, rast_out, pos_idx)
    norms    = normalize(norms)

    # Process Reflections
    vdir    = normalize(cpos[None, None, :] - pos[..., :3])
    vdir, _ = dr.interpolate(vdir, rast_out, pos_idx)
    vdir    = normalize(vdir)

    # Phong shading
    ndotl = dot(ldir, norms).clip(min=0)
    rdotv = dot(ldir-2*ndotl*norms, vdir).clip(min=0)
    color = Ka + Kd * ndotl + Ks * (rdotv ** alpha)
    color = torch.where(rast_out[..., -1:] == 0, bg_color, color)
    color = dr.antialias(color, rast_out, pos_clip, pos_idx)

    return color

def main():
    device = 8
    v,f,vn,fn,c = load_mesh("test.obj", device=device)
    proj = projection(x=np.tan(0.6911110281944275), n=0.1, f=1000, device=device)
    mv = torch.load("transform_matrix_train.pt").float().to(device)
    nm = mv.transpose(-1,-2)
    mvp = proj @ inv(mv)

    lights = (
        torch.tensor([
            [-1,  3,  3],
            [-1, -1,  3],
            [ 0,  0, -3],
            [ 4,  0,  3],
        ]).float(), (
            0.7,
            0.9,
            1.2,
            0.4,
        )
    )

    batch_v  = v[None, ...].repeat(len(mvp),1,1)
    color = render_diffuse_batch(mvp, nm, batch_v, f, fn, c, res=512, device=device, lights=lights)
    save_img_batch(color, [f"./test_render/diffuse_{i}.png" for i in range(len(mvp))])

if __name__ == '__main__':
    main()
