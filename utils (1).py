import numpy as np
import matplotlib.pyplot as plt
from geometry import *
from scipy.spatial import Delaunay
from scipy.sparse import lil_matrix
from scipy.sparse.linalg import spsolve
from matplotlib.collections import PolyCollection

def green_function_2d(R, r):
    """Define the Green fonction for our source term"""
    if r < 1e-9: r = 1e-9 
    return (1.0 / (2.0 * np.pi)) * np.log(R / r)

def sample_unit_sphere(dim=2):
    """Sample a vector in the unit sphere"""
    v = np.random.normal(0, 1, dim)
    norm = np.linalg.norm(v)
    if norm < 1e-9: return v 
    return v / norm

def sample_solid_disk(center, R):
    """Sample on a disk"""
    theta = np.random.uniform(0, 2 * np.pi)
    r_dist = np.sqrt(np.random.uniform(0, 1)) * R
    offset = np.array([r_dist * np.cos(theta), r_dist * np.sin(theta)])
    return center + offset, r_dist

def wos_poisson_2d(point, sdf, bc, src=None, n_walks=1, max_steps=100, eps=1e-4):
    '''Solve simple equation for Poisson problem'''

    dist = sdf(point)
    if dist >= -eps: return bc(point)

    total_est = 0.0

    for _ in range(n_walks):
        curr = np.copy(point)
        path_source_val = 0.0
        hit = False
        
        for _ in range(max_steps):
            R = abs(sdf(curr))
            
            if R < eps:
                total_est += bc(curr)
                hit = True
                break
            
            if src is not None: ## if there is a source term
                y_k, dist_y_k = sample_solid_disk(curr, R)
                vol = np.pi * R**2
                path_source_val += vol * src(y_k) * green_function_2d(R, dist_y_k)
            
            curr += sample_unit_sphere(2) * R
        
        if hit:
            total_est += path_source_val

    return total_est / n_walks

def wos_cv_2d(point, sdf, bc, src=None, n_samples=30):
    '''
    Add a control variate to our estimator
    '''

    dist = sdf(point)
    if dist >= -1e-4: return bc(point)
    
    R = abs(dist)
    dim = point.shape[0]
    
    total_value = 0.0
    running_grad_sum = np.zeros(dim)
    
    for k in range(1, n_samples + 1):
        
        u_dir = sample_unit_sphere(dim)
        x1 = point + u_dir * R
        
        val_boundary_path = wos_poisson_2d(x1, sdf, bc, src, n_walks=1)
        
        val_source_local = 0.0
        if src is not None: ## source term
            y_source, dist_y = sample_solid_disk(point, R)
            vol = np.pi * R**2
            val_source_local = vol * src(y_source) * green_function_2d(R, dist_y)
            
        val_sample = val_boundary_path + val_source_local
        
        current_grad_est = (dim / R) * val_boundary_path * u_dir
        
        control_term = 0.0
        if k > 1:
            avg_grad = running_grad_sum / (k - 1)
            control_term = -np.dot(avg_grad, u_dir * R)
            
        total_value += (val_sample + control_term)
        running_grad_sum += current_grad_est

    return total_value / n_samples


## Vectorization 

def solve_wos_generic(res, n_samples, sdf_func, bc_func, 
              bg_val=0.0, 
              domain_limit=1.2, 
              max_steps=50, 
              epsilon=1e-2,
              return_raw=False): 
    """
    Solve Wos Problem by adapting to structure of it
    """
    
    is_scalar = np.isscalar(bg_val)
    bg_val_array = float(bg_val) if is_scalar else np.array(bg_val)
    output_dim = 1 if is_scalar else len(bg_val)
    
    x = np.linspace(-1, 1, res)
    y = np.linspace(1, -1, res)
    X, Y = np.meshgrid(x, y)
    points_start = np.stack([X.ravel(), Y.ravel()], axis=1)
    
    active_pos = np.repeat(points_start, n_samples, axis=0)
    n_particles = active_pos.shape[0]
    
    # Final values 
    shape_vals = (n_particles) if is_scalar else (n_particles, output_dim)
    final_values = np.zeros(shape_vals)
    
    # if control variate 
    if return_raw:
        first_step_R = np.zeros(n_particles)
        first_step_offset = np.zeros((n_particles, 2))
    
    active_mask = np.ones(n_particles, dtype=bool)
    
    # solve 
    for step in range(max_steps):
        # Optimization 
        if not np.any(active_mask): break
            
        current_p = active_pos[active_mask]
        d = sdf_func(current_p)
        
        if return_raw and step == 0:
            idxs_active = np.where(active_mask)[0]
            first_step_R[idxs_active] = d
        
        hit_mask_local = d < epsilon
        kill_mask_local = (np.abs(current_p[:,0]) > domain_limit) | (np.abs(current_p[:,1]) > domain_limit)
        
        active_indices = np.where(active_mask)[0]
        
        # HITS
        hits_indices = active_indices[hit_mask_local]
        if len(hits_indices) > 0:
            val = bc_func(active_pos[hits_indices])
            final_values[hits_indices] = val
            active_mask[hits_indices] = False 
            
        # KILLS
        kills_indices = active_indices[kill_mask_local]
        if len(kills_indices) > 0:
            final_values[kills_indices] = bg_val_array
            active_mask[kills_indices] = False
        
        # jump
        continuing_mask_local = ~(hit_mask_local | kill_mask_local)
        if np.sum(continuing_mask_local) == 0: continue
            
        r = d[continuing_mask_local]
        theta = np.random.uniform(0, 2*np.pi, size=len(r))
        dx = r * np.cos(theta)
        dy = r * np.sin(theta)
        
        cont_indices = active_indices[continuing_mask_local]
        active_pos[cont_indices, 0] += dx
        active_pos[cont_indices, 1] += dy
        
        if return_raw and step == 0:
            first_step_offset[cont_indices, 0] = dx
            first_step_offset[cont_indices, 1] = dy

    # value
    if return_raw:
        return final_values, first_step_R, first_step_offset
    
    else:
        if is_scalar:
            return final_values.reshape((res*res, n_samples)).mean(axis=1).reshape((res, res))
        else:
            return final_values.reshape((res*res, n_samples, output_dim)).mean(axis=1).reshape((res, res, output_dim))
        

def render_cv_generic(res, total_samples, sdf_func, bc_func, bg_val):
    """
    call the walk on sphere 
    """
    
    # param
    n_pilot = max(4, total_samples // 2)
    n_final = total_samples - n_pilot
    is_scalar = np.isscalar(bg_val)
    dim = 1 if is_scalar else len(bg_val)
    
    
    vals_p, R_p, off_p = solve_wos_generic(res, n_pilot, sdf_func, bc_func, bg_val, return_raw=True)
    
    # gradient
    vals_exp = vals_p.reshape(-1, dim, 1) if not is_scalar else vals_p.reshape(-1, 1, 1)
    off_exp = off_p[:, np.newaxis, :]
    R_sq = (R_p**2 + 1e-8).reshape(-1, 1, 1)
    
    grads_particles = (2.0 * vals_exp * off_exp) / R_sq
    
    grads_reshaped = grads_particles.reshape(res*res, n_pilot, dim, 2)
    mean_gradient_field = np.mean(grads_reshaped, axis=1) 
    
    if is_scalar:
        img_pilot = vals_p.reshape(res*res, n_pilot).mean(axis=1).reshape(res, res)
    else:
        img_pilot = vals_p.reshape(res*res, n_pilot, dim).mean(axis=1).reshape(res, res, dim)

    
    vals_f, R_f, off_f = solve_wos_generic(res, n_final, sdf_func, bc_func, bg_val, return_raw=True)
    
    grad_per_particle = np.repeat(mean_gradient_field, n_final, axis=0)
    
    off_f_exp = off_f[:, np.newaxis, :]
    correction = np.sum(grad_per_particle * off_f_exp, axis=2)
    
    vals_corrected = vals_f.reshape(-1, dim) - correction
    
    if is_scalar:
        img_final = vals_corrected.reshape(res*res, n_final).mean(axis=1).reshape(res, res)
    else:
        img_final = vals_corrected.reshape(res*res, n_final, dim).mean(axis=1).reshape(res, res, dim)
    
    return img_pilot, img_final


def generate_mesh_from_lines(lines, density=50, n_fill=2500, box_res=30):
    """
    create a meshing with Delaunay library 
    """
    points = []
    bc_indices = []
    bc_values = []

    for line in lines:
        pts_line = np.linspace(line['a'], line['b'], density)
        start_idx = len(points)
        for i, p in enumerate(pts_line):
            points.append(p)
            bc_indices.append(start_idx + i)
            bc_values.append(line['color'])

    x = np.linspace(-1, 1, box_res)
    box_pts = []
    box_pts.extend([[v, -1] for v in x]) # Bas
    box_pts.extend([[v, 1] for v in x])  # Haut
    box_pts.extend([[-1, v] for v in x]) # Gauche
    box_pts.extend([[1, v] for v in x])  # Droite
    
    for p in box_pts:
        points.append(p)
        bc_indices.append(len(points)-1)
        bc_values.append([0.0, 0.0, 0.0])

    points.extend(np.random.uniform(-0.95, 0.95, (n_fill, 2)))
    
    # Delaunay step
    nodes = np.array(points)
    tri = Delaunay(nodes)
    
    return nodes, tri.simplices, np.array(bc_indices), np.array(bc_values)


def solve_fem_rgb(nodes, elems, bc_indices, bc_colors):
    """
    Solve Laplace with finite element method 
    """
    N = len(nodes)
    A = lil_matrix((N, N))
    
    for tri in elems:
        pts = nodes[tri]
        x, y = pts[:,0], pts[:,1]
        
        # Aire * 2
        detJ = x[0]*(y[1]-y[2]) + x[1]*(y[2]-y[0]) + x[2]*(y[0]-y[1])
        area = 0.5 * np.abs(detJ)
        if area < 1e-12: continue
        
        b = np.array([y[1]-y[2], y[2]-y[0], y[0]-y[1]])
        c = np.array([x[2]-x[1], x[0]-x[2], x[1]-x[0]])
        
        for i in range(3):
            for j in range(3):
                val = (b[i]*b[j] + c[i]*c[j]) / (4*area)
                A[tri[i], tri[j]] += val

    # limit conditions 
    B_rhs = np.zeros((N, 3))
    
    for i, idx in enumerate(bc_indices):
        A[idx, :] = 0.0
        A[idx, idx] = 1.0
        B_rhs[idx] = bc_colors[i]

    return spsolve(A.tocsr(), B_rhs)


def plot_fem_mesh(nodes, elems, U_rgb, ax=None):
    """
    plot solution 
    """
    if ax is None: ax = plt.gca()
    
    U_tri = U_rgb[elems].mean(axis=1)
    U_tri = np.clip(U_tri, 0, 1) 
    
    collection = PolyCollection(nodes[elems], array=None, 
                                facecolors=U_tri, edgecolors='none')
    
    ax.add_collection(collection)
    ax.set_xlim(-1.1, 1.1)
    ax.set_ylim(-1.1, 1.1)
    ax.axis('off')
    ax.set_aspect('equal')

def solve_wos_generic2(res, n_samples, sdf_func, bc_func, 
              bg_val=0.0, 
              domain_limit=1.2, 
              max_steps=50, 
              epsilon=1e-3,
              return_raw=False):
    """
    Adapt Wos to more complex problem
    """

    is_scalar = np.isscalar(bg_val)
    bg_val_array = float(bg_val) if is_scalar else np.array(bg_val)
    output_dim = 1 if is_scalar else len(bg_val)
    
    res_shape = None 
    
    if isinstance(res, np.ndarray):
        points_start = res
    else:
        x = np.linspace(-1, 1, res)
        y = np.linspace(1, -1, res)
        X, Y = np.meshgrid(x, y)
        points_start = np.stack([X.ravel(), Y.ravel()], axis=1)
        res_shape = (res, res) 
        
    dim_space = points_start.shape[1] 
    
    active_pos = np.repeat(points_start, n_samples, axis=0)
    n_particles = active_pos.shape[0]
    
    shape_vals = (n_particles) if is_scalar else (n_particles, output_dim)
    final_values = np.zeros(shape_vals)
    final_values[:] = np.nan 
    
    if return_raw:
        first_step_R = np.zeros(n_particles)
        first_step_offset = np.zeros((n_particles, dim_space))
    
    active_mask = np.ones(n_particles, dtype=bool)
    
    # Walk On sphere (Monte carlo)
    for step in range(max_steps):
        if not np.any(active_mask): break
            
        current_p = active_pos[active_mask]
        d = sdf_func(current_p)
        
        if return_raw and step == 0:
            idxs_active = np.where(active_mask)[0]
            first_step_R[idxs_active] = d
        
        # Hit / Kill
        hit_mask_local = d < epsilon
        kill_mask_local = np.any(np.abs(current_p) > domain_limit, axis=1)
        
        active_indices = np.where(active_mask)[0]
        
        # HITS
        hits_indices = active_indices[hit_mask_local]
        if len(hits_indices) > 0:
            val = bc_func(active_pos[hits_indices])
            final_values[hits_indices] = val
            active_mask[hits_indices] = False 
            
        # KILLS
        kills_indices = active_indices[kill_mask_local]
        if len(kills_indices) > 0:
            final_values[kills_indices] = bg_val_array
            active_mask[kills_indices] = False
        
        # WALK
        continuing_mask_local = ~(hit_mask_local | kill_mask_local)
        if np.sum(continuing_mask_local) == 0: continue
            
        r = d[continuing_mask_local]
        n_walks = len(r)
        
        random_dir = np.random.normal(0, 1, (n_walks, dim_space))
        norms = np.linalg.norm(random_dir, axis=1, keepdims=True)
        unit_dirs = random_dir / norms
        
        offsets = unit_dirs * r[:, np.newaxis]
        
        cont_indices = active_indices[continuing_mask_local]
        active_pos[cont_indices] += offsets
        
        if return_raw and step == 0:
            first_step_offset[cont_indices] = offsets

    # solution
    if return_raw:
        return final_values, first_step_R, first_step_offset
    else:
        if is_scalar:
            reshaped = final_values.reshape((points_start.shape[0], n_samples))
            pixel_means = np.nanmean(reshaped, axis=1)
            
            if res_shape is not None:
                return pixel_means.reshape(res_shape)
            else:
                return pixel_means # (N,)
        else:
            reshaped = final_values.reshape((points_start.shape[0], n_samples, output_dim))
            pixel_means = np.nanmean(reshaped, axis=1)
            
            if res_shape is not None:
                return pixel_means.reshape(res_shape + (output_dim,))
            else:
                return pixel_means
            




def solve_wos_pass(starting_points, scene, epsilon=0.5, max_steps=60):

    n_walkers = starting_points.shape[0]
    
    active_pos = starting_points.copy()
    
    final_colors = np.ones((n_walkers, 3)) 
    active_mask = np.ones(n_walkers, dtype=bool)
    
    for step in range(max_steps):
        if not np.any(active_mask): break
            
        current_active_idx = np.where(active_mask)[0]
        current_pos = active_pos[current_active_idx]
        
        dists, indices = scene.tree.query(current_pos, k=1)
        
        hit_local_mask = dists < epsilon
        
        if np.any(hit_local_mask):
            hit_global_idx = current_active_idx[hit_local_mask]
            
            hit_colors = scene.all_colors[indices[hit_local_mask]]
            final_colors[hit_global_idx] = hit_colors
            
            active_mask[hit_global_idx] = False
        
        walking_local_mask = ~hit_local_mask
        
        if np.any(walking_local_mask):
            walking_global_idx = current_active_idx[walking_local_mask]
            
            r = dists[walking_local_mask]
            
            theta = np.random.uniform(0, 2*np.pi, size=len(r))
            dx = r * np.cos(theta)
            dy = r * np.sin(theta)
            
            active_pos[walking_global_idx, 0] += dx
            active_pos[walking_global_idx, 1] += dy

    return final_colors




def estimate_gradient_monte_carlo(pos, n_samples=1000):
    """
    gradient estimation with monte carlo method and WoS
    """
    R = get_distance(pos)
    
    theta = np.random.uniform(0, 2*np.pi, size=n_samples)
    nx = np.cos(theta) 
    
    y_x = pos[0] + R * nx
    
    u_val = y_x 
    
    grad_estimates = (2.0 / R) * u_val * nx
    
    return np.mean(grad_estimates), np.std(grad_estimates)

