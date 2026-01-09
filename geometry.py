import numpy as np 
from scipy.spatial import cKDTree
import numpy as np
import matplotlib.pyplot as plt
from geometry import get_distance

def sdf_peanut(p):
    '''Defining a more complex geometry to visualize adaptation of the Walk On Sphere algorithm'''
    c1 = np.array([-0.5, 0.0])
    d1 = np.linalg.norm(p - c1) - 0.8
    
    c2 = np.array([0.5, 0.0])
    d2 = np.linalg.norm(p - c2) - 0.6
    
    return min(d1, d2)

def circle_sdf(p):
    '''Define the geometry of a ball'''

    R = 5.0
    return R - np.linalg.norm(p)

def sdf_box(p):
    '''It definies a box'''
    w, h = 1.0, 1.0 # Demi-largeur, Demi-hauteur

    dist_x = np.abs(p[0]) - w
    dist_y = np.abs(p[1]) - h
    return max(dist_x, dist_y)

def sd_segment_vectorized(p, a, b):
    """
    define a segment 
    """
    pa = p - a
    ba = b - a

    dot_pa_ba = np.sum(pa * ba, axis=1)
    dot_ba_ba = np.dot(ba, ba)
    
    h = np.clip(dot_pa_ba / dot_ba_ba, 0.0, 1.0)
    
    return np.linalg.norm(pa - ba * h[:, np.newaxis], axis=1)



def get_min_dist_vectorized(P,A_vec,B_vec):
    """
    Compute the distance of point to a segment
    P shape: (N_points, 2)
    """
    P_expanded = P[np.newaxis, :, :]  # (1, N_points, 2)
    A_expanded = A_vec[:, np.newaxis, :] # (N_lines, 1, 2)
    B_expanded = B_vec[:, np.newaxis, :]
    
    # Vectors PA et BA
    PA = P_expanded - A_expanded
    BA = B_expanded - A_expanded
    
    L2 = np.sum(BA**2, axis=2)
    H = np.sum(PA * BA, axis=2) / (L2 + 1e-8) # Evite div/0
    H = np.clip(H, 0.0, 1.0)
    
    H_expanded = H[:, :, np.newaxis]
    D_vec = PA - BA * H_expanded
    
    dists_sq = np.sum(D_vec**2, axis=2) # (N_lines, N_points)
    
    min_dists = np.sqrt(np.min(dists_sq, axis=0))
    
    return min_dists



def generate_points_on_segment(a, b, density=50):
    """Create a discretization of Space"""
    return np.linspace(a, b, density)


def sdf_reactor_3d(p):
    """
    Reactor in 3D
    """
    q = np.abs(p) - 1.0
    dist_box_signed = np.linalg.norm(np.maximum(q, 0.0), axis=1) + np.minimum(np.max(q, axis=1), 0.0)
    
    dist_sphere_signed = np.linalg.norm(p, axis=1) - 0.4
    
    d_walls = -dist_box_signed
    d_core = dist_sphere_signed
    
    return np.minimum(d_walls, d_core)


def cubic_bezier(t, p0, p1, p2, p3):
    """
    define bezier curves 
    """
    return (1-t)**3 * p0 + \
           3 * (1-t)**2 * t * p1 + \
           3 * (1-t) * t**2 * p2 + \
           t**3 * p3


def evaluate_bezier_cubic(P, t):
    t = t[:, np.newaxis] 
    return (1-t)**3 * P[0] + 3*(1-t)**2 * t * P[1] + \
           3*(1-t) * t**2 * P[2] + t**3 * P[3]

class FastCurveScene:
    """Gère la discrétisation des courbes et la recherche rapide (KD-Tree)."""
    def __init__(self):
        self.sample_points = []
        self.sample_colors = []
        self.tree = None
        self.all_colors = None
        
    def add_curve(self, p0, p1, p2, p3, color, resolution=500):
        P = np.array([p0, p1, p2, p3])
        t_vals = np.linspace(0, 1, resolution)
        points = evaluate_bezier_cubic(P, t_vals)
        
        self.sample_points.append(points)
        self.sample_colors.append(np.tile(color, (resolution, 1)))
        
    def build(self):
        if not self.sample_points: return
        self.all_points = np.vstack(self.sample_points)
        self.all_colors = np.vstack(self.sample_colors)
        # Construction de l'arbre pour les requêtes rapides
        self.tree = cKDTree(self.all_points)

class VoronoiScene:
    def __init__(self, num_seeds=20):
        self.seeds = np.random.rand(num_seeds, 2) * 100
        self.colors = np.random.rand(num_seeds, 3)
        self.tree = cKDTree(self.seeds)


L = 10.0
W = 1.0
def heat_sdf_wrapper(p):
    x = p[:, 0]
    y = p[:, 1]
    return np.minimum.reduce([x, L - x, y, W - y])

def heat_bc_wrapper(p):
    x = p[:, 0]
    y = p[:, 1]
    values = np.zeros(len(p))
    is_right_wall = x > (L - 0.1) 
    
    if np.any(is_right_wall):
        values[is_right_wall] = np.sin(np.pi * y[is_right_wall] / W)
        
    return values

def get_distance(pos):
    x, y = pos
    return min(x, 1-x, y, 1-y)

