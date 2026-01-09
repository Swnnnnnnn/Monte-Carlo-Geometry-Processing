import numpy as np

#expérience 1 : condensateur 

def boundary_condition_plates(p):
    '''
    Limit condition of a condensator
    '''

    x, y = p
    return x*(y-2)

def electron_density_term(p):
    '''
    Create a density with elctrons 
    '''
    
    x, y = p
    dist_from_left_wall = x - (-1.0)
    
    density_val = 60.0 * np.exp(-8.0 * dist_from_left_wall)
    
    return density_val

#expérience 2 : soupe de polygone

def get_hit_color_vectorized(P_hit, A_vec, B_vec, Colors_vec):
    """
    Define the color of a point in the Polygon soup. 
    """


    P_exp = P_hit[np.newaxis, :, :]
    A_exp = A_vec[:, np.newaxis, :]
    B_exp = B_vec[:, np.newaxis, :]
    
    PA = P_exp - A_exp
    BA = B_exp - A_exp
    L2 = np.sum(BA**2, axis=2)
    H = np.clip(np.sum(PA * BA, axis=2) / (L2 + 1e-8), 0.0, 1.0)[:, :, np.newaxis]
    
    D_vec = PA - BA * H
    dists_sq = np.sum(D_vec**2, axis=2) # (N_lines, N_hit_points)
    
    closest_line_idx = np.argmin(dists_sq, axis=0)
    
    return Colors_vec[closest_line_idx]

## Réacteur 3D

def bc_reactor_3d(p):
    """
    Limit condition of a reactor in 3D with a warm heart and a cold border. 
    """

    d_sphere = np.linalg.norm(p, axis=1) - 0.4
    return (np.abs(d_sphere) < 0.05).astype(float)