import numpy as np 

def random_walk_absorption_nD(n_dim, R, start_position, step_size, max_steps=100000):
    '''
    Simulate a random walk with a browninan motion
    '''

    if n_dim <= 0:
        raise ValueError("dimension not valid")
    if len(start_position) != n_dim:
        raise ValueError(f"vector size")
        
    x_centre = np.zeros(n_dim) 
    current_position = np.copy(start_position)
    
    # Vérification initiale
    if np.linalg.norm(current_position - x_centre) >= R:
        raise ValueError("choose a starting point inside the ball")

    trajectory = [current_position]
    hit_boundary = False
    step_counter = 0


    while not hit_boundary and step_counter < max_steps:
        
        # Génération du pas aléatoire (Mouvement Brownien)
        dx = np.random.normal(loc=0.0, scale=step_size, size=n_dim)
        
        # Nouvelle position
        next_position = current_position + dx
        
        # Vérification de la condition d'absorption (Norme Euclidienne)
        distance_to_centre = np.linalg.norm(next_position - x_centre)
        
        if distance_to_centre >= R:
            hit_boundary = True
            
            # Calcul précis du point de sortie sur la frontière (Projection)
            vector_to_centre = next_position - x_centre
            boundary_position = x_centre + (R / distance_to_centre) * vector_to_centre
            
            trajectory.append(boundary_position)
        else:
            # Le point est toujours à l'intérieur
            current_position = next_position
            trajectory.append(current_position)
            
        step_counter += 1

    if step_counter == max_steps and not hit_boundary:
        print("Walk is too long")
    elif hit_boundary:
        print(f"Finish in {len(trajectory) - 1} step")

    return np.array(trajectory)


def sample_unit_sphere(dim):
    '''
    Sample a random vector on the unit sphere 
    '''
    v = np.random.normal(0, 1, dim)
    norm = np.linalg.norm(v)

    if norm == 0: return v

    return v / norm

def get_wos_path(start_point, sdf_func, epsilon=1e-4, max_steps=1000):
    '''Simulate the Walk on sphere algorithm'''
    
    path = [start_point]
    current_pos = np.copy(start_point)
    dim = len(start_point)
    
    for i in range(max_steps):
        
        radius = abs(sdf_func(current_pos))
        
        # Stop condition 
        if radius < epsilon:
            break
            
        direction = sample_unit_sphere(dim)
        current_pos = current_pos + direction * radius
        path.append(np.copy(current_pos))
        
    return np.array(path)
