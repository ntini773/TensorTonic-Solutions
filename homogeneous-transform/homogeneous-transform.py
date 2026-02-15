import numpy as np

def apply_homogeneous_transform(T, points):
    """
    Apply 4x4 homogeneous transform T to 3D point(s).
    """
    # Your code here
    homogeneous_points = np.array(points)
    homogeneous_points = homogeneous_points.reshape(-1,3)
    homogeneous_points = np.concatenate((homogeneous_points, np.ones((homogeneous_points.shape[0], 1))), axis=1)

    return ((T @ homogeneous_points.T).T)[:,:3].squeeze()