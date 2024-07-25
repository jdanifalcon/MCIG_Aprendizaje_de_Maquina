import numpy as np

def rotation_matrix(theta):
    """
    Genera una matriz de rotación 2x2 basada en el ángulo de rotación.
    
    - Parámetros:
      theta (float): El ángulo de rotación en grados.
    
    - Retorno:
      numpy.ndarray: La matriz de rotación correspondiente a theta.
    """
    # Convertir el ángulo a radianes
    theta_rad = np.radians(theta)

    # Calcular los valores de la matriz de rotación
    cos_theta = np.cos(theta_rad)
    sin_theta = np.sin(theta_rad)

    # Construir la matriz
    return np.array([[cos_theta, -sin_theta], [sin_theta, cos_theta]])

def scaling_matrix(sx, sy):
    """
    Genera una matriz de escala 2x2 basada en sx y sy.
    
    - Parámetros:
        sx (float): Factor de escala para el eje x.
        sy (float): Factor de escala para el eje y.
    
    - Retorno:
        numpy.ndarray: Una matriz de escala 2x2.
    """
    return np.array([[sx, 0], [0, sy]])

def shearing_matrix(h):
    """
    Genera una matriz de cizallamiento 2x2.
    
    - Parámetros:
        h (float): Factor de cizallamiento.
    
    - Retorno:
        numpy.ndarray: La matriz de cizallamiento 2x2.
    """
    return np.array([[1, h], [0, 1]])

def compute_matrix_transformation(R, S, H):
    """
    Computa una matriz A, donde A = R*S*H.
    
    - Parámetros:
        R (np.ndarray): Matriz de rotación.
        S (np.ndarray): Matriz de escala.
        H (np.ndarray): Matriz de cizallamiento.
    
    - Retorno:
        np.ndarray: La matriz de transformación A.
    """
    return np.matmul(H, np.matmul(S, R))

def get_matrix_augmented(A, b):
    """
    Crea una matriz aumentada con A y b.
    
    - Parámetros:
        A (ndarray): Una matriz de transformación lineal 2x2.
        b (ndarray): El vector de traslación (2x1).
    
    - Retorno:
        ndarray: La matriz aumentada con A y b.
    """
    A_augmented = np.hstack((A, b.reshape(-1, 1)))
    A_augmented = np.vstack((A_augmented, [0, 0, 1]))
    return A_augmented

def get_point_homogeneous(point):
    """
    Crea un punto homogéneo.
    
    - Parámetros:
        point (ndarray): Las coordenadas del punto (2x1).
    
    - Retorno:
        ndarray: El punto homogéneo.
    """
    return np.hstack((point, [1])).reshape(-1, 1)

def compute_affine(A, b, point):
    """
    Computa la transformación afín.
    
    - Parámetros:
        A (ndarray): Una matriz de transformación lineal 2x2.
        b (ndarray): El vector de traslación (2x1).
        point (ndarray): Las coordenadas del punto (2x1).
    
    - Retorno:
        ndarray: El punto transformado, fila[0] = x, fila[1] = y.
    """
    point_homogeneous = get_point_homogeneous(point)
    A_augmented = get_matrix_augmented(A, b)
    transformed_point_homogeneous = np.dot(A_augmented, point_homogeneous)
    return transformed_point_homogeneous[:2]

def fx(x1, x2, x3):
    """
    Computa la función f(x1, x2, x3) = 10*x1**2 - x1*x2 - 5*x1*x3 + 5*x1 + 10*x2**2 - 11*x2*x3 - 2*x2 - 5*x3*x1 - 11*x3*x2 - 4*x2 + 6*x3 + 9.
    
    - Parámetros:
        x1 (float): El valor de x1.
        x2 (float): El valor de x2.
        x3 (float): El valor de x3.
    
    - Retorno:
        float: El resultado de la función.
    """
    return 10*x1**2 - x1*x2 - 5*x1*x3 + 5*x1 + 10*x2**2 - 11*x2*x3 - 2*x2 - 5*x3*x1 - 11*x3*x2 - 4*x2 + 6*x3 + 9

def center_data(X):
    """
    Función para centrar los datos: X - X.mean(axis=0).
    
    - Parámetros:
        X (numpy array de forma (n, d)): La matriz de datos.
    
    - Retorno:
        numpy array de forma (n, d): La matriz de datos centrada.
    """
    return X - np.mean(X, axis=0)
