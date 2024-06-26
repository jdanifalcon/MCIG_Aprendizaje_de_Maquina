# -*- coding: utf-8 -*-
"""tarea2_jdof.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1wP_vsLTC5yljNqezsiq7n4H7QH1Pc2UM
"""

import numpy as np
import matplotlib.pyplot as plt

# Definir la matriz Sigma como una matriz simétrica y semidefinida positiva
Sigma = np.array([[2, 0], [0, 1]])

# Definir una función para calcular la norma \|x\|_\Sigma
def norm_Sigma(x, Sigma):
    """
    Calcula la norma de un vector x con respecto a la matriz de covarianza Sigma.

    Parámetros:
        x (array): El vector para el cual se calculará la norma.
        Sigma (array): La matriz de covarianza.

    Retorna:
        float: La norma de x con respecto a Sigma.
    """
    return np.sqrt(np.dot(x.T, np.dot(Sigma, x)))

# Crear una serie de vectores x para ilustrar la norma
vectors = [np.array([1, 1]), np.array([2, 3]), np.array([4, -1]), np.array([-3, 2])]

# Crear la gráfica
plt.figure(figsize=(8, 8))
plt.axhline(0, color='black', linewidth=0.5)
plt.axvline(0, color='black', linewidth=0.5)
plt.grid(color='gray', linestyle='--', linewidth=0.5)

# Dibujar los vectores y sus normas
plt.quiver(0, 0, vectors[0][0], vectors[0][1], angles='xy', scale_units='xy', scale=1, color='purple')
plt.text(vectors[0][0], vectors[0][1], f'{norm_Sigma(vectors[0], Sigma):.2f}', fontsize=12, ha='right')

plt.quiver(0, 0, vectors[1][0], vectors[1][1], angles='xy', scale_units='xy', scale=1, color='orange')
plt.text(vectors[1][0], vectors[1][1], f'{norm_Sigma(vectors[1], Sigma):.2f}', fontsize=12, ha='right')

plt.quiver(0, 0, vectors[2][0], vectors[2][1], angles='xy', scale_units='xy', scale=1, color='black')
plt.text(vectors[2][0], vectors[2][1], f'{norm_Sigma(vectors[2], Sigma):.2f}', fontsize=12, ha='right')

plt.quiver(0, 0, vectors[3][0], vectors[3][1], angles='xy', scale_units='xy', scale=1, color='red')
plt.text(vectors[3][0], vectors[3][1], f'{norm_Sigma(vectors[3], Sigma):.2f}', fontsize=12, ha='right')

plt.xlim(-5, 5)
plt.ylim(-5, 5)
plt.xlabel('$x_1$')
plt.ylabel('$x_2$')
plt.title('Norma $\|x\|_\Sigma = \sqrt{x^T \Sigma x}$ con $\Sigma$')
plt.show()