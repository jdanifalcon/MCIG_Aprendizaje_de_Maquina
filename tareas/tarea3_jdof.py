# -*- coding: utf-8 -*-
"""tarea3_jdof.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1KoB97bpYWiQF-V_uhO_M8rZkfVKyD5AG

![cg.jpg](https://raw.githubusercontent.com/jdanifalcon/FundamentosIA/main/logo/logo_cg.jpg)


# Tarea 3

#### Aprendizaje de máquina
#### @date 22/05/2023
#### @autor: Jessica Daniela Ocaña Falcón
"""

# Importar las librerías necesarias
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Definición de la función f1 y su gradiente
def f1(x1, x2):
    # Calcula los valores de la función cuadrática f1
    return 10*x1**2 - x1*x2 + 5*x1 + 10*x2**2 - 2*x2 + 9

def grad_f1(x1, x2):
    # Calcula las derivadas parciales (gradiente) de la función f1
    df1_dx1 = 20*x1 - x2 + 5
    df1_dx2 = -x1 + 20*x2 - 2
    return np.array([df1_dx1, df1_dx2])

# Rango de valores para x1 y x2
x1 = np.linspace(-2, 2, 400)  # Crear 400 puntos en el rango de -2 a 2 para x1
x2 = np.linspace(-2, 2, 400)  # Crear 400 puntos en el rango de -2 a 2 para x2
x1, x2 = np.meshgrid(x1, x2)  # Crear una malla de valores de x1 y x2
f_values = f1(x1, x2)  # Evaluar la función f1 en cada punto de la malla

# Gráfica de la función f1 en 2D con contornos
plt.figure(figsize=(10, 6))  # Crear una figura de tamaño 10x6
contours = plt.contour(x1, x2, f_values, 50, cmap='magma')  # Dibujar los contornos de la función f1
plt.clabel(contours, inline=True, fontsize=8)  # Etiquetar los contornos
plt.title('Contorno de la función $f_1(x_1, x_2)$')  # Título de la gráfica
plt.xlabel('$x_1$')  # Etiqueta del eje x
plt.ylabel('$x_2$')  # Etiqueta del eje y

# Gradiente
X, Y = np.meshgrid(np.linspace(-2, 2, 20), np.linspace(-2, 2, 20))  # Crear una malla de menor resolución para el campo de gradiente
U, V = grad_f1(X, Y)  # Calcular las componentes del gradiente

# Gráfica del gradiente en el contorno
plt.quiver(X, Y, U, V, color='red')  # Dibujar las flechas del campo de gradiente en la gráfica de contorno
plt.title('Campo de gradiente de la función $f_1(x_1, x_2)$')  # Título de la gráfica
plt.xlabel('$x_1$')  # Etiqueta del eje x
plt.ylabel('$x_2$')  # Etiqueta del eje y
plt.grid(True)  # Mostrar la cuadrícula
plt.show()  # Mostrar la gráfica

# Crear figura 3D
fig = plt.figure(figsize=(12, 8))  # Crear una figura de tamaño 12x8
ax = fig.add_subplot(111, projection='3d')  # Añadir un subplot 3D

# Gráfica de la función f1 en 3D
surf = ax.plot_surface(x1, x2, f_values, cmap='magma', edgecolor='none')  # Dibujar la superficie de la función f1
fig.colorbar(surf, ax=ax, shrink=0.5, aspect=5)  # Añadir una barra de colores
ax.set_title('Superficie de la función $f_1(x_1, x_2)$')  # Título de la gráfica
ax.set_xlabel('$x_1$')  # Etiqueta del eje x
ax.set_ylabel('$x_2$')  # Etiqueta del eje y
ax.set_zlabel('$f_1(x_1, x_2)$')  # Etiqueta del eje z

# Escalar el gradiente para reducir su longitud
scale_factor = 10  # Factor de escala para las flechas del gradiente
U_scaled = U / scale_factor  # Escalar las componentes del gradiente en x
V_scaled = V / scale_factor  # Escalar las componentes del gradiente en y

# Gráfica del gradiente en 3D con flechas menos saturadas
ax.quiver(X, Y, f1(X, Y), U_scaled, V_scaled, np.zeros_like(U_scaled), color='orange', length=0.1, normalize=True)  # Dibujar las flechas del gradiente en 3D

plt.show()  # Mostrar la gráfica

"""**Gráfica $f_1(x_1,x_2)$**

La gráfica generada muestra la superficie de la función $f_1(x_1, x_2)$ en 3D junto con el campo de gradiente.

1. **Superficie de la Función**:
   - La superficie en 3D permite visualizar cómo cambia la función $f_1$ en el espacio tridimensional. Las zonas más altas y bajas de la superficie corresponden a los valores máximos y mínimos de la función, respectivamente.

2. **Campo de Gradiente**:
   - Las flechas naranjas representan el gradiente de $f_1$ en diferentes puntos de la superficie. Indican la dirección del cambio más rápido de la función. En un valle (mínimo), las flechas apuntan hacia arriba y fuera del valle, mientras que en una cresta (máximo), las flechas apuntan hacia abajo y fuera de la cresta.
   - La longitud y dirección de las flechas muestran cómo la pendiente cambia en cada punto. Flechas más largas indican un cambio más rápido en esa dirección, mientras que flechas más cortas indican un cambio más lento.

En resumen, la visualización conjunta de la superficie de la función y su campo de gradiente nos da una comprensión intuitiva de cómo la función se comporta y cómo cambia en cada punto del espacio $(x_1, x_2)$.
"""

# Importar las librerías necesarias
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Definición de la función f2 y su gradiente
def f2(x1, x2):
    # Calcula los valores de la función cúbica f2
    return x1**3 + x2**3 - 3*x1 - 3*x2

def grad_f2(x1, x2):
    # Calcula las derivadas parciales (gradiente) de la función f2
    df2_dx1 = 3*x1**2 - 3
    df2_dx2 = 3*x2**2 - 3
    return np.array([df2_dx1, df2_dx2])

# Rango de valores para x1 y x2
x1 = np.linspace(-2, 2, 400)  # Crear 400 puntos en el rango de -2 a 2 para x1
x2 = np.linspace(-2, 2, 400)  # Crear 400 puntos en el rango de -2 a 2 para x2
x1, x2 = np.meshgrid(x1, x2)  # Crear una malla de valores de x1 y x2
f_values = f2(x1, x2)  # Evaluar la función f2 en cada punto de la malla

# Gráfica de la función f2 en 2D con contornos
plt.figure(figsize=(10, 6))  # Crear una figura de tamaño 10x6
contours = plt.contour(x1, x2, f_values, 50, cmap='magma')  # Dibujar los contornos de la función f2
plt.clabel(contours, inline=True, fontsize=8)  # Etiquetar los contornos
plt.title('Contorno de la función $f_2(x_1, x_2)$')  # Título de la gráfica
plt.xlabel('$x_1$')  # Etiqueta del eje x
plt.ylabel('$x_2$')  # Etiqueta del eje y

# Gradiente
X, Y = np.meshgrid(np.linspace(-2, 2, 20), np.linspace(-2, 2, 20))  # Crear una malla de menor resolución para el campo de gradiente
U, V = grad_f2(X, Y)  # Calcular las componentes del gradiente

# Gráfica del gradiente en el contorno
plt.quiver(X, Y, U, V, color='red')  # Dibujar las flechas del campo de gradiente en la gráfica de contorno
plt.title('Campo de gradiente de la función $f_2(x_1, x_2)$')  # Título de la gráfica
plt.xlabel('$x_1$')  # Etiqueta del eje x
plt.ylabel('$x_2$')  # Etiqueta del eje y
plt.grid(True)  # Mostrar la cuadrícula
plt.show()  # Mostrar la gráfica

# Crear figura 3D
fig = plt.figure(figsize=(12, 8))  # Crear una figura de tamaño 12x8
ax = fig.add_subplot(111, projection='3d')  # Añadir un subplot 3D

# Gráfica de la función f2 en 3D
surf = ax.plot_surface(x1, x2, f_values, cmap='magma', edgecolor='none')  # Dibujar la superficie de la función f2
fig.colorbar(surf, ax=ax, shrink=0.5, aspect=5)  # Añadir una barra de colores
ax.set_title('Superficie de la función $f_2(x_1, x_2)$')  # Título de la gráfica
ax.set_xlabel('$x_1$')  # Etiqueta del eje x
ax.set_ylabel('$x_2$')  # Etiqueta del eje y
ax.set_zlabel('$f_2(x_1, x_2)$')  # Etiqueta del eje z

# Escalar el gradiente para reducir su longitud
scale_factor = 10  # Factor de escala para las flechas del gradiente
U_scaled = U / scale_factor  # Escalar las componentes del gradiente en x
V_scaled = V / scale_factor  # Escalar las componentes del gradiente en y

# Gráfica del gradiente en 3D con flechas menos saturadas
ax.quiver(X, Y, f2(X, Y), U_scaled, V_scaled, np.zeros_like(U_scaled), color='orange', length=0.1, normalize=True)  # Dibujar las flechas del gradiente en 3D

plt.show()  # Mostrar la gráfica

"""**Gráfica $f_2(x_1, x_2)$**

Ahora con $f_2(x_1, x_2)$ hay una visualización completa de cómo se comporta la función en un espacio tridimensional y muestra las direcciones de los gradientes en ese espacio.

#### Superficie de la Función

1. **Superficie de la función**: La gráfica 3D de la superficie de $f_2(x_1, x_2)$ revela cómo varían los valores de $f_2$ en función de $x_1$ y $x_2$. La superficie muestra las crestas y los valles, indicando los puntos altos y bajos de la función.

2. **Estructura de la función**: La gráfica muestra que la función tiene una forma ondulada con varias crestas y valles. Esto indica que la función tiene múltiples máximos y mínimos locales.

#### Campo de Gradiente

1. **Dirección del gradiente**: Las flechas naranjas en la gráfica representan el gradiente de la función en varios puntos de la superficie. La dirección de las flechas indica la dirección de la pendiente más pronunciada en cada punto. Es decir, las flechas apuntan en la dirección en la que la función aumenta más rápidamente.

2. **Magnitud del gradiente**: La longitud de las flechas del gradiente está escalada para ser más visible en la gráfica. La magnitud del gradiente indica la velocidad con la que la función está cambiando en esa dirección. Flechas más largas representarían un cambio más rápido.

3. **Puntos críticos**: Los puntos donde las flechas son muy cortas o no hay flechas indican posibles puntos críticos (máximos, mínimos o puntos de silla) donde el gradiente es cero o muy pequeño.

### En las observaciones se puede entender que:

- **Máximos y mínimos locales**: Se encontro que hay máximos y mínimos locales en la superficie, correspondientes a las crestas y valles. Estos son los puntos donde la función cambia de dirección.
- **Puntos de inflexión**: Las áreas donde la curvatura de la superficie cambia de positiva a negativa (o viceversa) pueden ser puntos de inflexión.

"""