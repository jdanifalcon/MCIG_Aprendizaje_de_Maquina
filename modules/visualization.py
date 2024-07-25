import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def plot_points(point, transformed_point):
    """
    Graficar los puntos originales y transformados.

    Parámetros:
    - point: np.array, punto original.
    - transformed_point: np.array, punto transformado.
    """
    plt.figure(figsize=(8, 6))
    plt.plot(point[0], point[1], 'bo', label='Punto Original')
    plt.plot(transformed_point[0][0], transformed_point[1][0], 'ro', label='Punto Transformado')
    plt.xlabel('Eje X')
    plt.ylabel('Eje Y')
    plt.title('Transformación Afín')
    plt.legend()
    plt.grid(True)
    plt.axis('equal')
    plt.show()

def show_image(original_image, transformed_image):
    """
    Mostrar una imagen usando matplotlib.

    Parámetros:
    - original_image: np.array, imagen original.
    - transformed_image: np.array, imagen transformada.
    """
    # Crear un subplot con 1 fila y 2 columnas
    fig, axes = plt.subplots(1, 2)

    # Mostrar la imagen original en el primer subplot
    axes[0].imshow(original_image, cmap='gray')
    axes[0].set_title('Imagen Original')

    # Mostrar la imagen transformada en el segundo subplot
    axes[1].imshow(transformed_image, cmap='gray')
    axes[1].set_title('Imagen Transformada')

    # Mostrar la gráfica
    plt.show()

def plot_norm_excersice(X, Y, Z):
    """
    Graficar la forma cuadrática sqrt(x.T Sigma x).

    Parámetros:
    - X: np.array, valores de x.
    - Y: np.array, valores de y.
    - Z: np.array, valores de la norma.
    """
    plt.figure(figsize=(6, 6))
    plt.contour(X, Y, Z, levels=[1], colors='purple')
    plt.title(r'Norma  $\sqrt{x^T \Sigma x}$')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.grid(True)
    plt.axis('equal')
    plt.show()

def plot_periodic_data(x, y, y_true):
    """
    Graficar los datos.

    Parámetros:
    - x: array de forma (N,), los datos de entrada.
    - y: array de forma (N,), los datos de salida.
    - y_true: array de forma (N,), la función verdadera.
    """
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x=x, y=y, label='Puntos de Datos', color='purple', alpha=0.2)
    sns.lineplot(x=x, y=y_true, label='Función Verdadera', color='black')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Datos Sintéticos con Patrón Periódico')
    plt.legend()
    plt.savefig('synthetic_data.png')
    plt.show()

def plot_classfication(array_accuracy_implementation, array_accuracy_sklearn, xlabel, ylabel, title, filename):
    """
    Esta función grafica los datos de clasificación para homeworks/8-homework_jmpc.

    Parámetros:
    - array_accuracy_implementation: lista de precisiones de la implementación de PCA.
    - array_accuracy_sklearn: lista de precisiones de PCA tradicional.
    - xlabel: string, etiqueta del eje X.
    - ylabel: string, etiqueta del eje Y.
    - title: string, título de la gráfica.
    - filename: string, nombre del archivo para guardar la gráfica.
    """
    feature_numbers = np.arange(1, len(array_accuracy_implementation) + 1)
    plt.figure(figsize=(10, 6))

    # Puntajes PCA
    plt.plot(feature_numbers, array_accuracy_implementation, color='orange', label='Puntajes PCA', linestyle='dashed')
    plt.scatter(feature_numbers, array_accuracy_implementation, color='orange', marker='s')

    # PCA Tradicional
    plt.plot(feature_numbers, array_accuracy_sklearn, color='purple', label='PCA')
    plt.scatter(feature_numbers, array_accuracy_sklearn, color='purple')

    # Añadir etiquetas y título
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)

    # Mostrar la leyenda
    plt.legend(loc='lower right')

    # Configurar las marcas del eje X
    plt.xticks(feature_numbers)

    # Guardar la gráfica
    plt.savefig(filename, dpi=300)

    # Mostrar la gráfica
    plt.show()
