# Cargar librerías necesarias
import numpy as np
import pandas as pd

class PcaClass:
    def __init__(self, X, y, n_components, feature_names):
        """
        Parámetros:
        X: matriz numpy de forma (n, d), la matriz de datos
        y: vector numpy de forma (n,), el vector objetivo
        n_components: int, el número de componentes principales a mantener
        feature_names: lista de strings, los nombres de las características
        """
        self.X = X # matriz de datos
        self.y = y # vector objetivo
        self.n_components = n_components # número de componentes principales a mantener
        self.feature_names = feature_names # lista de nombres de las características
        self.eigenvalues = None # valores propios, d elementos ordenados
        self.eigenvectors = None # matriz de pesos dxd, todos los d vectores propios ordenados
        self.features_eigenvalues = None # nombres de características y valores propios, d elementos característica:valor propio
        self.features_scores = None # nombres de características y puntuaciones, d elementos característica:puntuación
        self.W_pca = None # matriz de datos en el espacio PCA nxd
        self.W_mean_class = None # media de las características para cada clase, matriz 2xd
        self.scores = None # puntuaciones de las características
        self.S_k = None # características seleccionadas por puntuación
        self.features_selected = None # características seleccionadas por puntuación

    def compute_covariance_matrix(self, X):
        """
        Computar la matriz de covarianza de los datos.
        Parámetros:
        X: matriz numpy de forma (n, d), la matriz de datos
        Retorna: cov: matriz numpy de forma (d, d), la matriz de covarianza
        """
        return np.dot(X.T, X) / X.shape[0]

    def sorted_features_by_eigenvalues(self, eigen_values, eigen_vectors, feature_names):
        """
        Función para ordenar las características por valores propios.
        Parámetros:
        eigen_values: matriz numpy de forma (d,), los valores propios
        eigen_vectors: matriz numpy de forma (d, d), los vectores propios
        feature_names: lista de strings, los nombres de las características
        Retorna: sorted_features: lista de strings, los nombres de las características ordenadas por valores propios
        """
        self.features_eigenvalues = dict(zip(self.feature_names, eigen_values))
        # ordenar los nombres de las características por valores propios
        self.features_eigenvalues = dict(sorted(self.features_eigenvalues.items(), key=lambda item: item[1], reverse=True))

        eigen_values = np.sort(eigen_values)[::-1]
        idx = np.argsort(eigen_values)[::-1]
        eigen_vectors = eigen_vectors[:, idx]

        return eigen_values, eigen_vectors

    def compute_projections(self, X):
        """
        Proyectar la matriz de datos X en el espacio PCA.
        Parámetros:
        X: matriz numpy de forma (n, d), la matriz de datos, X está centrada
        eigenvectors: matriz numpy de forma (d, d), la matriz de pesos
        Retorna: X_pca: matriz numpy de forma (n, d), la matriz de datos en el espacio PCA
        """
        return np.dot(X, self.eigenvectors)

    def compute_mean_feature_by_class(self, W, y):
        """
        Computar la media de las características para cada clase.
        Parámetros:
        W: matriz numpy de forma (d, d), la matriz de pesos
        y: matriz numpy de forma (n,), las etiquetas de clase
        Retorna: means: matriz numpy de forma (n_classes, d), la media de las características para cada clase
        """
        W_c = np.zeros((len(np.unique(y)), W.shape[1]))
        for i, c in enumerate(np.unique(y)):
            # Computar el numerador y denominador de la ecuación
            numerator = np.sum(W[np.where(y == c)], axis=0)
            denominator = np.sum(y == c)

            # Computar el peso para la clase c y la característica i
            W_c[i] = numerator / denominator

        return W_c

    def compute_score_feature(self):
        """
        Computar la puntuación de las características.
        Parámetros:
        W_means: matriz numpy de forma (n_classes, d), la media de las características para cada clase
        eigen_values: matriz numpy de forma (d,), los valores propios
        Retorna: scores: matriz numpy de forma (d,), la puntuación de las características
        """
        self.scores = np.zeros(self.W_mean_class.shape[1])
        difference = np.abs(self.W_mean_class[1] - self.W_mean_class[0])

        for i in range(difference.shape[0]):
            if self.eigenvalues[i] == 0:
                self.scores[i] = 0
            else:
                self.scores[i] = difference[i] / self.eigenvalues[i]

    def select_top_features(self):
        """
        Seleccionar las mejores n_components características basadas en la puntuación.
        Retorna: top_features: dict, el diccionario de las mejores n_components características
        """
        items = list(self.features_scores.items())
        self.features_selected = dict(items[:self.n_components])

    def select_features_by_score(self):
        """
        Seleccionar las características basadas en la puntuación.
        Retorna:
            -selected_features: lista de strings, los nombres de las características seleccionadas.
            -S_k: matriz numpy de forma (d, n_components), las características seleccionadas
        """
        # ordenar los nombres de las características por valores propios
        sorted_features_by_eigenvalues = dict(sorted(self.features_eigenvalues.items(), key=lambda item: item[1], reverse=True))

        # asignar las puntuaciones a las características
        items = list(sorted_features_by_eigenvalues.items())
        self.features_scores = {}
        for i in range(len(items)):
            key, value = items[i]
            self.features_scores[key] = self.scores[i]

        # ordenar las características por puntuaciones
        idx = np.argsort(self.scores)[::-1]
        sorted_eigen_vectors = self.eigenvectors[:, idx]

        # seleccionar las primeras n_components columnas de la matriz de vectores propios ordenados según la puntuación
        self.S_k = sorted_eigen_vectors[:, :self.n_components]

        # ordenar los nombres de las características por puntuación
        self.features_scores = dict(sorted(self.features_scores.items(), key=lambda item: item[1], reverse=True))

        return self.features_scores, self.S_k

    def fit(self):
        """
        Método principal para ajustar el modelo PCA.
        Usar los pasos del algoritmo PCA y seleccionar las características basadas en la puntuación
        Retorna: X_projected: matriz numpy de forma (n, n_components), la matriz de datos en el espacio PCA
        """
        # paso 1: matriz de covarianza de los datos dxd
        matrix_covariance = self.compute_covariance_matrix(self.X)

        # paso 2: valores propios y vectores propios
        eig_values, eig_vectors = np.linalg.eig(matrix_covariance)

        # paso 3: ordenar los vectores propios por valores propios decrecientes
        self.eigenvalues, self.eigenvectors = self.sorted_features_by_eigenvalues(eig_values, eig_vectors, self.feature_names)

        # paso 4: proyectar los datos en el espacio PCA
        self.W_pca = self.compute_projections(self.X)

        # paso 5: computar la media de las características para cada clase. paso 2 del algoritmo
        self.W_mean_class = self.compute_mean_feature_by_class(self.W_pca, self.y)

        # paso 6: computar la puntuación de las características. paso 3 del algoritmo
        self.compute_score_feature()

        # paso 7: seleccionar las características basadas en la puntuación. paso 4 del algoritmo
        self.select_features_by_score()
        self.select_top_features()

        # paso 8: retornar los datos proyectados y las características seleccionadas
        X_projected = np.dot(self.X, self.S_k)

        return X_projected

    def transform(self, X):
        """
        Proyectar la matriz de datos X en el espacio PCA.
        Parámetros:
        X: matriz numpy de forma (n, d), la matriz de datos
        Retorna: X_transformed: matriz numpy de forma (n, n_components), la matriz de datos en el espacio PCA
        """
        X_transformed = np.dot(X, self.S_k)
        return X_transformed

    