import numpy as np


class LogisticRegressionMultiClass:
    def __init__(
        self,
        learning_rate=0.001,
        epochs=1000,
        regularization=None,
        reg_strength=0.01,
        beta1=0.9,
        beta2=0.999,
        epsilon=1e-8,
    ):
        """
        Constructor de la clase LogisticRegressionMultiClass.

        Parámetros:
        - learning_rate (float): Tasa de aprendizaje para la optimización.
        - epochs (int): Número máximo de iteraciones de entrenamiento.
        - regularization (str): Tipo de regularización ('l1' o 'l2'). Por defecto, None.
        - reg_strength (float): Fuerza de la regularización. Por defecto, 0.01.
        - beta1 (float): Parámetro de decaimiento para el primer momento en Adam. Por defecto, 0.9.
        - beta2 (float): Parámetro de decaimiento para el segundo momento en Adam. Por defecto, 0.999.
        - epsilon (float): Pequeño valor para evitar divisiones por cero en Adam. Por defecto, 1e-8.
        """
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.regularization = regularization
        self.reg_strength = reg_strength
        self.weights = None  # Matriz de pesos (inicializada en fit)
        self.bias = None  # Vector de sesgos (inicializado en fit)
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon

        # Parámetros para Adam
        self.m_w, self.v_w, self.m_b, self.v_b = None, None, None, None

        # Para registrar las pérdidas durante el entrenamiento y validación
        self.train_losses = []
        self.val_losses = []

    def softmax(self, z):
        """
        Calcula la función softmax para la matriz de entrada z.

        Parámetros:
        - z (ndarray): Matriz de entrada (salida lineal).

        Retorna:
        - ndarray: Probabilidades calculadas con softmax.
        """
        exp_z = np.exp(
            z - np.max(z, axis=1, keepdims=True)
        )  # Evita overflows numéricos
        return exp_z / np.sum(exp_z, axis=1, keepdims=True)

    def fit(self, X, y, X_val=None, y_val=None, patience_level=50):
        """
        Entrena el modelo utilizando el conjunto de datos de entrada.

        Parámetros:
        - X (ndarray): Matriz de características de entrenamiento.
        - y (ndarray): Vector de etiquetas de entrenamiento.
        - X_val (ndarray, opcional): Matriz de características de validación.
        - y_val (ndarray, opcional): Vector de etiquetas de validación.
        - patience_level (int): Cantidad de épocas sin mejora antes de detener el entrenamiento.
        """
        num_samples, num_features = X.shape
        num_classes = np.max(y) + 1

        # Inicialización de pesos y sesgos
        rng = np.random.default_rng(42)
        self.weights = rng.standard_normal((num_features, num_classes)) * 0.01
        self.bias = np.zeros(num_classes)

        # Inicialización de momentos para Adam
        self.m_w = np.zeros_like(self.weights)
        self.v_w = np.zeros_like(self.weights)
        self.m_b = np.zeros_like(self.bias)
        self.v_b = np.zeros_like(self.bias)

        # One-hot encoding de las etiquetas
        y_one_hot = np.eye(num_classes)[y]

        # Variables para early stopping
        best_loss = np.inf
        patience = patience_level
        no_improve_count = 0

        for epoch in range(self.epochs):
            # Cálculo de la salida lineal y las predicciones
            linear_model = np.dot(X, self.weights) + self.bias
            predictions = self.softmax(linear_model)

            # Error entre predicciones y etiquetas reales
            error = predictions - y_one_hot

            # Gradientes de pesos y sesgos
            dw = np.dot(X.T, error) / num_samples
            db = np.sum(error, axis=0) / num_samples

            # Agregar regularización a los gradientes
            if self.regularization == "l2":
                dw += (self.reg_strength / num_samples) * self.weights
            elif self.regularization == "l1":
                dw += (self.reg_strength / num_samples) * np.sign(self.weights)

            # Actualización de parámetros con Adam
            self.m_w = self.beta1 * self.m_w + (1 - self.beta1) * dw
            self.v_w = self.beta2 * self.v_w + (1 - self.beta2) * (dw**2)
            self.m_b = self.beta1 * self.m_b + (1 - self.beta1) * db
            self.v_b = self.beta2 * self.v_b + (1 - self.beta2) * (db**2)

            # Corrección de sesgo en Adam
            m_w_hat = self.m_w / (1 - self.beta1 ** (epoch + 1))
            v_w_hat = self.v_w / (1 - self.beta2 ** (epoch + 1))
            m_b_hat = self.m_b / (1 - self.beta1 ** (epoch + 1))
            v_b_hat = self.v_b / (1 - self.beta2 ** (epoch + 1))

            # Actualización final de pesos y sesgos
            self.weights -= (
                self.learning_rate * m_w_hat / (np.sqrt(v_w_hat) + self.epsilon)
            )
            self.bias -= (
                self.learning_rate * m_b_hat / (np.sqrt(v_b_hat) + self.epsilon)
            )

            # Cálculo de la pérdida de entrenamiento
            loss = -np.mean(
                np.einsum("ij,ij->i", y_one_hot, np.log(predictions + 1e-9))
            )
            if self.regularization == "l2":
                loss += (self.reg_strength / (2 * num_samples)) * np.sum(
                    self.weights**2
                )
            elif self.regularization == "l1":
                loss += (self.reg_strength / num_samples) * np.sum(np.abs(self.weights))
            self.train_losses.append(loss)

            # Cálculo de la pérdida de validación si se proporciona un conjunto de validación
            if X_val is not None and y_val is not None:
                val_predictions = self.predict_proba(X_val)
                val_loss = -np.mean(
                    np.einsum(
                        "ij,ij->i",
                        np.eye(num_classes)[y_val],
                        np.log(val_predictions + 1e-9),
                    )
                )
                if self.regularization == "l2":
                    val_loss += (self.reg_strength / (2 * num_samples)) * np.sum(
                        self.weights**2
                    )
                elif self.regularization == "l1":
                    val_loss += (self.reg_strength / num_samples) * np.sum(
                        np.abs(self.weights)
                    )
                self.val_losses.append(val_loss)

            # Validación temprana (early stopping)
            if loss < best_loss:
                best_loss = loss
                no_improve_count = 0
            else:
                no_improve_count += 1

            if no_improve_count >= patience:
                print(
                    f"Deteniendo el entrenamiento en la época {epoch} por validación temprana."
                )
                break

            # Imprimir progreso cada 10 épocas
            if epoch % 10 == 0:
                print(f"Epoch {epoch}, Loss: {loss:.4f}")

    def predict_proba(self, X):
        """
        Calcula las probabilidades de cada clase para las muestras dadas.

        Parámetros:
        - X (ndarray): Matriz de características.

        Retorna:
        - ndarray: Probabilidades por clase.
        """
        linear_model = np.dot(X, self.weights) + self.bias
        return self.softmax(linear_model)

    def classify(self, X):
        """
        Clasifica las muestras dadas asignando la clase con mayor probabilidad.

        Parámetros:
        - X (ndarray): Matriz de características.

        Retorna:
        - ndarray: Clases predichas para cada muestra.
        """
        probabilities = self.predict_proba(X)
        return np.argmax(probabilities, axis=1)

    def evaluate(self, X, y):
        """
        Evalúa el modelo en un conjunto de datos.

        Parámetros:
        - X (ndarray): Matriz de características.
        - y (ndarray): Etiquetas verdaderas.

        Retorna:
        - float: Precisión del modelo.
        - ndarray: Índices de las muestras mal clasificadas.
        """
        predictions = self.classify(X)
        accuracy = np.mean(predictions == y)
        incorrect_indices = np.where(predictions != y)[0]
        return accuracy, incorrect_indices

    def get_misclassified_elements(self, X, y):
        """
        Devuelve los elementos mal clasificados con información detallada.

        Parámetros:
        - X (ndarray): Matriz de características.
        - y (ndarray): Etiquetas verdaderas.

        Retorna:
        - list[dict]: Lista de elementos mal clasificados con índice, clase predicha y clase real.
        """
        predictions = self.classify(X)
        incorrect_indices = np.where(predictions != y)[0]
        return [
            {"index": i, "predicted_class": predictions[i], "true_class": y[i]}
            for i in incorrect_indices
        ]
