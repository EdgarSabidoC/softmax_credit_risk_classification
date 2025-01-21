import pandas as pd
import numpy as np
from sklearn.preprocessing import (
    StandardScaler,
    MinMaxScaler,
    RobustScaler,
    Normalizer,
    PowerTransformer,
)


# Gráficas
# ==============================================================================
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import roc_curve, auc, RocCurveDisplay
from sklearn.preprocessing import label_binarize


original_columns = [
    "time_since_recent_payment",
    "num_times_delinquent",
    "max_delinquency_level",
    "max_recent_level_of_deliq",
    "num_deliq_6mts",
    "num_deliq_12mts",
    "num_deliq_6_12mts",
    "max_deliq_6mts",
    "max_deliq_12mts",
    "num_times_30p_dpd",
    "num_times_60p_dpd",
    "num_std",
    "num_std_6mts",
    "num_std_12mts",
    "num_sub",
    "num_sub_6mts",
    "num_sub_12mts",
    "num_dbt",
    "num_dbt_6mts",
    "num_dbt_12mts",
    "num_lss",
    "num_lss_6mts",
    "num_lss_12mts",
    "recent_level_of_deliq",
    "tot_enq",
    "CC_enq",
    "CC_enq_L6m",
    "CC_enq_L12m",
    "PL_enq",
    "PL_enq_L6m",
    "PL_enq_L12m",
    "time_since_recent_enq",
    "enq_L12m",
    "enq_L6m",
    "enq_L3m",
    "MARITALSTATUS",
    "EDUCATION",
    "AGE",
    "GENDER",
    "NETMONTHLYINCOME",
    "Time_With_Curr_Empr",
    "pct_of_active_TLs_ever",
    "pct_opened_TLs_L6m_of_L12m",
    "pct_currentBal_all_TL",
    "CC_utilization",
    "CC_Flag",
    "PL_utilization",
    "PL_Flag",
    "pct_PL_enq_L6m_of_L12m",
    "pct_CC_enq_L6m_of_L12m",
    "pct_PL_enq_L6m_of_ever",
    "pct_CC_enq_L6m_of_ever",
    "max_unsec_exposure_inPct",
    "HL_Flag",
    "GL_Flag",
    "last_prod_enq2",
    "first_prod_enq2",
    "Credit_Score",
    "Approved_Flag",
    "Total_TL",
    "Tot_Closed_TL",
    "Tot_Active_TL",
    "Total_TL_opened_L6M",
    "Tot_TL_closed_L6M",
    "pct_tl_open_L6M",
    "pct_tl_closed_L6M",
    "pct_active_tl",
    "pct_closed_tl",
    "Total_TL_opened_L12M",
    "Tot_TL_closed_L12M",
    "pct_tl_open_L12M",
    "pct_tl_closed_L12M",
    "Tot_Missed_Pmnt",
    "Auto_TL",
    "CC_TL",
    "Consumer_TL",
    "Gold_TL",
    "Home_TL",
    "PL_TL",
    "Secured_TL",
    "Unsecured_TL",
    "Other_TL",
    "Age_Oldest_TL",
    "Age_Newest_TL",
]


def get_label(class_int):
    """Función que mapea un entero a su correspondiente clase"""
    class_labels = {"0": "P2", "1": "P1", "2": "P3", "3": "P4"}
    return class_labels[str(class_int)]


def evaluate_metrics(model, X, y):
    """Evalúa el modelo y devuelve precisión, recall y F1-score por clase."""
    y_pred = model.predict(X)
    classes = np.unique(y)

    metrics = {}
    for cls in classes:
        tp = np.sum((y == cls) & (y_pred == cls))
        fp = np.sum((y != cls) & (y_pred == cls))
        fn = np.sum((y == cls) & (y_pred != cls))
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = (
            2 * (precision * recall) / (precision + recall)
            if (precision + recall) > 0
            else 0
        )
        support = np.sum(y == cls)
        metrics[cls] = {
            "precision": precision,
            "recall": recall,
            "f1-score": f1,
            "support": support,
        }

    # Promedios macro y ponderados
    total_support = sum(m["support"] for m in metrics.values())
    macro_avg = {
        k: np.mean([m[k] for m in metrics.values()])
        for k in ["precision", "recall", "f1-score"]
    }
    weighted_avg = {
        k: sum(m[k] * m["support"] for m in metrics.values()) / total_support
        for k in ["precision", "recall", "f1-score"]
    }

    return metrics, macro_avg, weighted_avg


def confusion_matrix(y_true, y_pred, class_labels=None):
    """
    Calcula la matriz de confusión con etiquetas personalizadas.

    Args:
        y_true: Valores verdaderos de las clases.
        y_pred: Valores predichos de las clases.
        class_labels: Lista de etiquetas personalizadas. Debe estar en el mismo orden que las clases.

    Returns:
        pd.DataFrame: Matriz de confusión con etiquetas personalizadas P1, P2, P3 y P4.
    """
    # Identificar las clases únicas
    classes = np.unique(np.concatenate((y_true, y_pred)))

    # Crear la matriz de confusión
    matrix = np.zeros((len(classes), len(classes)), dtype=int)
    for i, true_class in enumerate(classes):
        for j, pred_class in enumerate(classes):
            matrix[i, j] = np.sum((y_true == true_class) & (y_pred == pred_class))

    df = pd.DataFrame(matrix, index=classes, columns=classes)

    # Se intercambian las filas y columnas específicas (0 y 1)
    if class_labels is not None and len(class_labels) > 1:
        row_order = df.index.tolist()
        col_order = df.columns.tolist()

        # Se intercambia 0 con 1
        row_order[0], row_order[1] = row_order[1], row_order[0]
        col_order[0], col_order[1] = col_order[1], col_order[0]

        # Se reordena el DataFrame
        df = df.loc[row_order, col_order]

        # Se cambia el nombre de las filas y columnas por el de las clases personalizadas
        df.index = class_labels
        df.columns = class_labels
    return df


def classification_report(y_true, y_pred, class_labels=None):
    """Genera un informe de clasificación con métricas comunes."""
    classes = np.unique(y_true)
    report = {}
    for cls in classes:
        tp = np.sum((y_true == cls) & (y_pred == cls))
        fp = np.sum((y_true != cls) & (y_pred == cls))
        fn = np.sum((y_true == cls) & (y_pred != cls))
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = (
            2 * (precision * recall) / (precision + recall)
            if (precision + recall) > 0
            else 0
        )
        support = np.sum(y_true == cls)
        report[cls] = {
            "precision": precision,
            "recall": recall,
            "f1-score": f1,
            "support": support,
        }
    df = pd.DataFrame(report).T  # Se convierte a dataFrame para mejorar la presentación
    if class_labels != None:
        df.index = (
            class_labels  # Se cambian los índices por las etiquetas personalizadas
        )
        df.iloc[0], df.iloc[1] = df.iloc[1], df.iloc[0].copy()
    return df


def calculate_macro_avg(report):
    """Calcula los promedios macro de precisión, recall y F1-score."""
    macro_avg = report[["precision", "recall", "f1-score"]].mean()
    macro_avg["support"] = report["support"].sum()
    return macro_avg


def calculate_weighted_avg(report):
    """Calcula los promedios ponderados de precisión, recall y F1-score."""
    total_support = report["support"].sum()
    weighted_avg = (
        (report[["precision", "recall", "f1-score"]].T * report["support"]).T.sum()
    ) / total_support
    weighted_avg["support"] = total_support
    return weighted_avg


def display_metrics(
    conf_matrix,
    class_report,
    macro_avg,
    weighted_avg,
    train_accuracy=None,
    validation_accuracy=None,
    test_accuracy=None,
):
    """Imprime las métricas en formato tabular."""
    print("\n|----Matriz de Confusión----|")
    print(conf_matrix)
    print("\n|----Informe de Clasificación----|")
    print(class_report.to_string())
    if train_accuracy:
        print(f"\nTrain Accuracy: {train_accuracy:.4%}")
    if validation_accuracy:
        print(f"\nValidation Accuracy: {validation_accuracy:.4%}")
    if test_accuracy:
        print(f"\nTest Accuracy: {test_accuracy:.4%}")
    print("\n|----Promedios----|")
    print(
        f"Macro Avg    -> Precision: {macro_avg['precision']:.4f}, Recall: {macro_avg['recall']:.4f}, F1-score: {macro_avg['f1-score']:.4f}"
    )
    print(
        f"Weighted Avg -> Precision: {weighted_avg['precision']:.4f}, Recall: {weighted_avg['recall']:.4f}, F1-score: {weighted_avg['f1-score']:.4f}"
    )


def graph_loss(losses, dataset_title="Entrenamiento", plot_color="blue"):
    """
    Grafica la función de costo a lo largo de las épocas.

    Parámetros:
        losses (list or np.ndarray): Lista o arreglo de valores de la función de costo por época.
    """
    plt.figure(figsize=(10, 6))
    plt.plot(losses, label="Pérdida (Loss)", color=plot_color, linewidth=2)
    plt.xlabel("Época")
    plt.ylabel("Pérdida")
    plt.title(f"Evolución de la Pérdida durante {dataset_title}")
    plt.legend()
    plt.grid(True)
    plt.show()


def plot_learning_curve(train_losses, val_losses):
    """
    Grafica la curva de aprendizaje para entrenamiento y validación.

    Parámetros:
    train_losses (list): Lista con los valores de la pérdida en el conjunto de entrenamiento por cada época.
    val_losses (list): Lista con los valores de la pérdida en el conjunto de validación por cada época.
    """
    plt.figure(figsize=(10, 6))
    plt.plot(range(len(train_losses)), train_losses, label="Pérdida del Entrenamiento")
    plt.plot(range(len(val_losses)), val_losses, label="Pérdida de la Validación")
    plt.xlabel("Época")
    plt.ylabel("Pérdida")
    plt.title("Curva de Aprendizaje: Entrenamiento vs Validación")
    plt.legend()
    plt.grid(True)
    plt.show()


def plot_confusion_matrix(conf_matrix, class_labels, dataset_title=None):
    """
    Grafica la matriz de confusión.
    """
    plt.figure(figsize=(8, 6))
    sns.heatmap(
        conf_matrix,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=class_labels,
        yticklabels=class_labels,
    )
    if dataset_title:
        plt.title(f"Matriz de Confusión de {dataset_title}")
    else:
        plt.title("Matriz de Confusión")
    plt.ylabel("Clases verdaderas")
    plt.xlabel("Clases predichas")
    plt.show()


def plot_predicted_probabilities(y_prob, class_labels):
    """
    Grafica la distribución de probabilidades predichas para cada clase.
    """
    probs = pd.DataFrame(y_prob, columns=class_labels)
    probs.plot(kind="density", figsize=(10, 6), alpha=0.7)
    plt.title("Distribución de Probabilidades Predichas por Clase")
    plt.xlabel("Probabilidad")
    plt.ylabel("Densidad")
    plt.grid()
    plt.legend(class_labels)
    plt.show()


def plot_one_vs_rest_roc_curve(
    y_true_onehot, y_prob, class_id, class_label, plot_color="darkorange"
):
    """
    Grafica la curva ROC para un caso One-vs-Rest.
    """
    display = RocCurveDisplay.from_predictions(
        y_true_onehot[:, class_id],
        y_prob[:, class_id],
        name=f"{class_label} contra todos",
        color=plot_color,
        plot_chance_level=True,
    )
    display.ax_.set(
        xlabel="Tasa de Falsos Positivos",
        ylabel="Tasa de Verdaderos Positivos",
        title=f"Curva ROC Uno contra Todos de {class_label}",
    )
    display.ax_.grid(True)  # Agrega la cuadrícula
    plt.show()


def plot_multiclass_roc_curves(y_true, y_prob, class_labels=[0, 1, 2, 3]):
    """
    Calcula y grafica las curvas ROC y el AUC para un caso multiclase.
    """
    y_true_binarized = label_binarize(y_true, classes=class_labels)

    fpr = {}
    tpr = {}
    roc_auc = {}
    roc_auc_mean = 0.0

    # Listas para almacenar todos los FPR y TPR para promediar después
    all_fpr = np.linspace(0, 1, 100)
    mean_tpr = np.zeros_like(all_fpr)

    for i, class_name in enumerate(class_labels):
        if np.sum(y_true_binarized[:, i]) > 0:  # Verificar si hay muestras positivas
            fpr[i], tpr[i], _ = roc_curve(y_true_binarized[:, i], y_prob[:, i])
            roc_auc[i] = auc(fpr[i], tpr[i])
            roc_auc_mean += roc_auc[i]

            # Interpolar tpr para cada clase
            mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])
        else:
            print(
                f"Clase {class_name} no tiene muestras positivas en el conjunto de prueba y será omitida."
            )

    # Se calcula el AUC promedio:
    roc_auc_mean /= len(roc_auc)
    mean_tpr /= len(roc_auc)

    # Grafica las curvas ROC para cada clase
    plt.figure(figsize=(10, 8))

    plt.plot(fpr[1], tpr[1], color="blue", label=f"Clase P{1} (AUC = {roc_auc[1]:.2f})")
    plt.plot(
        fpr[0], tpr[0], color="darkorange", label=f"Clase P{2} (AUC = {roc_auc[0]:.2f})"
    )
    plt.plot(
        fpr[2], tpr[2], color="green", label=f"Clase P{3} (AUC = {roc_auc[2]:.2f})"
    )
    plt.plot(fpr[3], tpr[3], color="red", label=f"Clase P{4} (AUC = {roc_auc[3]:.2f})")

    # Agregar curva ROC promedio
    plt.plot(
        all_fpr,
        mean_tpr,
        linestyle="--",
        lw=2,
        color="purple",
        label=f"AUC promedio = {roc_auc_mean:.2f}",
    )

    plt.plot([0, 1], [0, 1], "k--", lw=2)
    plt.title("Curvas ROC Multiclase")
    plt.xlabel("Tasa de Falsos Positivos")
    plt.ylabel("Tasa de Verdaderos Positivos")
    plt.legend(loc="lower right")
    plt.grid()
    plt.show()

    return roc_auc_mean  # Se retorna el AUC promedio


def preprocess_data(
    X_train,
    y_train,
    X_val,
    y_val,
    X_test,
    y_test,
    standarize=True,
    normalize=True,
    stand_type="custom",
    norm_type="robust",
):
    """Limpieza y preprocesamiento del dataset."""

    # Función auxiliar para el preprocesamiento
    def preprocess(X, fit_scalers=False):
        # 1. Elimina las columnas que no estén en la lista de columnas originales
        columns_to_drop = [col for col in X.columns if col not in original_columns]
        X.drop(columns=columns_to_drop, inplace=True)

        # 2. Reemplaza valores faltantes (-99999) con NaN
        X.replace(-99999, np.nan, inplace=True)

        # 3. Elimina columnas con más del threshold% de valores NaN
        threshold = 0.0
        columns_to_drop = X.columns[X.isnull().mean() > threshold]
        if len(columns_to_drop) > 0:
            print(f"Eliminando columnas con demasiados NaN: {list(columns_to_drop)}")
        X.drop(columns=columns_to_drop, inplace=True)

        # 4. Imputación de valores faltantes con la media para columnas numéricas
        num_cols = X.select_dtypes(include=["float64", "int64"]).columns
        X[num_cols] = X[num_cols].fillna(X[num_cols].mean())

        # 5. Elimina columnas con baja varianza
        low_variance_cols = X[num_cols].std()[X[num_cols].std() < 1e-6].index
        if len(low_variance_cols) > 0:
            print(f"Eliminando columnas con baja varianza: {list(low_variance_cols)}")
            X.drop(columns=low_variance_cols, inplace=True)

        # 6. Convierte las variables categóricas en dummies
        cat_cols = X.select_dtypes(include=["object"]).columns
        X = pd.get_dummies(X, columns=cat_cols, drop_first=True)

        return X

    # Se preprocesan los conjuntos de entrenamiento, validación y pruebas por separado
    X_train = preprocess(X_train, fit_scalers=True)
    X_val = preprocess(X_val)
    X_test = preprocess(X_test)

    # Se identifican columnas numéricas originales (antes de generar dummies)
    num_cols = X_train.select_dtypes(include=["float64", "int64"]).columns
    # Se identifican columnas dummy (todas las demás)
    dummy_cols = [col for col in X_train.columns if col not in num_cols]

    # Se definen y ajustan escaladores con el conjunto de entrenamiento
    if standarize:
        if stand_type == "custom":
            mean = X_train[num_cols].mean()
            std = X_train[num_cols].std()
            X_train[num_cols] = (X_train[num_cols] - mean) / std
            X_val[num_cols] = (X_val[num_cols] - mean) / std
            X_test[num_cols] = (X_test[num_cols] - mean) / std
        elif standarize and stand_type == "sklearn":
            scaler = StandardScaler()
            X_train[num_cols] = scaler.fit_transform(X_train[num_cols])
            X_val[num_cols] = scaler.transform(X_val[num_cols])
            X_test[num_cols] = scaler.transform(X_test[num_cols])

    # Normalización (opcional)
    if normalize:
        if norm_type == "min-max":
            scaler = MinMaxScaler(feature_range=(0, 1))
            X_train[num_cols] = scaler.fit_transform(X_train[num_cols])
            X_val[num_cols] = scaler.transform(X_val[num_cols])
            X_test[num_cols] = scaler.transform(X_test[num_cols])
        elif norm_type == "robust":
            scaler = RobustScaler()
            X_train[num_cols] = scaler.fit_transform(X_train[num_cols])
            X_val[num_cols] = scaler.transform(X_val[num_cols])
            X_test[num_cols] = scaler.transform(X_test[num_cols])
        elif norm_type == "normalizer":
            normalizer = Normalizer(norm="l2")
            X_train[num_cols] = normalizer.fit_transform(X_train[num_cols])
            X_val[num_cols] = normalizer.transform(X_val[num_cols])
            X_test[num_cols] = normalizer.transform(X_test[num_cols])
        elif norm_type == "box-cox":
            power_transformer = PowerTransformer(method="box-cox")
            X_train[num_cols] = power_transformer.fit_transform(X_train[num_cols])
            X_val[num_cols] = power_transformer.transform(X_val[num_cols])
            X_test[num_cols] = power_transformer.transform(X_test[num_cols])
        elif norm_type == "log":
            X_train[num_cols] = np.log1p(X_train[num_cols])
            X_val[num_cols] = np.log1p(X_val[num_cols])
            X_test[num_cols] = np.log1p(X_test[num_cols])
        elif norm_type == "decimal-scaling":
            max_abs_value = np.abs(X_train[num_cols]).max()
            scaling_factor = np.ceil(np.log10(max_abs_value))
            X_train[num_cols] = X_train[num_cols] / (10**scaling_factor)
            X_val[num_cols] = X_val[num_cols] / (10**scaling_factor)
            X_test[num_cols] = X_test[num_cols] / (10**scaling_factor)
        elif norm_type == "percentiles":
            percentiles = np.percentile(X_train[num_cols], [25, 75], axis=0)
            Q1, Q3 = percentiles[0], percentiles[1]
            IQR = Q3 - Q1
            X_train[num_cols] = (X_train[num_cols] - Q1) / IQR
            X_val[num_cols] = (X_val[num_cols] - Q1) / IQR
            X_test[num_cols] = (X_test[num_cols] - Q1) / IQR

    # Se vuelven a unir las columnas numéricas y las dummy variables
    X_train = pd.concat([X_train[num_cols], X_train[dummy_cols]], axis=1)
    X_val = pd.concat([X_val[num_cols], X_val[dummy_cols]], axis=1)
    X_test = pd.concat([X_test[num_cols], X_test[dummy_cols]], axis=1)

    # Se asegura que X e y sean del tipo adecuado
    X_train = X_train.astype(float)
    X_val = X_val.astype(float)
    X_test = X_test.astype(float)
    y_train = y_train.astype(int)
    y_val = y_val.astype(int)
    y_test = y_test.astype(int)

    return X_train, y_train, X_val, y_val, X_test, y_test


def separate_data_labels(df):
    # 1. Convierte las etiquetas de 'Approved_Flag' a valores numéricos
    if "Approved_Flag" in df.columns:
        unique_labels = df["Approved_Flag"].unique()
        label_mapping = {label: idx for idx, label in enumerate(unique_labels)}
        print("Mapping de etiquetas para Approved_Flag:", label_mapping)
        df["Approved_Flag"] = df["Approved_Flag"].map(label_mapping)
    else:
        raise KeyError("La columna 'Approved_Flag' no se encuentra en los datos.")

    # 2. Separa las características (X) y etiquetas (y)
    X = df.drop(columns=["Approved_Flag"])
    y = df["Approved_Flag"]

    return X, y
