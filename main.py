import pandas as pd
from sklearn.model_selection import train_test_split
from model import LogisticRegressionMultiClass
from sklearn.preprocessing import LabelBinarizer
from utils import (
    preprocess_data,
    confusion_matrix,
    classification_report,
    calculate_macro_avg,
    calculate_weighted_avg,
    display_metrics,
    separate_data_labels,
    plot_learning_curve,
    graph_loss,
    plot_confusion_matrix,
    plot_one_vs_rest_roc_curve,
    plot_multiclass_roc_curves,
    plot_predicted_probabilities,
    get_label,
)

# Parámetros globales:
learning_rate = 0.1  # 0.1
epochs = 3500  # 3500
regularization = "l2"  # l2
reg_strength = 0.01  # 0.01
patience = 50  # 50

"""# MAIN - ENTRENAMIENTO"""

if __name__ == "__main__":
    # Se cargan los datos desde el CSV
    file_path = "./Datasets/credit_risk_file_2.csv"
    file_path_second = "./Datasets/credit_risk_file_1.csv"
    data_first = pd.read_csv(file_path)
    data_Second = pd.read_csv(file_path_second)
    data = pd.merge(
        data_first, data_Second, on="PROSPECTID", how="inner", validate="1:1"
    )
    data.set_index("PROSPECTID", inplace=True, drop=False)
    original_ids = data.index.values

    # Se muestran las columnas para verificar su estructura
    print(
        f"Las {len(data.columns)} columnas en el archivo CSV antes del preprocesamiento son:",
        data.columns,
    )

    # Limpiar nombres de columnas por si hay espacios adicionales
    data.columns = data.columns.str.strip()

    # Se separan los datos y las etiquetas:
    X, y = separate_data_labels(data)

    # Se dividen los datos en entrenamiento, validación y prueba
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )
    X_validation, X_test, y_validation, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp
    )

    # Se procesan los datos de entrenamiento, validación y prueba
    try:
        X_train, y_train, X_validation, y_validation, X_test, y_test = preprocess_data(
            X_train,
            y_train,
            X_validation,
            y_validation,
            X_test,
            y_test,
            standarize=True,
            normalize=True,
            stand_type="sklearn",
            norm_type="robust",
        )
    except KeyError as e:
        print(f"Error: {e}")
        print(
            "Revisar la estructura del archivo CSV y confirmar que 'Approved_Flag' no se haya eliminado accidentalmente."
        )
        exit()

    print(f"Tamaño del conjunto de entrenamiento: {len(X_train)}")
    print(f"Tamaño del conjunto de validación: {len(X_validation)}")
    print(f"Tamaño del conjunto de prueba: {len(X_test)}")

    # Se crea y entrena el modelo
    model = LogisticRegressionMultiClass(
        learning_rate, epochs, regularization, reg_strength
    )
    model.fit(
        X_train.values,
        y_train.values,
        X_validation.values,
        y_validation.values,
        patience_level=patience,
    )

    # Se imprimen las columnas que finalmente pasaron al modelo:
    print(
        f"\nLas {len(X_train.columns)} columnas que pasaron al modelo después de ser procesadas son:\n"
    )
    print(X_train.columns)

    """# MÉTRICAS """
    # Etiquetas de las clases:
    class_labels = ["P1", "P2", "P3", "P4"]

    # Se evalúa el modelo para los conjunetos de entrenamiento, validación y prueba
    train_accuracy, train_errors = model.evaluate(X_train.values, y_train.values)
    validation_accuracy, validation_errors = model.evaluate(
        X_validation.values, y_validation.values
    )
    test_accuracy, test_errors = model.evaluate(X_test.values, y_test.values)

    # Se generan las predicciones de las clasificaciones y métricas del entrenamiento
    y_train_pred = model.classify(X_train.values)
    conf_matrix = confusion_matrix(y_train.values, y_train_pred, class_labels)
    class_report = classification_report(y_train.values, y_train_pred, class_labels)

    # Se calculan los promedios del entrenamiento
    macro_avg = calculate_macro_avg(class_report)
    weighted_avg = calculate_weighted_avg(class_report)

    print("\n\nEntrenamiento", end="\n\n")

    # Gráfica de la matriz de confusión del entrenamiento
    plot_confusion_matrix(conf_matrix, class_labels, dataset_title="Entrenamiento")
    # Se muestran las métricas con formato
    display_metrics(
        conf_matrix=conf_matrix,
        class_report=class_report,
        macro_avg=macro_avg,
        weighted_avg=weighted_avg,
        train_accuracy=train_accuracy,
    )

    print("\n\n")
    print("Validación", end="\n\n")

    # Se generan las predicciones de las clasificaciones y métricas de la validación
    y_validation_pred = model.classify(X_validation.values)
    conf_matrix = confusion_matrix(y_validation.values, y_validation_pred, class_labels)
    class_report = classification_report(
        y_validation.values, y_validation_pred, class_labels
    )

    # Se calculan los promedios de la validación
    macro_avg = calculate_macro_avg(class_report)
    weighted_avg = calculate_weighted_avg(class_report)

    # Gráfica de la matriz de confusión del validación
    plot_confusion_matrix(conf_matrix, class_labels, dataset_title="Validación")
    # Se muestran las métricas con formato
    display_metrics(
        conf_matrix=conf_matrix,
        class_report=class_report,
        macro_avg=macro_avg,
        weighted_avg=weighted_avg,
        validation_accuracy=validation_accuracy,
    )

    # Se generan las predicciones de las clasificaciones y métricas de validación
    y_test_pred = model.classify(X_test.values)
    conf_matrix = confusion_matrix(y_test.values, y_test_pred, class_labels)
    class_report = classification_report(y_test.values, y_test_pred, class_labels)

    # Se calculan los promedios de validación
    macro_avg = calculate_macro_avg(class_report)
    weighted_avg = calculate_weighted_avg(class_report)

    """# CONJUNTO DE PRUEBA"""
    print("\n\nPrueba", end="\n\n")

    # Gráfica de la matriz de confusión de la prueba
    plot_confusion_matrix(conf_matrix, class_labels, dataset_title="Prueba")
    # Se muestran las métricas con formato
    display_metrics(
        conf_matrix=conf_matrix,
        class_report=class_report,
        macro_avg=macro_avg,
        weighted_avg=weighted_avg,
        test_accuracy=test_accuracy,
    )

    # Gráfica de la curva de aprendizaje (entrenamiento vs. validación)
    plot_learning_curve(model.train_losses, model.val_losses)
    # Gráfica del costo/pérdida del conjunto de datos de entrenamiento:
    graph_loss(model.train_losses, dataset_title="Entrenamiento")
    # Gráfica del costo/pérdida del conjunto de datos de validación:
    graph_loss(model.val_losses, dataset_title="Validación", plot_color="orange")

    # Probabilidades predichas de las clases del conjunto de prueba
    y_prob_test = model.predict_proba(X_test)

    # Gráfica de distribución de probabilidades predichas del conjunto de prueba
    plot_predicted_probabilities(y_prob_test, class_labels)

    # Gráfica de curvas ROC One-vs-Rest para cada una de las clases específicas del conjunto de prueba
    label_binarizer = LabelBinarizer().fit(y_train)
    y_onehot_test = label_binarizer.transform(y_test)
    # Clase P1
    class_of_interest = 1
    plot_one_vs_rest_roc_curve(
        y_onehot_test,
        y_prob_test,
        class_id=class_of_interest,
        class_label=get_label(class_of_interest),
        plot_color="blue",
    )
    # Clase P2
    class_of_interest = 0
    plot_one_vs_rest_roc_curve(
        y_onehot_test,
        y_prob_test,
        class_id=class_of_interest,
        class_label=get_label(class_of_interest),
        plot_color="darkorange",
    )
    # Clase P3
    class_of_interest = 2
    plot_one_vs_rest_roc_curve(
        y_onehot_test,
        y_prob_test,
        class_id=class_of_interest,
        class_label=get_label(class_of_interest),
        plot_color="green",
    )
    # Clase P4
    class_of_interest = 3
    plot_one_vs_rest_roc_curve(
        y_onehot_test,
        y_prob_test,
        class_id=class_of_interest,
        class_label=get_label(class_of_interest),
        plot_color="red",
    )

    # Graficar curvas ROC multiclase para el conjunto de prueba
    roc_auc_mean = plot_multiclass_roc_curves(y_test, y_prob_test)
    print(f"\n\nEl AUC promedio es: {roc_auc_mean:.2f}")

    # Elementos mal clasificados del conjunto de prueba con sus valores reales del dataset original
    test_errors = model.get_misclassified_elements(X_test.values, y_test.values)

    print(
        f"\n\nSe encontraron un total de {len(test_errors)} elementos mal clasificados"
    )

    # Lista para almacenar los datos de los elementos mal clasificados
    misclassified_data = []

    for element in test_errors:
        index = element["index"]
        predicted_class = get_label(element["predicted_class"])
        true_class = get_label(element["true_class"])
        true_index = X_test.iloc[index].name

        misclassified_entry = {
            "TRUE_CLASS": true_class,
            "PREDICTED_CLASS": predicted_class,
        }
        # Añadir las características originales del dataset (hace -1 porque es el número de fila del CSV)
        # de tal forma que el PROSCPECTID corresponda con su valor correcto:
        misclassified_entry.update(data.iloc[true_index - 1].to_dict())

        misclassified_data.append(misclassified_entry)

    # Crear un DataFrame de pandas con los datos
    misclassified_df = pd.DataFrame(misclassified_data)

    # Guardar el DataFrame en un archivo CSV
    misclassified_df.to_csv("./Datasets/Elementos_Mal_Clasificados.csv", index=False)

    print(
        "Los elementos mal clasificados se han guardado en './Datasets/Elementos_Mal_Clasificados.csv'\n\n"
    )
