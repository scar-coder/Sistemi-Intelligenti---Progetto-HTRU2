import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
    RocCurveDisplay
)

def test_model(model, features_test, target_test, model_name="Modello"):
    """
    Testa un modello sul dataset di test e stampa/plottare metriche principali:
    Accuracy, Precision, Recall, F1, AUC-ROC, Confusion Matrix.
    """
    # Predizioni
    target_pred = model.predict(features_test)
    target_proba = model.predict_proba(features_test)[:, 1] if hasattr(model, "predict_proba") else None

    # Calcolo metriche
    acc = accuracy_score(target_test, target_pred)
    prec = precision_score(target_test, target_pred)
    rec = recall_score(target_test, target_pred)
    f1 = f1_score(target_test, target_pred)
    auc = roc_auc_score(target_test, target_proba) if target_proba is not None else None

    print(f"\n📊 Risultati {model_name}:")
    print(f"Accuracy : {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall   : {rec:.4f}")
    print(f"F1-score : {f1:.4f}")
    if auc is not None:
        print(f"AUC-ROC  : {auc:.4f}")

    # Confusion Matrix
    cm = confusion_matrix(target_test, target_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot(cmap="Blues")
    plt.title(f"Confusion Matrix - {model_name}")
    plt.show()

    # ROC Curve
    if target_proba is not None:
        RocCurveDisplay.from_predictions(target_test, target_proba)
        plt.title(f"ROC Curve - {model_name}")
        plt.show()

    return {"accuracy": acc, "precision": prec, "recall": rec, "f1": f1, "auc": auc}