from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    roc_curve, 
    auc,
    ConfusionMatrixDisplay,
)
import matplotlib.pyplot as plt
import numpy as np
from sklearn.tree import plot_tree

from matplotlib.colors import ListedColormap


def test_model(model, features_test):
    target_pred = model.predict(features_test)
    target_proba = model.predict_proba(features_test)[:, 1]
    return target_pred, target_proba
    

def calcola_metriche(target_test, target_pred, target_proba):
    acc = accuracy_score(target_test, target_pred)
    prec = precision_score(target_test, target_pred)
    rec = recall_score(target_test, target_pred)
    f1 = f1_score(target_test, target_pred)
    fpr, tpr, _ = roc_curve(target_test, target_proba)
    auc_roc = auc(fpr, tpr)
    tn, fp, fn, tp = confusion_matrix(target_test, target_pred).ravel()
    fpr_val = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    fnr_val = fn / (fn + tp) if (fn + tp) > 0 else 0.0
        
    return {
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1": f1,
        "auc_roc": auc_roc,
        "fpr_roc":fpr,
        "tpr_roc":tpr,
        "fpr": fpr_val,
        "fnr": fnr_val
    }


def display_metriche(metriche, target_test, target_pred, model_name="Modello"):
    
    acc = metriche["accuracy"]
    prec = metriche["precision"]
    rec = metriche["recall"]
    f1 = metriche["f1"]
    auc_roc = metriche["auc_roc"]
    fpr = metriche["fpr"]
    fnr = metriche["fnr"]
    fpr_roc = metriche["fpr_roc"]
    tpr_roc = metriche["tpr_roc"]
    
    print(f"\n---\nRisultati {model_name}:")
    print(f"Accuracy : {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall   : {rec:.4f}")
    print(f"F1-score : {f1:.4f}")
    print(f"AUC-ROC  : {auc_roc:.4f}")
    print(f"FPR      : {fpr:.4f}")
    print(f"FNR      : {fnr:.4f}")
    
    # Plot ROC curve
    plt.plot(fpr_roc, tpr_roc, color='purple', lw=2,
            label='ROC curve (area = %0.2f)' % auc_roc)
    plt.plot(["Non Pulsar", "Pulsar"], ["Non Pulsar", "Pulsar"], color='purple', lw=2, linestyle='--')
    plt.title(f"ROC Curve - {model_name}", color="purple")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.legend(loc="lower right")
    plt.show()
    
    # Confusion Matrix
    cm = confusion_matrix(target_test, target_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot(cmap="Purples")
    plt.title(f"Confusion Matrix - {model_name}", color="purple")
    plt.show()
    plt.close()


def plot_comparazione(results_dict):
    metrics = ["accuracy", "precision", "recall", "f1", "auc_roc", "fpr", "fnr"]
    models = list(results_dict.keys())

    #Creiamo una matrice dei valori
    values = [[results_dict[m][metric] if results_dict[m][metric] is not None else 0 for metric in metrics] for m in models]

    x = np.arange(len(metrics))  # posizioni delle metriche
    width = 0.25  # larghezza delle barre

    fig, ax = plt.subplots(figsize=(10,6))

    for i, model in enumerate(models):
        ax.bar(x + i*width, values[i], width, label=model)

    ax.set_xticks(x + width)
    ax.set_xticklabels([m.capitalize() for m in metrics])
    ax.set_ylim(0,1.1)
    ax.set_ylabel("Score")
    ax.set_title("Confronto delle metriche sui modelli")
    ax.legend()
    plt.show()


def plot_decision_tree(dt_model, feature_names, max_depth=3):
    plt.figure(figsize=(50,40))
    plot_tree(dt_model,
              feature_names=feature_names,
              class_names=["Non Pulsar", "Pulsar"],
              filled=True, 
              rounded=True,
              fontsize=6,
              max_depth=max_depth,
              impurity=True)
    plt.title("Esempio di albero Decision Tree sul dataset HTRU2 (tagliato a profondità 3).\nOgni nodo rappresenta una regola basata su una feature, con le foglie che indicano la classe predetta (Pulsar o Non Pulsar).", fontsize=13)
    plt.show() 

def plot_random_forest(rf_model, feature_names, max_depth=3):
    estimator = rf_model.estimators_
    plt.figure(figsize=(30,40))
    plot_tree(estimator[0],
              feature_names=feature_names,
              class_names=["Non Pulsar", "Pulsar"],
              filled=True, 
              rounded=True,
              fontsize=6,
              max_depth=max_depth,
              impurity=True)
    plt.title(f"Esempio di un albero (profondità max {max_depth}) dalla Random Forest", fontsize=13)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()

def plot_decision_boundary(X, y, model, title):
    """
    Funzione per visualizzare il confine di decisione di un classificatore.
    X: le feature (2 colonne)
    y: il target
    model: il modello addestrato
    title: il titolo del grafico
    """
    # Passo della mesh
    h = .02
    cmap_light = ListedColormap(['#FFAAAA', '#AAFFAA'])
    cmap_bold = ListedColormap(['#FF0000', '#00FF00'])

    # Crea un reticolo (mesh) per il grafico
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    
    # Previsione su ogni punto del reticolo
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    
    # Plot del reticolo e dei punti di training
    plt.figure(figsize=(10, 7))
    plt.pcolormesh(xx, yy, Z, cmap=cmap_light, shading='auto')
    
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap=cmap_bold, edgecolor='k', s=20)
    plt.xlim(xx.min(), xx.max())
    plt.ylim(yy.min(), yy.max())
    plt.title(title)
    plt.xlabel('Profile_std')
    plt.ylabel('Profile_kurtosis')
    plt.show()

def plot_knn_decision_boundary(model, X, y, feature_names):
    X = X.iloc[:, :2].values  # prime due feature
    h = .02  
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    
    plt.figure(figsize=(10, 6))
    plt.contourf(xx, yy, Z, alpha=0.4)
    plt.scatter(X[:, 0], X[:, 1], c=y, edgecolor='k')
    plt.xlabel(feature_names[0])
    plt.ylabel(feature_names[1])
    plt.title("Decision Boundary KNN")
    plt.show()
