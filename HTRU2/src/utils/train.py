from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, precision_score,auc,roc_curve, recall_score, f1_score, roc_auc_score, confusion_matrix
from src.utils.preprocess import dividi_feature_target, unisci_feature_target
from imblearn.over_sampling import SMOTE
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier


def dividi_train_test(dataset, test_size=0.3):
    features, target = dividi_feature_target(dataset)
    features_train, features_test, target_train, target_test = train_test_split(features, target, test_size=test_size, stratify=target, random_state=1234)
    return features_train, features_test, target_train, target_test

def bilancia_dataset(features, target):
    
    smote = SMOTE(random_state=1234)
    features_res, target_res = smote.fit_resample(features, target)
    
    dataset_bilanciato = unisci_feature_target(features_res, target_res)
    dataset_bilanciato.to_csv('data/interim/balanced_HTRU_2.csv', index=False)
    print("Dataset bilanciato salvato in 'data/interim/balanced_HTRU_2.csv'")
    
    return features_res, target_res

def grid_search(model, param_grid, features, target, skf=StratifiedKFold(n_splits=10, shuffle=True, random_state=1234)):
    grid = GridSearchCV(model, param_grid, cv=skf, scoring='recall')
    grid.fit(features, target)
    print(f"\nMiglior set di iperparametri:\n {grid.best_params_}")
    print(f"Miglior recall in CV: {grid.best_score_:.4f}")
    return grid.best_estimator_




def gs_knn(features_train, y_train, skf):
    param_grid = {
        # basic neighbors / distance
        "n_neighbors": [3, 5, 7, 10, 15, 20, 30, 50, 70, 100],
        "weights": ["uniform", "distance"],
        "metric": ["minkowski", "euclidean", "manhattan", "chebyshev", "cosine"],
        }
    knn = KNeighborsClassifier()
    best_model = grid_search(knn, param_grid, features_train, y_train, skf)
    return best_model


def gs_decision_tree(features_train, y_train, skf):
    param_grid = {
        "criterion": ["gini", "entropy", "log_loss"], 
        "max_depth": [None, 3, 5, 10, 15, 20, 30, 50],
        }
    dt = DecisionTreeClassifier(random_state=1234)
    best_model = grid_search(dt, param_grid, features_train, y_train, skf)
    return best_model


def gs_random_forest(features_train, y_train, skf):
    param_grid = [
        {
            "n_estimators": [50, 100, 200],
            "criterion": ["gini", "entropy"],
            "max_depth": [None, 5, 10, 20, 50]
        }
    ]
    rf = RandomForestClassifier(random_state=1234)
    best_model = grid_search(rf, param_grid, features_train, y_train, skf)
    return best_model




def train_model(model, features, target, skf=StratifiedKFold(n_splits=10, shuffle=True, random_state=1234)):
    
    accuracies, precisions, recalls, f1s, aucs, fprs, fnrs, =[], [], [], [], [], [], []
    
    for fold, (train_idx, val_idx) in enumerate(skf.split(features, target), start=1):
        X_train, X_val = features.iloc[train_idx], features.iloc[val_idx]
        y_train, y_val = target.iloc[train_idx], target.iloc[val_idx]
        
        
        
        model.fit(X_train, y_train)
        
        preds = model.predict(X_val)
        proba = model.predict_proba(X_val)[:, 1] if hasattr(model, "predict_proba") else None
        
        acc = accuracy_score(y_val, preds)
        prec = precision_score(y_val, preds, zero_division=0)
        rec = recall_score(y_val, preds, zero_division=0)
        f1 = f1_score(y_val, preds, zero_division=0)
        fpr, tpr, _ = roc_curve(y_val, proba)
        auc_roc = auc(fpr, tpr)
        
        tn, fp, fn, tp = confusion_matrix(y_val, preds).ravel()
        fpr_val = fp / (fp + tn) if (fp + tn) > 0 else 0.0
        fnr_val = fn / (fn + tp) if (fn + tp) > 0 else 0.0
        
        accuracies.append(acc)
        precisions.append(prec)
        recalls.append(rec)
        f1s.append(f1)
        aucs.append(auc_roc)
        fprs.append(fpr_val)
        fnrs.append(fnr_val)

    metrics = {"accuracy": sum(accuracies)/len(accuracies)
            , "precision": sum(precisions)/len(precisions)
            , "recall": sum(recalls)/len(recalls)
            , "f1": sum(f1s)/len(f1s)
            , "auc_roc": sum(aucs)/len(aucs)
            , "fpr": sum(fprs)/len(fprs)
            , "fnr": sum(fnrs)/len(fnrs)
            }
        
    print(f"\n📊 Risultati medi su {skf.get_n_splits()} fold:")
    print(f"Accuracy media : {metrics['accuracy']:.4f}")
    print(f"Precision media: { metrics['precision']:.4f}")
    print(f"Recall media   : { metrics['recall']:.4f}")
    print(f"F1 media       : { metrics['f1']:.4f}")
    print(f"AUC-ROC media  : { metrics['auc_roc']:.4f}")
    print(f"FPR media      : { metrics['fpr']:.4f}")
    print(f"FNR media      : { metrics['fnr']:.4f}")
    
    
    
    return model, metrics
