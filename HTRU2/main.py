# Un file di entry point è il punto di ingresso principale di 
# un programma Python. È lo script che viene eseguito 
# per avviare l’applicazione o eseguire una funzionalità specifica

import multiprocessing
multiprocessing.freeze_support()  # evita errori multiprocess su Windows

from src.models import KNN, Decision_Tree, Random_Forest
from src.utils.read import *
from src.utils.preprocess import *
from src.utils.train import *
from src.utils.test import *


    
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score

def train_with_stratified_kfold(model, features, target, n_splits=5):
    """
    Addestra un modello usando Stratified K-Fold Cross Validation
    e stampa le accuracy ottenute nei vari fold.
    """
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    
    accuracies = []
    precisions = []
    recalls = []
    f1s = []
    aucs = []
    
    fold = 1
    for train_idx, val_idx in skf.split(features, target):
        features_train, features_val = features.iloc[train_idx], features.iloc[val_idx]
        target_train, target_val = target.iloc[train_idx], target.iloc[val_idx]
        
        model.fit(features_train, target_train)
        preds = model.predict(features_val)
        
        proba = model.predict_proba(features_val)[:, 1]
        acc = accuracy_score(target_val, preds)
        prec = precision_score(target_val, preds, zero_division=0)
        rec = recall_score(target_val, preds, zero_division=0)
        f1 = f1_score(target_val, preds, zero_division=0)
        auc_val = roc_auc_score(target_val, proba)

        # append alle liste
        accuracies.append(acc)
        precisions.append(prec)
        recalls.append(rec)
        f1s.append(f1)
        aucs.append(auc_val)
        
        fold += 1

    # calcola e stampa le medie (uso sum/len come richiesto)
    print(f"\nAccuracy media sui {n_splits} fold: {sum(accuracies)/len(accuracies):.4f}")
    print(f"Precision media sui {n_splits} fold: {sum(precisions)/len(precisions):.4f}")
    print(f"Recall media sui {n_splits} fold: {sum(recalls)/len(recalls):.4f}")
    print(f"F1 media sui {n_splits} fold: {sum(f1s)/len(f1s):.4f}")
    print(f"AUC media sui {n_splits} fold: {sum(aucs)/len(aucs):.4f}")
    
    return model

if __name__ == "__main__":
    
    set_random_state(969902)
    print(f"Assegnazione seed per riproducibilità:{random_state}")
    
    print ("1. Fase di caricamento detaset...")
    dataset_grezzo = load_dataset()
    dataset_info(dataset_grezzo)
    
    print("2. Fase di preprocessing del dataset...")
    
    print("\n\n---\n2.1 Normalizzazione delle features...")  
    dataset_normalizzato = normalizza_features(dataset_grezzo)
    dataset_info(dataset_normalizzato)
    
    print("\n\n---\n2.2 Bilanciamento del dataset...")  
    balanced_dataset = bilancia_dataset(dataset_normalizzato)
    print("\nDistribuzione dopo SMOTE:")
    print(balanced_dataset["Class"].value_counts())
    dataset_info(balanced_dataset)
    
    print("\n\n---\n2.3 Feature Selection: RFE (Recursive Feature Elimination)...")
    n_features_to_select = 8
    selected_features_dataset = seleziona_features(balanced_dataset, n_features_to_select=n_features_to_select)
    dataset_info(selected_features_dataset)
    
    
    print("\n\n---\n3 Fase di addestramento del modello...")  
    
    print("\n\n---\n3.1 Divisione dei dati con Hold-Out...")  
    test_size = 0.2
    #features_train, features_test, target_train, target_test = dividi_train_test(dataset_grezzo, test_size)
    #features_train, features_test, target_train, target_test = dividi_train_test(dataset_normalizzato, test_size)
    #features_train, features_test, target_train, target_test = dividi_train_test(balanced_dataset, test_size)
    features_train, features_test, target_train, target_test = dividi_train_test(selected_features_dataset, test_size)
    
    print("\n\n---\n3.1.1 Stratified K-Fold Cross-Validation...")  
    n_splits = 6
    print("\n\n---\n3.2 Ricerca iperparametri...")  
    
    print("\n\n---\n3.2.1 Grid Search...")  
    best_knn_model = gs_knn(features_train, target_train)
    best_dt_model = gs_decision_tree(features_train, target_train)
    best_rf_model = gs_random_forest(features_train, target_train)
    
    
    
    print("\n\n---\n3.3 Training dei modelli di classificazione...")  
    
    print("\n--- Training KNN con Stratified K-Fold ---")
    train_with_stratified_kfold(best_knn_model, features_train, target_train)

    print("\n--- Training Decision Tree con Stratified K-Fold ---")
    train_with_stratified_kfold(best_dt_model, features_train, target_train)

    print("\n--- Training Random Forest con Stratified K-Fold ---")
    train_with_stratified_kfold(best_rf_model, features_train, target_train)
    
    print("\n\n---\n3.3.1 Training classificatore KNN...")  
    
    print("\n\n---\n3.3.2 Training Decision Tree...")  
    
    print("\n\n---\n3.3.3 Training Random Forest...")  
    
    
    
    print("\n\n---\n4 Test del modello...")
    print("\n--- Test KNN ---")
    results_knn = test_model(best_knn_model, features_test, target_test, model_name="KNN")

    print("\n--- Test Decision Tree ---")
    results_dt = test_model(best_dt_model, features_test, target_test, model_name="Decision Tree")

    print("\n--- Test Random Forest ---")
    results_rf = test_model(best_rf_model, features_test, target_test, model_name="Random Forest")
    print("\n\n---\n4.1 Training classificatore KNN...")  
    
    print("\n\n---\n4.2 Training Decision Tree...")  
    
    print("\n\n---\n4.3 Training Random Forest...")  
    
    

    