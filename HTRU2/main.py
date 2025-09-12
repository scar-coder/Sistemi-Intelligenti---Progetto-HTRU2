# Un file di entry point è il punto di ingresso principale di 
# un programma Python. È lo script che viene eseguito 
# per avviare l’applicazione o eseguire una funzionalità specifica

from src.models import KNN, Decision_Tree, Random_Forest
from src.utils.read import *
from src.utils.preprocess import *
from src.utils.train import *
from src.utils.test import *

from sklearn import tree
import matplotlib.pyplot as plt
import numpy as np

from sklearn.model_selection import StratifiedKFold


import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap


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
    
    
    
    print("\n\n---\n2.3 Feature Selection: RFE (Recursive Feature Elimination)...")
    n_features_to_select = 2
    selected_features_dataset = seleziona_features(dataset_normalizzato, n_features_to_select=n_features_to_select)
    dataset_info(selected_features_dataset)
    
    
    
    
    print("\n\n---\n3 Fase di addestramento del modello...")  
    
    
    print("\n\n---\n3.1 Divisione in training set e test set con Hold-Out...")  
    test_size = 0.3
    features_train, features_test, target_train, target_test = dividi_train_test(selected_features_dataset, test_size)
    print("\nDistribuzione dopo Hold-Out:")
    print(target_train.value_counts())
    print(target_test.value_counts())
    
    print("\n\n---\n3.1 Stratified K-Fold Cross-Validation sul training set...")  
    n_splits = 5
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    
    
    
    print("\n\n---\n3.2 Bilanciamento del training set...")  
    features_train, target_train = bilancia_dataset(features_train, target_train)
    print("\nDistribuzione dopo SMOTE:")
    print(target_train.value_counts())
    
    
    print("\n\n---\n3.3 Ricerca iperparametri Random Forest (Grid Search)...")  
    #best_rf_model = gs_random_forest(features_train, target_train, skf)
    best_rf_model = RandomForestClassifier(n_estimators=20, max_depth=10, random_state=1234) #gs_random_forest(features_train, target_train, skf)
    
    
    print("\n\n---\n3.4 Training Random Forest (Stratified K-Fold)...")
    trained_rf_model, metriche_rf_train = train_model(best_rf_model, features_train, target_train, skf)
    
    print("\n\n---\n4 Test Random Forest...")
    target_pred, target_proba = test_model(trained_rf_model, features_test)
    metriche_rf_test = calcola_metriche(target_test, target_pred, target_proba)
    display_metriche(metriche_rf_test, target_test, target_pred, model_name="Random Forest")
    plot_random_forest(trained_rf_model, features_train.columns, max_depth=3)
    
    

    
    print("\n\n---\n3.3 Ricerca iperparametri Decision Tree (Grid Search)...")  
    best_dt_model = gs_decision_tree(features_train, target_train, skf)
    
    print("\n\n---\n3.4 Training Decision Tree (Stratified K-Fold)...")
    trained_dt_model, metriche_dt_train = train_model(best_dt_model, features_train, target_train, skf)
    
    print("\n\n---\n4 Test Decision Tree...")
    target_pred, target_proba = test_model(trained_dt_model, features_test)
    metriche_dt_test = calcola_metriche(target_test, target_pred, target_proba)
    display_metriche(metriche_dt_test, target_test, target_pred, model_name="Decision Tree")
    plot_decision_tree(trained_dt_model, features_train.columns, max_depth=3)
    
    
    
    
    print("\n\n---\n3.3 Ricerca iperparametri KNN (Grid Search)...")  
    best_knn_model = gs_knn(features_train, target_train, skf)
    
    print("\n\n---\n3.4 Training KNN (Stratified K-Fold)...")
    trained_knn_model, metriche_knn_train = train_model(best_knn_model, features_train, target_train, skf)
    
    print("\n\n---\n4 Test KNN...")
    target_pred, target_proba = test_model(trained_knn_model, features_test)
    metriche_knn_test = calcola_metriche(target_test, target_pred, target_proba)
    display_metriche(metriche_knn_test, target_test, target_pred, model_name="KNN")
    plot_decision_boundary(features_test.values, target_test.values, trained_knn_model, title="KNN Decision Boundary (test set)")
    plot_knn_decision_boundary(trained_knn_model, features_test, target_test, feature_names=features_test.columns[:2])
    
    
    
    print("\n\n---\n3.5 Confronto metriche di valutazione tra i modelli...")
    metriche_modelli_train = {
        "KNN": metriche_knn_train,
        "Decision Tree": metriche_dt_train,
        "Random Forest": metriche_rf_train
    }
    plot_comparazione(metriche_modelli_train)
 
    
    print("\n\n---\n5 Confronto metriche di test tra i modelli...")
    metriche_modelli_test = {
        "KNN": metriche_knn_test,
        "Decision Tree": metriche_dt_test,
        "Random Forest": metriche_rf_test
    }
    
    plot_comparazione(metriche_modelli_test)
    
    

