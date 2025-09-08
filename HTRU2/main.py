# Un file di entry point è il punto di ingresso principale di 
# un programma Python. È lo script che viene eseguito 
# per avviare l’applicazione o eseguire una funzionalità specifica

from src.models import KNN, Decision_Tree, Random_Forest
from src.utils.read import *
from src.utils.preprocess import *
from src.utils.train import *
from src.utils.test import *


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
    
    
    print("\n\n---\n2.3 Feature Selection per il classificatore KNN...")
    selected_features_dataset = seleziona_features(balanced_dataset, n_features_to_select=5)
    dataset_info(selected_features_dataset)
    
    print("\n\n---\n3 Fase di addestramento del modello...")  
    
    
    print("\n\n---\n3.1 Divisione dei dati con Hold-Out...")  
    #features_train, features_test, target_train, target_test = dividi_train_test(dataset_grezzo)
    #features_train, features_test, target_train, target_test = dividi_train_test(dataset_normalizzato)
    #features_train, features_test, target_train, target_test = dividi_train_test(balanced_dataset)
    features_train, features_test, target_train, target_test = dividi_train_test(selected_features_dataset)
    
    print("\n\n---\n3.1.1 Stratified K-Fold Cross-Validation...")  
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state)
    
    for fold, (train_index, val_index) in enumerate(skf.split(features_train, target_train)):
        print(f"\nFold {fold + 1}")
        X_train_fold, X_val_fold = features_train.iloc[train_index], features_train.iloc[val_index]
        y_train_fold, y_val_fold = target_train.iloc[train_index], target_train.iloc[val_index]
        print(f"Training set size: {X_train_fold.shape[0]} samples")
        print(f"Validation set size: {X_val_fold.shape[0]} samples")
        print(f"Class distribution in training set:\n{y_train_fold.value_counts()}")
        print(f"Class distribution in validation set:\n{y_val_fold.value_counts()}")
    
    print("\n\n---\n3.2 Ricerca iperparametri...")  
    
    print("\n\n---\n3.2.1 Grid Search...")  
    best_knn_model = gs_knn(features_train, target_train)
    best_dt_model = gs_decision_tree(features_train, target_train)
    best_rf_model = gs_random_forest(features_train, target_train)
    
    
    
    print("\n\n---\n3.3 Training dei modelli di classificazione...")  
    
    
    print("\n\n---\n3.3.1 Training classificatore KNN...")  
    
    print("\n\n---\n3.3.2 Training Decision Tree...")  
    
    print("\n\n---\n3.3.3 Training Random Forest...")  
    
    
    
    print("\n\n---\n4 Test del modello...")
    
    print("\n\n---\n4.1 Training classificatore KNN...")  
    
    print("\n\n---\n4.2 Training Decision Tree...")  
    
    print("\n\n---\n4.3 Training Random Forest...")  
    
    

    
    
    