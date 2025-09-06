# Un file di entry point è il punto di ingresso principale di 
# un programma Python. È lo script che viene eseguito 
# per avviare l’applicazione o eseguire una funzionalità specifica

from src.models import KNN, Decision_Tree, Random_Forest
from src.utils.read import *
from src.utils.preprocess import *
from src.utils.train import *
from src.utils.test import *


if __name__ == "__main__":
    print ("1. Fase di caricamento detaset...")
    
    dataset_grezzo = load_dataset()
    dataset_info(dataset_grezzo)
    
    print("2. Fase di preprocessing del dataset...")
    
    print("\n\n---\n2.1 Normalizzazione delle features...")  
    dataset_normalizzato = normalizza_features(dataset_grezzo)
    dataset_info(dataset_normalizzato)
    
    print("\n\n---\n2.2 Bilanciamento del dataset...")  
    balanced_dataset = bilancia_dataset(dataset_normalizzato, random_state=42)
    print("\nDistribuzione dopo SMOTE:")
    print(balanced_dataset["Class"].value_counts())
    dataset_info(balanced_dataset)
    
    
    print("\n\n---\n2.3 Feature Selection per il classificatore KNN...")
    selected_features_dataset = seleziona_feature_knn(balanced_dataset, n_features_to_select=5)
    dataset_info(selected_features_dataset)
    
    print("\n\n---\n3 Fase di addestramento del modello...")  
    
    
    print("\n\n---\n3.1 Divisione dei dati con Hold-Out...")  
    features_train, features_test, target_train, target_test = dividi_train_test(selected_features_dataset)
    
    print("\n\n---\n3.1.1 Stratified K-Fold Cross-Validation + Grid Search...")

    print("\n\n---\n3.3.1 Training classificatore KNN...")
    knn_model = train_knn(features_train, target_train)

    print("\n\n---\n3.3.2 Training Decision Tree...")
    dt_model = train_decision_tree(features_train, target_train)

    print("\n\n---\n3.3.3 Training Random Forest...")
    rf_model = train_random_forest(features_train, target_train)
    print("\n\n---\n3.1.1 Stratified K-Fold Cross-Validation...")  
    
    
    
    print("\n\n---\n3.2 Ricerca iperparametri Grid Search...")  
    
    print("\n\n---\n3.2.1 Grid Search...")  
    
    
    
    print("\n\n---\n3.3 Training dei modelli di classificazione...")  
    
    
    print("\n\n---\n3.3.1 Training classificatore KNN...")  
    
    print("\n\n---\n3.3.2 Training Decision Tree...")  
    
    print("\n\n---\n3.3.3 Training Random Forest...")  
    
    
    
    print("\n\n---\n4 Test del modello...")
    
    print("\n\n---\n4.1 Training classificatore KNN...")  
    
    print("\n\n---\n4.2 Training Decision Tree...")  
    
    print("\n\n---\n4.3 Training Random Forest...")  
    
    

    
    
    