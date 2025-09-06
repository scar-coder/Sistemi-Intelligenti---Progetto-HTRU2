from imblearn.over_sampling import SMOTE
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import RFE


def dividi_feature_target(dataset):
    X = dataset.drop("Class", axis=1)
    y = dataset["Class"]
    return X, y


def unisci_feature_target(X, y):
    dataset_unito = pd.DataFrame(X, columns=X.columns)
    dataset_unito["Class"] = y
    return dataset_unito


def bilancia_dataset(dataset, random_state=42):

    features, target = dividi_feature_target(dataset)

    smote = SMOTE(random_state=random_state)
    X_res, y_res = smote.fit_resample(features, target)
    
    dataset_bilanciato = unisci_feature_target(X_res, y_res)
    
    dataset_bilanciato.to_csv('data/interim/balanced_HTRU_2.csv', index=False)
    print("Dataset bilanciato salvato in 'data/interim/balanced_HTRU_2.csv'")
    
    return dataset_bilanciato

def normalizza_features(dataset):
    features, target = dividi_feature_target(dataset)
    
    scaler = MinMaxScaler()
    features_normalizzate = scaler.fit_transform(features)
    
    features_normalizzate = pd.DataFrame(features_normalizzate, columns=features.columns)
    dataset_normalizzato = unisci_feature_target(features_normalizzate, target)
    
    dataset_normalizzato.to_csv('data/interim/normalized_HTRU_2.csv', index=False)
    print("Dataset normalizzato salvato in 'data/interim/normalized_HTRU_2.csv'")
    
    return dataset_normalizzato

def seleziona_feature_knn(dataset, n_features_to_select=5):
    features, target = dividi_feature_target(dataset)
    
    # Usiamo RandomForest come base estimator per RFE
    rf = RandomForestClassifier(random_state=42, n_jobs=-1)
    rfe = RFE(rf, n_features_to_select=n_features_to_select)
    rfe.fit(features, target)
    
    selected_features = features.columns[rfe.support_]
    print(f"Feature selezionate per KNN (con RFE su Random Forest): {list(selected_features)}")
    
    features_selezionate = features[selected_features]
    dataset_selezionato = unisci_feature_target(features_selezionate, target)
    
    dataset_selezionato.to_csv('data/interim/selected_features_HTRU_2.csv', index=False)
    print("Dataset con feature selezionate salvato in 'data/interim/selected_features_HTRU_2.csv'")
    
    return dataset_selezionato
