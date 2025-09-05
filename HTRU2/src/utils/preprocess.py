from imblearn.over_sampling import SMOTE
import pandas as pd
from sklearn.preprocessing import MinMaxScaler


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