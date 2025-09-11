import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import RFE
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier


global random_state
random_state=1234

def set_random_state(seed=random_state):
    global random_state
    random_state = seed

def dividi_feature_target(dataset):
    features = dataset.drop("Class", axis=1)
    target = dataset["Class"]
    return features, target

def unisci_feature_target(features, target):
    dataset_unito = pd.DataFrame(features, columns=features.columns)
    dataset_unito["Class"] = target
    return dataset_unito


# 2 Feature engineering

def normalizza_features(dataset):
    features, target = dividi_feature_target(dataset)
    
    scaler = MinMaxScaler()
    features_normalizzate = scaler.fit_transform(features)
    
    features_normalizzate = pd.DataFrame(features_normalizzate, columns=features.columns)
    dataset_normalizzato = unisci_feature_target(features_normalizzate, target)
    
    dataset_normalizzato.to_csv('data/interim/normalized_HTRU_2.csv', index=False)
    print("Dataset normalizzato salvato in 'data/interim/normalized_HTRU_2.csv'")
    
    return dataset_normalizzato

def seleziona_features(dataset, n_features_to_select=8):
    features, target = dividi_feature_target(dataset)
    
    # Usiamo RandomForest come base estimator per RFE
    rf = RandomForestClassifier(random_state=random_state, n_jobs=-1)
    rfe = RFE(rf, n_features_to_select=n_features_to_select)
    rfe.fit(features, target)
    
    selected_features = features.columns[rfe.support_]
    print(f"Feature selezionate per KNN (con RFE su Random Forest): {list(selected_features)}")
    
    features_selezionate = features[selected_features]
    dataset_selezionato = unisci_feature_target(features_selezionate, target)
    
    dataset_selezionato.to_csv('data/interim/selected_features_HTRU_2.csv', index=False)
    print("Dataset con feature selezionate salvato in 'data/interim/selected_features_HTRU_2.csv'")
    
    return dataset_selezionato



# Training

# 3.2 Grid search
