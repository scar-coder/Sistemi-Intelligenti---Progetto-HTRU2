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


from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

# 3.1 Divisione dei dati (Hold-Out)
def dividi_train_test(dataset, test_size=0.3, random_state=1234):
    features, target = dividi_feature_target(dataset)
    features_train, features_test, target_train, target_test = train_test_split(features, target, test_size=test_size, stratify=target, random_state=random_state)
    return features_train, features_test, target_train, target_test


# 3.2 Funzione generica di grid search con Stratified K-Fold
def grid_search(model, param_grid, X, y, folds=5):
    skf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=1234)
    grid = GridSearchCV(model, param_grid, cv=skf, scoring='accuracy', n_jobs=-1)
    grid.fit(X, y)
    print(f"\nMiglior set di iperparametri: {grid.best_params_}")
    print(f"Miglior accuracy in CV: {grid.best_score_:.4f}")
    return grid.best_estimator_


# 3.3 Training specifico per ogni modello
def train_knn(X_train, y_train):
    param_grid = {
        "n_neighbors": [3, 5, 7, 9, 11],
        "weights": ["uniform", "distance"],
        "metric": ["euclidean", "manhattan"]
    }
    return grid_search(KNeighborsClassifier(), param_grid, X_train, y_train)


def train_decision_tree(X_train, y_train):
    param_grid = {
        "max_depth": [5, 10, 15, None],
        "criterion": ["gini", "entropy"]
    }
    return grid_search(DecisionTreeClassifier(random_state=42), param_grid, X_train, y_train)


def train_random_forest(X_train, y_train):
    param_grid = {
        "n_estimators": [50, 100, 200],
        "max_depth": [5, 10, None],
        "criterion": ["gini", "entropy"]
    }
    return grid_search(RandomForestClassifier(random_state=42, n_jobs=-1), param_grid, X_train, y_train)