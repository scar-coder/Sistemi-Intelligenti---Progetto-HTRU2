from imblearn.over_sampling import SMOTE
import pandas as pd

def balance_with_smote(dataset, random_state=42):

    X = dataset.drop("Class", axis=1)
    y = dataset["Class"]

    smote = SMOTE(random_state=random_state)
    X_res, y_res = smote.fit_resample(X, y)

    dataset_res = pd.DataFrame(X_res, columns=X.columns)
    dataset_res["Class"] = y_res
    
    return dataset_res