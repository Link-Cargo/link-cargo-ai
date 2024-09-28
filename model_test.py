import pickle
import numpy as np
import pandas as pd

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

def predict_fare(fairway, d1, d2, d3, d4, d5, m) :
    datas = np.array([[d1, d2, d3, d4, d5, m]], dtype=float)
    
    with open('models.pkl', 'rb') as f:
        loaded_models = pickle.load(f)
    
    # print(loaded_models["북미"])
    loaded_models = loaded_models[fairway]

    scaler = loaded_models["Scaler"]
    pca = loaded_models["PCA"]

    datas = scaler.transform(datas)
    
    month = datas[:, -1]
    shipping_features = datas[:, :-1]
    X_pca = pca.transform(shipping_features)
    combined_features = np.column_stack((X_pca, month))
    
    pred = loaded_models["Model"].predict(combined_features)
    error = loaded_models["Error"]
    
    return (pred, error)

    
if __name__ == "__main__" :
    pred = predict_fare("북미", 24381.880, 93.600, 4363951,	862964,	25.90793, 9)
    print(pred)