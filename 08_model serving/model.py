import pandas as pd
import pickle
from sklearn.datasets import load_boston
from sklearn.linear_model import LinearRegression 


def train_model():
    data = load_boston()
    keep_vars = ['CRIM', 'ZN', 'INDUS', 'CHAS']
    X = pd.DataFrame(data["data"], columns=data["feature_names"])[keep_vars]
    y = pd.DataFrame(data["target"], columns=["MEDV"])
    model = LinearRegression()
    model.fit(X, y)

    pkl_filename = "./boston_model.pkl"
    with open(pkl_filename, 'wb') as file:
        pickle.dump(model, file)

    
if __name__ == "__main__":
    train_model()