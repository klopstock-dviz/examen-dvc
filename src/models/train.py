import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.svm import SVR, LinearSVR
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error, mean_absolute_percentage_error
from pathlib import Path
import pickle
import joblib

scr_dir = Path(__file__).resolve().parent

data_dir_X = scr_dir.parent.parent/"data/processed_data/_scaling"
data_dir_y = scr_dir.parent.parent/"data/processed_data/_split"
models_dir_out = scr_dir.parent.parent/"models/best_models"
def main():

    X_train_scaled=pd.read_csv(data_dir_X / "X_train_scaled.csv")    
    y_train=pd.read_csv(data_dir_y / "y_train.csv")
    

    rfr=RandomForestRegressor()

    with open(scr_dir.parent.parent/ "models/best_params/best_params.pkl", "rb") as f:
        loaded_params = pickle.load(f)


    # 4. train du modèle avec les best params
    rfr=RandomForestRegressor(**loaded_params)

    rfr.fit(X_train_scaled, y_train)    

    models_dir_out.mkdir(parents=True, exist_ok=True)
    joblib.dump(rfr, models_dir_out / "rfr_model.joblib")


if __name__ =="__main__":
    main()