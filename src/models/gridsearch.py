import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
from pathlib import Path
import pickle

scr_dir = Path(__file__).resolve().parent

data_dir_X = scr_dir.parent.parent/"data/processed_data/_scaling"
data_dir_y = scr_dir.parent.parent/"data/processed_data/_split"
data_dir_out = scr_dir.parent.parent/"models/best_params"


def main():

    X_train_scaled=pd.read_csv(data_dir_X / "X_train_scaled.csv")    
    y_train=pd.read_csv(data_dir_y / "y_train.csv")
    

    rfr=RandomForestRegressor()

    param_grid = {
        'n_estimators': [50, 100, 200],
        'max_depth': [None, 5, 10],
        'min_samples_split': [2, 5, 10]
    }

    grid_search = GridSearchCV(
        estimator=rfr,
        param_grid=param_grid,
        scoring='neg_mean_absolute_error',
        cv=3,
        n_jobs=-1,
        verbose=0,    
    )

    grid_search.fit(X_train_scaled, y_train)
    
    
    data_dir_out.mkdir(parents=True, exist_ok=True)
    with open(data_dir_out/"best_params.pkl", "wb") as f:
        pickle.dump(grid_search.best_params_, f)

if __name__=="__main__":
    main()