import pandas as pd
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error, mean_absolute_percentage_error
import json
from pathlib import Path
import joblib

scr_dir = Path(__file__).resolve().parent

data_dir_X = scr_dir.parent.parent/"data/processed_data/_scaling"
data_dir_y = scr_dir.parent.parent/"data/processed_data/_split"
data_dir_out = scr_dir.parent.parent/"metrics"



def main():
    X_test_scaled=pd.read_csv(data_dir_X/"X_test_scaled.csv")
    y_test=pd.read_csv(data_dir_y/"y_test.csv")

    rfr=joblib.load(scr_dir.parent.parent/"models/best_models/rfr_model.joblib")
        
    # 5. evaluation du modèle
    data=rfr.predict(X_test_scaled)

    scores=[
        {"best_r2": r2_score(y_test, data)},
        {"best_mse": mean_squared_error(y_test, data)},
        {"best_mae": mean_absolute_error(y_test, data)},
    ]

    data_dir_out.mkdir(parents=True, exist_ok=True)
    p=data_dir_out/ "scores.json"
    with open(p, 'w') as f:        
        json.dump(scores, f, indent=1)

    pd.DataFrame(data).to_csv(data_dir_out/"data.csv")


if __name__=="__main__":
    main()