from sklearn.preprocessing import StandardScaler
import pandas as pd
from pathlib import Path


scr_dir = Path(__file__).resolve().parent
data_dir_in = scr_dir.parent.parent/"data/processed_data/_split"
data_dir_out = scr_dir.parent.parent/"data/processed_data/_scaling"

def main():
    X_train=pd.read_csv(data_dir_in/"X_train.csv")
    X_test=pd.read_csv(data_dir_in/"X_test.csv")
    
    scaler=StandardScaler()

    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    data_dir_out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(X_train_scaled).to_csv(data_dir_out/"X_train_scaled.csv")
    pd.DataFrame(X_test_scaled).to_csv(data_dir_out/"X_test_scaled.csv")

if __name__=="__main__":
    main()