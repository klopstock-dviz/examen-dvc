import pandas as pd
from sklearn.model_selection import train_test_split
from pathlib import Path

scr_dir = Path(__file__).resolve().parent
data_dir_in=scr_dir.parent/"data"
data_dir_out = scr_dir.parent.parent/"data/processed_data/_split"

def main():
    # load
    df = pd.read_csv(f"{data_dir_in}/raw.csv")
    print(df.shape)
    print(df.head())

    # select target
    X=df.drop(columns=['silica_concentrate', 'date'])
    Y=df['silica_concentrate']

    # split
    X_train, X_test, y_train, y_test=train_test_split(X, Y, test_size=0.2, random_state=123)

    # save
    data_dir_out.mkdir(parents=True, exist_ok=True)
    
    pd.DataFrame(X_train).to_csv(data_dir_out/ "X_train.csv")
    pd.DataFrame(X_test).to_csv(data_dir_out/"X_test.csv")
    pd.DataFrame(y_train).to_csv(data_dir_out/"y_train.csv")
    pd.DataFrame(y_test).to_csv(data_dir_out/"y_test.csv")


if __name__ == "__main__":
    main()