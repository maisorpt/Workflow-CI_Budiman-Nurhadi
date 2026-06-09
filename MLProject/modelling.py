import sys
import mlflow
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

if __name__ == "__main__":
    n_estimators = int(sys.argv[1]) if len(sys.argv) > 1 else 100
    max_depth = int(sys.argv[2]) if len(sys.argv) > 2 else 5
    dataset_path = sys.argv[3] if len(sys.argv) > 3 else "btc_preprocessing.csv"


    print(f"Membaca dataset: {dataset_path}...")
    
    # 2. Load Dataset
    data = pd.read_csv(dataset_path)

    if 'Unnamed: 0' in data.columns:
        data = data.sort_values(by='Unnamed: 0').reset_index(drop=True)
        data = data.drop('Unnamed: 0', axis=1)

    kolom_di_drop = ["Target", "Close", "High", "Low", "Open", "Volume"]
    kolom_di_drop = [col for col in kolom_di_drop if col in data.columns]

    X = data.drop(columns=kolom_di_drop)
    y = data["Target"].astype(int)

    # 3. Train-Test Split (Kronologis)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, shuffle=False
    )

    input_example = X_train.iloc[0:5]

    mlflow.sklearn.autolog()
    
    model = RandomForestClassifier(
        n_estimators=n_estimators, 
        max_depth=max_depth,
        min_samples_leaf=5,
        random_state=42
    )
    
    print("Memulai proses training model...")
    model.fit(X_train, y_train)
    
    # Log Model & Artefak
    mlflow.sklearn.log_model(
        sk_model=model,
        artifact_path="model",
        input_example=input_example
    )

    test_accuracy = model.score(X_test, y_test)
    mlflow.log_metric("test_accuracy", test_accuracy)

    print(f"Proses Run MLProject Selesai! Akurasi Akhir: {test_accuracy:.4f}")