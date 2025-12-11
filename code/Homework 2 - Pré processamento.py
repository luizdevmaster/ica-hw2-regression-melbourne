# Pré-Processamento

import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# 1) Caminho para arquivo original
DATA_PATH = Path(r"C:\Users\augus\Documents\ICA - HOMEWORK 1\Data-Melbourne_F.csv")

# 2) Caminho para salvar os arquivos limpos
OUTPUT_DIR = Path(r"C:\Users\augus\Documents\ICA - HOMEWORK 1\outputs_hw2")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def load_and_clean():
    df = pd.read_csv(DATA_PATH)

    # Mantém apenas as colunas que vamos usar
    cols = [
        "avg_outflow", "avg_inflow", "total_grid", "Am", "BOD", "COD", "TN",
        "T", "TM", "Tm", "SLP", "H", "PP", "VV", "V", "VM", "VG",
        "year", "month", "day"
    ]
    df = df[cols]

    # Remove duplicatas exatas
    df = df.drop_duplicates()

    # Remove linhas com NaN em qualquer coluna relevante
    df = df.dropna(subset=cols)

    # Remove consumos não positivos
    df = df[df["total_grid"] > 0]

    # (Opcional) cortar outliers extremos de total_grid (1% e 99%)
    q_low = df["total_grid"].quantile(0.01)
    q_high = df["total_grid"].quantile(0.99)
    df = df[(df["total_grid"] >= q_low) & (df["total_grid"] <= q_high)]

    print("Após limpeza básica:", df.shape)
    return df


def log_transform(X, feature_names, cols_to_log):
    X_trans = X.copy()
    for col in cols_to_log:
        idx = feature_names.index(col)
        X_trans[:, idx] = np.log1p(np.maximum(X_trans[:, idx], 0.0))
    return X_trans


def main():
    df = load_and_clean()

    feature_cols = [
        "avg_outflow", "avg_inflow", "Am", "BOD", "COD", "TN",
        "T", "TM", "Tm", "SLP", "H", "PP", "VV", "V", "VM", "VG",
        "year", "month", "day"
    ]
    target_col = "total_grid"

    X = df[feature_cols].values
    y = df[target_col].values

    # Split treino/teste (75/25)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42
    )
    print("Shape treino:", X_train.shape, "Shape teste:", X_test.shape)

    # Log transform em PP (pode adicionar mais colunas se necessário)
    cols_to_log = ["PP"]
    X_train_log = log_transform(X_train, feature_cols, cols_to_log)
    X_test_log = log_transform(X_test, feature_cols, cols_to_log)

    # Padronização
    scaler = StandardScaler()
    X_train_prep = scaler.fit_transform(X_train_log)
    X_test_prep = scaler.transform(X_test_log)

    # Salva datasets limpos
    # 1) DataFrame completo limpo
    df.to_csv(OUTPUT_DIR / "Data-Melbourne_F_clean.csv", index=False)

    # 2) Train/test pré-processados
    train_df = pd.DataFrame(X_train_prep, columns=feature_cols)
    train_df[target_col] = y_train
    train_df.to_csv(OUTPUT_DIR / "train_preprocessed.csv", index=False)

    test_df = pd.DataFrame(X_test_prep, columns=feature_cols)
    test_df[target_col] = y_test
    test_df.to_csv(OUTPUT_DIR / "test_preprocessed.csv", index=False)

    print("Arquivos salvos em:", OUTPUT_DIR)


if __name__ == "__main__":
    main()
