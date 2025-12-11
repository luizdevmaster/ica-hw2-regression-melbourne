# hw2_models_melbourne.py

import numpy as np
import pandas as pd
from pathlib import Path

from sklearn.model_selection import KFold
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.decomposition import PCA
from sklearn.cross_decomposition import PLSRegression
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error, r2_score

# -------------------------------------------------------------------
# Caminhos
# -------------------------------------------------------------------

OUTPUT_DIR = Path(r"C:\Users\augus\Documents\ICA - HOMEWORK 1\outputs_hw2")

TRAIN_PATH = OUTPUT_DIR / "train_preprocessed.csv"
TEST_PATH  = OUTPUT_DIR / "test_preprocessed.csv"


# -------------------------------------------------------------------
# Utilitários
# -------------------------------------------------------------------

def rmse(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))

def r2(y_true, y_pred):
    return r2_score(y_true, y_pred)


# -------------------------------------------------------------------
# Carregar dados já pré-processados
# -------------------------------------------------------------------

def load_preprocessed():
    train_df = pd.read_csv(TRAIN_PATH)
    test_df  = pd.read_csv(TEST_PATH)

    target_col = "total_grid"
    feature_cols = [c for c in train_df.columns if c != target_col]

    X_train = train_df[feature_cols].values
    y_train = train_df[target_col].values

    X_test = test_df[feature_cols].values
    y_test = test_df[target_col].values

    return X_train, y_train, X_test, y_test, feature_cols


# -------------------------------------------------------------------
# OLS manual (e ridge via solução fechada)
# -------------------------------------------------------------------

def fit_ols_manual(X, y, l2_reg=0.0):
    X_ext = np.c_[np.ones(X.shape[0]), X]
    n_features = X_ext.shape[1]

    I = np.eye(n_features)
    I[0, 0] = 0.0  # não penaliza intercepto

    A = X_ext.T @ X_ext + l2_reg * I
    b = X_ext.T @ y
    beta = np.linalg.solve(A, b)
    return beta

def predict_ols_manual(X, beta):
    X_ext = np.c_[np.ones(X.shape[0]), X]
    return X_ext @ beta


# -------------------------------------------------------------------
# K-fold CV manual genérico
# -------------------------------------------------------------------

def kfold_cv_manual(model_fit_fn, model_pred_fn, X, y, k=5, **fit_kwargs):
    kf = KFold(n_splits=k, shuffle=True, random_state=42)
    rmses, r2s = [], []

    for tr_idx, val_idx in kf.split(X):
        X_tr, X_val = X[tr_idx], X[val_idx]
        y_tr, y_val = y[tr_idx], y[val_idx]

        params = model_fit_fn(X_tr, y_tr, **fit_kwargs)
        y_pred = model_pred_fn(X_val, params)

        rmses.append(rmse(y_val, y_pred))
        r2s.append(r2(y_val, y_pred))

    return np.mean(rmses), np.mean(r2s), rmses, r2s


# -------------------------------------------------------------------
# OLS: resultados e comparação
# -------------------------------------------------------------------

def run_ols(X_train, y_train, X_test, y_test):
    # Manual
    beta = fit_ols_manual(X_train, y_train, l2_reg=0.0)
    y_pred_test_manual = predict_ols_manual(X_test, beta)
    rmse_test_manual = rmse(y_test, y_pred_test_manual)
    r2_test_manual = r2(y_test, y_pred_test_manual)

    # CV manual
    rmse_cv_manual, r2_cv_manual, _, _ = kfold_cv_manual(
        fit_ols_manual, predict_ols_manual, X_train, y_train, k=5, l2_reg=0.0
    )

    # scikit-learn
    lr = LinearRegression()
    lr.fit(X_train, y_train)
    y_pred_test_sklearn = lr.predict(X_test)
    rmse_test_sklearn = rmse(y_test, y_pred_test_sklearn)
    r2_test_sklearn = r2(y_test, y_pred_test_sklearn)

    return {
        "manual": {
            "beta": beta,
            "rmse_test": rmse_test_manual,
            "r2_test": r2_test_manual,
            "rmse_cv": rmse_cv_manual,
            "r2_cv": r2_cv_manual,
        },
        "sklearn": {
            "coef": lr.coef_,
            "intercept": lr.intercept_,
            "rmse_test": rmse_test_sklearn,
            "r2_test": r2_test_sklearn,
        },
    }


# -------------------------------------------------------------------
# Ridge (L2) via solução fechada + comparação sklearn
# -------------------------------------------------------------------

def run_ridge(X_train, y_train, X_test, y_test, lambdas=None, k=5):
    if lambdas is None:
        lambdas = np.logspace(-4, 3, 10)

    results = []
    for lam in lambdas:
        rmse_cv, r2_cv, _, _ = kfold_cv_manual(
            fit_ols_manual, predict_ols_manual, X_train, y_train, k=k, l2_reg=lam
        )
        results.append((lam, rmse_cv, r2_cv))

    best_idx = np.argmin([r[1] for r in results])
    best_lambda, best_rmse_cv, best_r2_cv = results[best_idx]

    # ajuste final manual
    beta_best = fit_ols_manual(X_train, y_train, l2_reg=best_lambda)
    y_pred_test_manual = predict_ols_manual(X_test, beta_best)
    rmse_test_manual = rmse(y_test, y_pred_test_manual)
    r2_test_manual = r2(y_test, y_pred_test_manual)

    # sklearn
    ridge = Ridge(alpha=best_lambda)
    ridge.fit(X_train, y_train)
    y_pred_test_sklearn = ridge.predict(X_test)
    rmse_test_sklearn = rmse(y_test, y_pred_test_sklearn)
    r2_test_sklearn = r2(y_test, y_pred_test_sklearn)

    return {
        "cv_profile": results,
        "best_lambda": best_lambda,
        "manual": {
            "beta": beta_best,
            "rmse_cv": best_rmse_cv,
            "r2_cv": best_r2_cv,
            "rmse_test": rmse_test_manual,
            "r2_test": r2_test_manual,
        },
        "sklearn": {
            "coef": ridge.coef_,
            "intercept": ridge.intercept_,
            "rmse_test": rmse_test_sklearn,
            "r2_test": r2_test_sklearn,
        },
    }


# -------------------------------------------------------------------
# PCR
# -------------------------------------------------------------------

def run_pcr(X_train, y_train, X_test, y_test, max_components=None, k=5):
    n_features = X_train.shape[1]
    if max_components is None:
        max_components = min(10, n_features)

    rmse_cv_list = []
    r2_cv_list = []
    components_list = list(range(1, max_components + 1))

    for n_comp in components_list:
        kf = KFold(n_splits=k, shuffle=True, random_state=42)
        rmses_fold, r2s_fold = [], []

        for tr_idx, val_idx in kf.split(X_train):
            X_tr, X_val = X_train[tr_idx], X_train[val_idx]
            y_tr, y_val = y_train[tr_idx], y_train[val_idx]

            pca = PCA(n_components=n_comp)
            X_tr_pca = pca.fit_transform(X_tr)
            X_val_pca = pca.transform(X_val)

            beta = fit_ols_manual(X_tr_pca, y_tr)
            y_pred_val = predict_ols_manual(X_val_pca, beta)

            rmses_fold.append(rmse(y_val, y_pred_val))
            r2s_fold.append(r2(y_val, y_pred_val))

        rmse_cv_list.append(np.mean(rmses_fold))
        r2_cv_list.append(np.mean(r2s_fold))

    best_idx = int(np.argmin(rmse_cv_list))
    best_n_comp = components_list[best_idx]

    pca_final = PCA(n_components=best_n_comp)
    X_train_pca = pca_final.fit_transform(X_train)
    X_test_pca = pca_final.transform(X_test)

    beta_final = fit_ols_manual(X_train_pca, y_train)
    y_pred_test = predict_ols_manual(X_test_pca, beta_final)

    rmse_test = rmse(y_test, y_pred_test)
    r2_test = r2(y_test, y_pred_test)

    return {
        "cv_profile": {
            "n_components": components_list,
            "rmse_cv": rmse_cv_list,
            "r2_cv": r2_cv_list,
        },
        "best_n_components": best_n_comp,
        "test": {
            "rmse_test": rmse_test,
            "r2_test": r2_test,
        },
    }


# -------------------------------------------------------------------
# PLS
# -------------------------------------------------------------------

def run_pls(X_train, y_train, X_test, y_test, max_components=None, k=5):
    n_features = X_train.shape[1]
    if max_components is None:
        max_components = min(10, n_features)

    rmse_cv_list = []
    r2_cv_list = []
    components_list = list(range(1, max_components + 1))

    for n_comp in components_list:
        kf = KFold(n_splits=k, shuffle=True, random_state=42)
        rmses_fold, r2s_fold = [], []

        for tr_idx, val_idx in kf.split(X_train):
            X_tr, X_val = X_train[tr_idx], X_train[val_idx]
            y_tr, y_val = y_train[tr_idx], y_train[val_idx]

            pls = PLSRegression(n_components=n_comp)
            pls.fit(X_tr, y_tr)
            y_pred_val = pls.predict(X_val).ravel()

            rmses_fold.append(rmse(y_val, y_pred_val))
            r2s_fold.append(r2(y_val, y_pred_val))

        rmse_cv_list.append(np.mean(rmses_fold))
        r2_cv_list.append(np.mean(r2s_fold))

    best_idx = int(np.argmin(rmse_cv_list))
    best_n_comp = components_list[best_idx]

    pls_final = PLSRegression(n_components=best_n_comp)
    pls_final.fit(X_train, y_train)
    y_pred_test = pls_final.predict(X_test).ravel()

    rmse_test = rmse(y_test, y_pred_test)
    r2_test = r2(y_test, y_pred_test)

    return {
        "cv_profile": {
            "n_components": components_list,
            "rmse_cv": rmse_cv_list,
            "r2_cv": r2_cv_list,
        },
        "best_n_components": best_n_comp,
        "test": {
            "rmse_test": rmse_test,
            "r2_test": r2_test,
        },
    }


# -------------------------------------------------------------------
# MLP
# -------------------------------------------------------------------

def run_mlp(X_train, y_train, X_test, y_test):
    mlp = MLPRegressor(
        hidden_layer_sizes=(32, 16),
        activation="relu",
        solver="adam",
        alpha=1e-3,
        max_iter=2000,
        random_state=42,
    )
    mlp.fit(X_train, y_train)
    y_pred_test = mlp.predict(X_test)

    return {
        "rmse_test": rmse(y_test, y_pred_test),
        "r2_test": r2(y_test, y_pred_test),
    }


# -------------------------------------------------------------------
# Salvar resumo em CSV
# -------------------------------------------------------------------

def save_results_to_csv(ols_results, ridge_results, pcr_results, pls_results, mlp_results, path):
    rows = []

    rows.append({
        "model": "OLS_manual",
        "rmse_test": ols_results["manual"]["rmse_test"],
        "r2_test": ols_results["manual"]["r2_test"],
    })
    rows.append({
        "model": "OLS_sklearn",
        "rmse_test": ols_results["sklearn"]["rmse_test"],
        "r2_test": ols_results["sklearn"]["r2_test"],
    })
    rows.append({
        "model": "Ridge_manual",
        "rmse_test": ridge_results["manual"]["rmse_test"],
        "r2_test": ridge_results["manual"]["r2_test"],
    })
    rows.append({
        "model": "Ridge_sklearn",
        "rmse_test": ridge_results["sklearn"]["rmse_test"],
        "r2_test": ridge_results["sklearn"]["r2_test"],
    })
    rows.append({
        "model": "PCR",
        "rmse_test": pcr_results["test"]["rmse_test"],
        "r2_test": pcr_results["test"]["r2_test"],
    })
    rows.append({
        "model": "PLS",
        "rmse_test": pls_results["test"]["rmse_test"],
        "r2_test": pls_results["test"]["r2_test"],
    })
    rows.append({
        "model": "MLP",
        "rmse_test": mlp_results["rmse_test"],
        "r2_test": mlp_results["r2_test"],
    })

    df = pd.DataFrame(rows)
    df.to_csv(path, index=False)
    print("Resultados salvos em:", path)


# -------------------------------------------------------------------
# Main
# -------------------------------------------------------------------

def main():
    X_train, y_train, X_test, y_test, feature_names = load_preprocessed()
    print("Treino:", X_train.shape, "Teste:", X_test.shape)

    # OLS
    ols_results = run_ols(X_train, y_train, X_test, y_test)
    print("\n=== OLS ===")
    print("Manual - RMSE:", ols_results["manual"]["rmse_test"], "R2:", ols_results["manual"]["r2_test"])
    print("Sklearn - RMSE:", ols_results["sklearn"]["rmse_test"], "R2:", ols_results["sklearn"]["r2_test"])

    # Ridge
    ridge_results = run_ridge(X_train, y_train, X_test, y_test, k=5)
    print("\n=== Ridge ===")
    print("Melhor lambda:", ridge_results["best_lambda"])
    print("Manual - RMSE test:", ridge_results["manual"]["rmse_test"], "R2:", ridge_results["manual"]["r2_test"])
    print("Sklearn - RMSE test:", ridge_results["sklearn"]["rmse_test"], "R2:", ridge_results["sklearn"]["r2_test"])

    # PCR
    pcr_results = run_pcr(X_train, y_train, X_test, y_test, k=5)
    print("\n=== PCR ===")
    print("Melhor n_components:", pcr_results["best_n_components"])
    print("PCR - RMSE test:", pcr_results["test"]["rmse_test"], "R2:", pcr_results["test"]["r2_test"])

    # PLS
    pls_results = run_pls(X_train, y_train, X_test, y_test, k=5)
    print("\n=== PLS ===")
    print("Melhor n_components:", pls_results["best_n_components"])
    print("PLS - RMSE test:", pls_results["test"]["rmse_test"], "R2:", pls_results["test"]["r2_test"])

    # MLP
    mlp_results = run_mlp(X_train, y_train, X_test, y_test)
    print("\n=== MLP ===")
    print("MLP - RMSE test:", mlp_results["rmse_test"], "R2:", mlp_results["r2_test"])

    # Salvar CSV
    csv_path = OUTPUT_DIR / "hw2_models_summary.csv"
    save_results_to_csv(ols_results, ridge_results, pcr_results, pls_results, mlp_results, csv_path)


if __name__ == "__main__":
    main()
