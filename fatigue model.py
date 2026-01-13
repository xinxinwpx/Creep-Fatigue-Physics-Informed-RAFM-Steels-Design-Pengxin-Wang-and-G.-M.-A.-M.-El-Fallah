
import matplotlib
matplotlib.use('TkAgg')

import optuna, warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.neural_network import MLPRegressor
import xgboost as xgb

# ================================================================
# Step 1. Data loading
# ================================================================
FILE = "疲劳建模对比.xlsx"  # Fatigue life dataset
df = pd.read_excel(FILE)
df = df.replace([np.inf, -np.inf], np.nan).dropna()

print(f"✅ Data loaded successfully: {df.shape[0]} samples, {df.shape[1]} features")

X = df.iloc[:, :-1]
y = df.iloc[:, -1]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ================================================================
# Step 2. General evaluation function (save scatter CSV)
# ================================================================
def evaluate_and_save(model_name, model, X_test, y_test, y_pred, results):
    """
    Compute metrics, save scatter CSV, and generate scatter plot
    """
    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))

    print(f"\n[{model_name}] R²={r2:.3f}, MAE={mae:.3f}, RMSE={rmse:.3f}")
    results.append([model_name, r2, mae, rmse])

    # Save scatter data (CSV)
    df_pred = pd.DataFrame({
        "True_log10Nf": y_test.values,
        "Pred_log10Nf": y_pred,
        "Residual": y_test.values - y_pred
    })
    df_pred.to_csv(f"{model_name}_Scatter_Data.csv", index=False)
    print(f"✅ Scatter data saved: {model_name}_Scatter_Data.csv")

    # Scatter plot
    plt.figure(figsize=(5.5, 5.5))
    plt.scatter(y_test, y_pred, alpha=0.7, color="royalblue", edgecolors="k")
    plt.plot(
        [y_test.min(), y_test.max()],
        [y_test.min(), y_test.max()],
        'r--', lw=2
    )
    plt.xlabel("True log10(Nf)", fontsize=12)
    plt.ylabel("Predicted log10(Nf)", fontsize=12)
    plt.title(f"{model_name}\nR²={r2:.3f}, RMSE={rmse:.3f}", fontsize=13)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{model_name}_Scatter.png", dpi=300)
    plt.close()

# ================================================================
# Step 3. Optuna hyperparameter optimization for each model
# ================================================================

# ---------- Random Forest ----------
def rf_objective(trial):
    n_estimators = trial.suggest_int("n_estimators", 100, 500)
    max_depth = trial.suggest_int("max_depth", 3, 20)
    min_samples_split = trial.suggest_int("min_samples_split", 2, 10)
    min_samples_leaf = trial.suggest_int("min_samples_leaf", 1, 5)

    model = RandomForestRegressor(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        min_samples_leaf=min_samples_leaf,
        random_state=42,
        n_jobs=-1
    )
    return -np.mean(
        cross_val_score(model, X_train_scaled, y_train, cv=5, scoring="r2")
    )

rf_study = optuna.create_study(direction="minimize")
rf_study.optimize(rf_objective, n_trials=25)
rf_best = rf_study.best_params
print("\n[RF Best Parameters]", rf_best)

rf_model = RandomForestRegressor(**rf_best, random_state=42, n_jobs=-1)
rf_model.fit(X_train_scaled, y_train)
rf_pred = rf_model.predict(X_test_scaled)

# ---------- Gradient Boosting ----------
def gbdt_objective(trial):
    n_estimators = trial.suggest_int("n_estimators", 100, 500)
    learning_rate = trial.suggest_float("learning_rate", 0.01, 0.3)
    max_depth = trial.suggest_int("max_depth", 3, 15)
    subsample = trial.suggest_float("subsample", 0.6, 1.0)

    model = GradientBoostingRegressor(
        n_estimators=n_estimators,
        learning_rate=learning_rate,
        max_depth=max_depth,
        subsample=subsample,
        random_state=42
    )
    return -np.mean(
        cross_val_score(model, X_train_scaled, y_train, cv=5, scoring="r2")
    )

gbdt_study = optuna.create_study(direction="minimize")
gbdt_study.optimize(gbdt_objective, n_trials=25)
gbdt_best = gbdt_study.best_params
print("\n[GBDT Best Parameters]", gbdt_best)

gbdt_model = GradientBoostingRegressor(**gbdt_best, random_state=42)
gbdt_model.fit(X_train_scaled, y_train)
gbdt_pred = gbdt_model.predict(X_test_scaled)

# ---------- XGBoost ----------
def xgb_objective(trial):
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 200, 800),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3),
        "max_depth": trial.suggest_int("max_depth", 3, 10),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
        "random_state": 42,
        "n_jobs": -1
    }
    model = xgb.XGBRegressor(**params)
    return -np.mean(
        cross_val_score(model, X_train_scaled, y_train, cv=5, scoring="r2")
    )

xgb_study = optuna.create_study(direction="minimize")
xgb_study.optimize(xgb_objective, n_trials=25)
xgb_best = xgb_study.best_params
print("\n[XGBoost Best Parameters]", xgb_best)

xgb_model = xgb.XGBRegressor(**xgb_best)
xgb_model.fit(X_train_scaled, y_train)
xgb_pred = xgb_model.predict(X_test_scaled)

# ---------- MLP ----------
def mlp_objective(trial):
    hidden_units = trial.suggest_int("hidden_units", 64, 256)
    n_layers = trial.suggest_int("n_layers", 1, 3)
    hidden_layer_sizes = tuple([hidden_units] * n_layers)

    learning_rate_init = trial.suggest_float(
        "learning_rate_init", 1e-4, 1e-2, log=True
    )
    alpha = trial.suggest_float("alpha", 1e-5, 1e-2, log=True)

    model = MLPRegressor(
        hidden_layer_sizes=hidden_layer_sizes,
        learning_rate_init=learning_rate_init,
        alpha=alpha,
        max_iter=1000,
        random_state=42
    )
    return -np.mean(
        cross_val_score(model, X_train_scaled, y_train, cv=5, scoring="r2")
    )

mlp_study = optuna.create_study(direction="minimize")
mlp_study.optimize(mlp_objective, n_trials=25)
mlp_best = mlp_study.best_params
print("\n[MLP Best Parameters]", mlp_best)

# Build final MLP model
hidden_units = mlp_best["hidden_units"]
n_layers = mlp_best["n_layers"]
hidden_layer_sizes = tuple([hidden_units] * n_layers)

mlp_model = MLPRegressor(
    hidden_layer_sizes=hidden_layer_sizes,
    learning_rate_init=mlp_best["learning_rate_init"],
    alpha=mlp_best["alpha"],
    max_iter=1000,
    random_state=42
)
mlp_model.fit(X_train_scaled, y_train)
mlp_pred = mlp_model.predict(X_test_scaled)

# ================================================================
# Step 4. Model evaluation + scatter CSV saving + performance table
# ================================================================
results = []
evaluate_and_save("RF_Model", rf_model, X_test_scaled, y_test, rf_pred, results)
evaluate_and_save("GBDT_Model", gbdt_model, X_test_scaled, y_test, gbdt_pred, results)
evaluate_and_save("XGBoost_Model", xgb_model, X_test_scaled, y_test, xgb_pred, results)
evaluate_and_save("MLP_Model", mlp_model, X_test_scaled, y_test, mlp_pred, results)

# Summary table
df_results = pd.DataFrame(results, columns=["Model", "R²", "MAE", "RMSE"])
df_results.to_csv("Model_Performance_Comparison.csv", index=False)

print("\n✅ Model performance comparison table saved: Model_Performance_Comparison.csv")
print(df_results)

# ================================================================
# Step 5. Visualization: performance bar chart
# ================================================================
plt.figure(figsize=(8, 5))
bar_width = 0.25
x = np.arange(len(df_results["Model"]))

plt.bar(x - bar_width, df_results["R²"], bar_width, label="R²")
plt.bar(x, df_results["MAE"], bar_width, label="MAE")
plt.bar(x + bar_width, df_results["RMSE"], bar_width, label="RMSE")

plt.xticks(x, df_results["Model"])
plt.xlabel("Model", fontsize=12)
plt.ylabel("Metric Value", fontsize=12)
plt.title("Model Performance Comparison (Fatigue Life Prediction)", fontsize=13)
plt.legend()
plt.tight_layout()
plt.savefig("Model_Performance_BarChart.png", dpi=300)
plt.show()

print("\n🎯 All models trained, evaluated, and all scatter CSVs and figures saved successfully!")
