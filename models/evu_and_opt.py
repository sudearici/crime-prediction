import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error
import xgboost as xgb
from sklearn.preprocessing import StandardScaler

# Veri setini yükle
df = pd.read_csv("/Users/sudearici/Documents/crime_prediction/data/processed_crime.csv")
# Kategorik değişkenlere one-hot encoding uygulama
df = pd.get_dummies(df, columns=['city', 'gender'])

# Özellikler ve hedef değişkeni ayır (target_column, verindeki hedef değişken adıyla değiştir)
X = df.drop('total_crime', axis=1)  # 'total_crime' hedef değişkeni
y = df['total_crime']

# Eğitim ve test setlerine ayır
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Veriyi ölçeklendir
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Hiperparametre arama alanlarını tanımla
param_grid_ridge = {
    'alpha': [0.1, 1, 10, 100]
}

param_grid_lasso = {
    'alpha': [0.1, 1, 10, 100],
    'max_iter': [1000]
}

param_dist_rf = {
    'n_estimators': [50, 100],  # Daha az seçim
    'max_depth': [5, 10, 15],  # Daha dar alan
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

param_dist_xgb = {
    'n_estimators': [50, 100],  # Daha az seçim
    'max_depth': [3, 5, 7],  # Daha dar alan
    'learning_rate': [0.01, 0.1],
    'subsample': [0.6, 0.8]
}

param_dist_tree = {
    'max_depth': [5, 10, 15],  # Daha dar alan
    'min_samples_split': [2, 5],
    'min_samples_leaf': [1, 2]
}


# Tuning işlemi sırasında verbose parametresi ekleyelim
def tune_model(model, param_grid, X_train_data, y_train_data, model_name='model'):
    print(f"Tuning {model_name}...")
    # Eğer param_grid bir dict ise GridSearchCV, değilse RandomizedSearchCV
    try:
        if isinstance(param_grid, dict):  # GridSearchCV
            search = GridSearchCV(model, param_grid, cv=5, scoring='neg_mean_squared_error', verbose=1)
        else:  # RandomizedSearchCV
            search = RandomizedSearchCV(model, param_distributions=param_grid, n_iter=20, cv=5,
                                        scoring='neg_mean_squared_error', random_state=42, verbose=1)

        search.fit(X_train_data, y_train_data)

        best_model = search.best_estimator_
        best_params = search.best_params_
        best_score = search.best_score_

        print(f"En iyi parametreler: {best_params}")
        print(f"En iyi MSE: {-best_score}")
        return best_model
    except Exception as e:
        print(f"Error in tuning {model_name}: {str(e)}")


# Model oluştur ve tuning yap
best_lr = tune_model(LinearRegression(), {}, X_train_scaled, y_train, 'Lineer Regresyon')
best_ridge = tune_model(Ridge(), param_grid_ridge, X_train_scaled, y_train, 'Ridge Regresyon')
best_lasso = tune_model(Lasso(), param_grid_lasso, X_train_scaled, y_train, 'Lasso Regresyon')
best_rf = tune_model(RandomForestRegressor(), param_dist_rf, X_train_scaled, y_train, 'Random Forest')
best_xgb = tune_model(xgb.XGBRegressor(), param_dist_xgb, X_train_scaled, y_train, 'XGBoost')
best_tree = tune_model(DecisionTreeRegressor(), param_dist_tree, X_train_scaled, y_train, 'Karar Ağacı')


# Test seti üzerinde değerlendirme
def evaluate_model(model, X_test_data, y_test_data, model_name='model'):
    try:
        y_pred = model.predict(X_test_data)
        mse = mean_squared_error(y_test_data, y_pred)
        print(f"{model_name} Test MSE: {mse}")
        return mse
    except Exception as e:
        print(f"Error in evaluation of {model_name}: {str(e)}")


# Test seti üzerinde performans
evaluate_model(best_lr, X_test_scaled, y_test, 'Lineer Regresyon')
evaluate_model(best_ridge, X_test_scaled, y_test, 'Ridge Regresyon')
evaluate_model(best_lasso, X_test_scaled, y_test, 'Lasso Regresyon')
evaluate_model(best_rf, X_test_scaled, y_test, 'Random Forest')
evaluate_model(best_xgb, X_test_scaled, y_test, 'XGBoost')
evaluate_model(best_tree, X_test_scaled, y_test, 'Karar Ağacı')
