import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error

from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.holtwinters import ExponentialSmoothing

import warnings
warnings.filterwarnings("ignore")

# Veri yükle
df = pd.read_csv("/Users/sudearici/Documents/crime_prediction/data/processed_crime.csv")

# Yıllık toplam suçları hesapla
yearly_crime = df.groupby("year")["total_crime"].sum().reset_index()

# 2020'den önceki verileri eğitim için al
train = yearly_crime[yearly_crime["year"] < 2020]
X = train["year"].values.reshape(-1, 1)
y = train["total_crime"].values

# === Linear Regression ===
lr_model = LinearRegression()
lr_model.fit(X, y)
lr_pred = lr_model.predict(np.array([[2020]]))[0]

#p d q değerlerini bulmak için
from pmdarima import auto_arima

auto_model = auto_arima(
    train["total_crime"],
    seasonal=False,
    trace=True,
    error_action='ignore',
    suppress_warnings=True
)

print(auto_model.summary())


# === ARIMA ===
model = ARIMA(train["total_crime"], order=(2, 0, 0))
arima_result = model.fit()
arima_pred = arima_result.forecast(steps=1).values[0]

# === Exponential Smoothing (Holt-Winters) ===
es_model = ExponentialSmoothing(train["total_crime"], trend="add")
es_result = es_model.fit()
es_pred = es_result.forecast(steps=1).values[0]

# === Gerçek 2020 değeri ===
true_2020 = yearly_crime[yearly_crime["year"] == 2020]["total_crime"].values[0]

# === Hata metrik fonksiyonu ===
def calculate_metrics(true, pred):
    mae = mean_absolute_error([true], [pred])
    mse = mean_squared_error([true], [pred])
    rmse = np.sqrt(mse)
    return mae, mse, rmse

# === Sonuçlar ===
results = {
    "Model": ["Linear Regression", "ARIMA", "Exponential Smoothing"],
    "Tahmin": [lr_pred, arima_pred, es_pred],
    "MAE": [],
    "MSE": [],
    "RMSE": []
}

for model_name, pred in zip(results["Model"], results["Tahmin"]):
    mae, mse, rmse = calculate_metrics(true_2020, pred)
    results["MAE"].append(mae)
    results["MSE"].append(mse)
    results["RMSE"].append(rmse)

# Check results dict before DataFrame creation
print("=== Results Dict ===")
print(results)

result_df = pd.DataFrame(results)

# === Gerçek 2020 değeri ve tahminleri birlikte yazdır
print(f"Gerçek 2020 Değeri: {true_2020}")
print("\n=== Tahmin Sonuçları (2020) ===")
print(result_df)

# === Görselleştirme ===
plt.figure(figsize=(12, 6))
metrics = ["MAE", "MSE", "RMSE"]
for i, metric in enumerate(metrics):
    plt.subplot(1, 3, i+1)
    sns.barplot(x="Model", y=metric, data=result_df, palette="Set2")
    plt.title(metric)
    plt.xticks(rotation=15)

plt.tight_layout()
plt.suptitle("Zaman Serisi Modellerinin Performans Karşılaştırması (2020)", fontsize=16, y=1.05)
plt.show()

# === Gerçek ve Tahminleri Görselleştirme ===
plt.figure(figsize=(10, 6))
plt.bar(results["Model"], results["Tahmin"], color="skyblue", label="Tahmin")
plt.axhline(true_2020, color="red", linestyle="--", label="Gerçek 2020")
plt.title("Crime Predictions and Actual Values ​​for 2020")
plt.xlabel("Model")
plt.ylabel("Total Number of Crimes")
plt.legend()
plt.show()