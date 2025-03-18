import pandas as pd

# Veri setini yükle
df = pd.read_csv("/Users/sudearici/Documents/crime_prediction/data/crime.csv")

# Eksik değerleri kontrol et
print(df.isnull().sum())

# Boş değerleri 0 ile doldur
df = df.fillna(0)

# Verinin ilk 5 satırını göster
print(df.head())


