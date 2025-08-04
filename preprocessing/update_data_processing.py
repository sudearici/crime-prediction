import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
# Veriyi yükle
crime_details = pd.read_csv('/Users/sudearici/Documents/crime_prediction/data/update_crime.csv', sep=';')

# Tüm verilerdeki boşlukları kaldır (özellikle sayısal sütunlarda " 4 212 " gibi boşluklu sayılar var)
crime_details = crime_details.applymap(lambda x: str(x).replace(" ", "") if isinstance(x, str) else x)

# Genel Bilgiler
print("Veri setinin boyutu:", crime_details.shape)
print("\nSütun tipleri:\n", crime_details.dtypes)
print("\nEksik değer kontrolü:\n", crime_details.isnull().sum())
print("\nTanımlayıcı istatistikler:\n", crime_details.describe())
print("\nSütun isimleri:\n",crime_details.columns)
# Boş değerleri 0 ile doldur
df = crime_details.fillna(0)

# Kategorik değişkenleri encode etme (Label Encoding)
categorical_cols = df.select_dtypes(include=['object']).columns
encoder = LabelEncoder()
for col in categorical_cols:
    df[col] = encoder.fit_transform(df[col])
""""
# Son olarak işlenmiş veriyi kaydedelim
crime_details.to_csv('/Users/sudearici/Documents/crime_prediction/data/processed_update_crime.csv', index=False, sep=';')

"""