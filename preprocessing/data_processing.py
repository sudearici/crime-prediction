import pandas as pd
from sklearn.preprocessing import LabelEncoder

# Veri seti yüklendi.
df = pd.read_csv("/Users/sudearici/Documents/crime_prediction/data/crime.csv", sep=";")

#Veri seti üzerinde genel bilgiler.
print("Veri setinin boyutu:", df.shape)
print("\nSütun tipleri:\n", df.dtypes)
print("\nEksik değer kontrolü:\n", df.isnull().sum())
print("\nTanımlayıcı istatistikler:\n", df.describe())
print("\nSütun isimleri:\n",df.columns)

# Boş değerleri 0 ile dolduruldu.
df = df.fillna(0)

# Kategorik değişkenleri encode etme (Label Encoding)
categorical_cols = df.select_dtypes(include=['object']).columns
encoder = LabelEncoder()
for col in categorical_cols:
    df[col] = encoder.fit_transform(df[col])

# Şehir kodlarını isimlere dönüştürüldü.
city_mapping = {
    77: "İstanbul", 65: "Tekirdağ", 24: "Edirne", 46: "Kırklareli", 11: "Balıkesir",
    74: "Çanakkale", 78: "İzmir", 9: "Aydın", 21: "Denizli", 53: "Muğla", 50: "Manisa",
    2: "Afyonkarahisar", 45: "Kütahya", 69: "Uşak", 20: "Bursa", 28: "Eskişehir",
    15: "Bilecik", 43: "Kocaeli", 60: "Sakarya", 23: "Düzce", 18: "Bolu", 71: "Yalova",
    5: "Ankara", 44: "Konya", 38: "Karaman", 6: "Antalya", 34: "Isparta", 19: "Burdur",
    0: "Adana", 52: "Mersin", 33: "Hatay", 36: "Kahramanmaraş", 58: "Osmaniye", 47: "Kırıkkale",
    3: "Aksaray", 56: "Niğde", 55: "Nevşehir", 48: "Kırşehir", 41: "Kayseri", 64: "Sivas",
    72: "Yozgat", 73: "Zonguldak", 37: "Karabük", 12: "Bartın", 40: "Kastamonu", 75: "Çankırı",
    63: "Sinop", 61: "Samsun", 66: "Tokat", 76: "Çorum", 4: "Amasya", 67: "Trabzon", 57: "Ordu",
    30: "Giresun", 59: "Rize", 8: "Artvin", 31: "Gümüşhane", 27: "Erzurum", 26: "Erzincan",
    14: "Bayburt", 10: "Ağrı", 39: "Kars", 35: "Iğdır", 7: "Ardahan", 49: "Malatya", 25: "Elâzığ",
    16: "Bingöl", 68: "Tunceli", 70: "Van", 54: "Muş", 17: "Bitlis", 32: "Hakkari", 29: "Gaziantep",
    1: "Adıyaman", 42: "Kilis", 79: "Şanlıurfa", 22: "Diyarbakır", 51: "Mardin", 13: "Batman",
    80: "Şırnak", 62: "Siirt"
}

# Şehir isimlerini veri setine eklendi.
df["city"] = df["city"].map(city_mapping)

# Sayısal ve kategorik değişkenleri ayırt edildi.
numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns
categorical_cols = df.select_dtypes(include=['object']).columns

# İşlenmiş veriyi kaydedildi.
df.to_csv("/Users/sudearici/Documents/crime_prediction/data/processed_crime.csv", index=False)

# Dönüşmüş veri.
print(df.head())

# En yüksek suç türünin belirlenmesi için total ve other içeren sütunları hariç tutuldu.
exclude_cols = ['Total', 'total', 'Other', 'other', 'year', 'city_code']  # city_code varsa
suç_türü_sütunları = [col for col in numeric_cols if all(x.lower() not in col.lower() for x in ['total', 'other', 'year'])]

# Her suç türünün toplam sayısını hesaplandı.
suç_toplamları = df[suç_türü_sütunları].sum().sort_values(ascending=False)

# En yüksek 5 suç türü detay analizi için belirlendi.
print("\nEn yüksek 5 suç türü:\n")
print(suç_toplamları.head(5))

