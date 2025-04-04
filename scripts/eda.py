import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from scripts.data_processing import df, numeric_cols, categorical_cols  # Burada df'yi alıyoruz

# 🎯 Histogram
df[numeric_cols].hist(bins=20, figsize=(12, 10))
plt.suptitle("Veri Dağılımı (Histogramlar)")
plt.show()

# 🎯 Boxplot
plt.figure(figsize=(24, 16))
sns.boxplot(data=df[numeric_cols])
plt.xticks(rotation=90)
plt.title("Aykırı Değer Analizi ve Veri Dağılımı")
plt.show()

# 🎯 Korelasyon Matrisi
corr_matrix = df[numeric_cols].corr()
plt.figure(figsize=(24, 16))
sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5)
plt.title("Korelasyon Matrisi")
plt.show()

# 🎯 Cinsiyet Bazlı Suç Dağılımı
plt.figure(figsize=(10, 6))
sns.boxplot(data=df, x="gender", y="total_crime", hue="gender", palette="Set2", legend=False)
plt.title("Cinsiyete Göre Suç Dağılımı")
plt.show()

plt.figure(figsize=(10, 6))
sns.violinplot(data=df, x="gender", y="total_crime", hue="gender", palette="muted", inner="quartile", legend=False)
plt.title("Cinsiyete Göre Suç Dağılımı (Violin Plot)")
plt.show()

# 🎯 Zaman Serisi - Şehirlere Göre
plt.figure(figsize=(12, 6))
sns.lineplot(data=df, x="year", y="total_crime", hue="city", alpha=0.5, legend=False)
sns.lineplot(data=df.groupby("year")["total_crime"].mean(), label="Ortalama", linewidth=3, color="black")
plt.title("Şehirlere Göre Yıllık Suç Trendleri")
plt.grid(True)
plt.show()

# 🎯 Coğrafi Suç Dağılımı
city_crime = df.groupby("city")["total_crime"].sum().reset_index()
fig = px.bar(city_crime.sort_values("total_crime", ascending=False),
             x="total_crime", y="city", orientation="h",
             color="total_crime", title="Şehirlere Göre Toplam Suç Dağılımı")
fig.show()

# 🎯 Şiddet Suçları Toplamı
df["violent_crimes"] = df["homicide"] + df["assault"] + df["sexual_crimes"] + df["robbery"]
df_violent = df.groupby("city")["violent_crimes"].sum().reset_index()

plt.figure(figsize=(16, 10))
sns.barplot(y="city", x="violent_crimes", data=df_violent.sort_values("violent_crimes", ascending=False), palette="Oranges_r")
plt.title("Şehirlere Göre Şiddet Suçları Toplamı")
plt.show()

# 🎯 Yıllara Göre Toplam Suç Sayısı
plt.figure(figsize=(12, 6))
sns.lineplot(data=df.groupby("year")["total_crime"].sum().reset_index(), x="year", y="total_crime", marker="o", color="red")
plt.title("Yıllara Göre Toplam Suç Sayısı")
plt.grid(True)
plt.show()

# 🎯 Suç Türlerinin Toplam Dağılımı
numeric_crime_cols = df.select_dtypes(include='number').columns.drop(['TR_ID', 'year', 'total_crime', 'violent_crimes', 'city_encoded'], errors="ignore")
crime_totals = df[numeric_crime_cols].sum().sort_values(ascending=False)

plt.figure(figsize=(14, 7))
sns.barplot(x=crime_totals.values, y=
crime_totals.index, palette="Blues_r")
plt.title("Suç Türlerinin Toplam Dağılımı")
plt.xlabel("Toplam Suç Sayısı")
plt.show()

#🎯 En Çok Suç İşlenen 10 Şehir

top_cities = df.groupby("city")["total_crime"].sum().nlargest(10).reset_index()
plt.figure(figsize=(12, 6))
sns.barplot(x="total_crime", y="city", data=top_cities, palette="Reds_r")
plt.title("En Çok Suç İşlenen 10 Şehir")
plt.show()

#🎯 Kategorik Değişken Frekansları

for col in categorical_cols: print(f"{col} - Kategorik Dağılım:\n", df[col].value_counts(), "\n")