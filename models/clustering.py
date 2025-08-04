import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.colors as mcolors

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.metrics import silhouette_score, davies_bouldin_score
from scipy.cluster.hierarchy import dendrogram, linkage

import warnings

from models.config import crime_features

warnings.filterwarnings("ignore")

# Veri yükle
df = pd.read_csv("/Users/sudearici/Documents/crime_prediction/data/processed_crime.csv")

# Şehir-yıl bazlı suç tiplerini grupla
cluster_df = df.groupby(['city', 'year'])[crime_features].sum().reset_index()

# Sadece suç sayılarını al (model giriş verisi)
X = cluster_df[crime_features]

# Ölçekleme
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# === Elbow Yöntemi (K için en uygun değer) ===
sse = []
k_range = range(2, 11)
for k in k_range:
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(X_scaled)
    sse.append(kmeans.inertia_)



# === KMeans ile Kümeleme ===
kmeans = KMeans(n_clusters=4, random_state=42)
kmeans_labels = kmeans.fit_predict(X_scaled)

# Küme etiketlerini DataFrame'e ekle (önce)
cluster_df['kmeans_cluster'] = kmeans_labels

# === Hierarchical Clustering ===
agg = AgglomerativeClustering(n_clusters=4)
agg_labels = agg.fit_predict(X_scaled)

# === Değerlendirme ===
def evaluate_clustering(X, labels, method_name):
    if len(set(labels)) > 1 and -1 not in labels:
        silhouette = silhouette_score(X, labels)
        db_index = davies_bouldin_score(X, labels)
        print(f"{method_name} - Silhouette Score: {silhouette:.4f} | Davies-Bouldin Index: {db_index:.4f}")
    else:
        print(f"{method_name} - Yeterli küme oluşmadı veya tüm veriler tek kümeye ait.")

evaluate_clustering(X_scaled, kmeans_labels, "KMeans")
evaluate_clustering(X_scaled, agg_labels, "Hierarchical")



# === Silhouette ve Davies-Bouldin Skorlarını Tek Grafikte Karşılaştır ===

# Veriyi uzun forma getir
comparison_df = pd.DataFrame({
    "KMeans": [0.5574, 0.9204],
    "Hierarchical": [0.6082, 0.7952]
}, index=["Silhouette Score (↑)", "Davies-Bouldin Index (↓)"])

# DataFrame'i .T ile transpose edip çizim için uygun hale getir
comparison_df_plot = comparison_df.T

# Renkler
colors = ["#1f77b4", "#ff7f0e"]

# Grafik çizimi
comparison_df_plot.plot(kind='bar', figsize=(8, 5), color=colors)

# Başlık ve etiketler
plt.title("Comparison of Clustering Quality Metrics", fontsize=14)
plt.ylabel("Score")
plt.xticks(rotation=0)
plt.ylim(0, 1.2)
plt.legend(title="Metrics")
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()


# === KMeans Kümeleme Sonuçları Görselleştirme ===
plt.figure(figsize=(10, 6))
sns.scatterplot(x=X_scaled[:, 0], y=X_scaled[:, 1], hue=kmeans_labels, palette="viridis", s=100, edgecolor="k")
plt.title("KMeans Clustering Results")
plt.xlabel("Attribute 1 (scaled)")
plt.ylabel("Attribute 2 (scaled)")
plt.legend(title="Clustering")
plt.tight_layout()
plt.show()

# === Hierarchical Dendrogram (isteğe bağlı) ===
linked = linkage(X_scaled, method='ward')
plt.figure(figsize=(10, 4))
dendrogram(linked, truncate_mode='level', p=5)
plt.title("Hierarchical Clustering Dendrogram")
plt.xlabel("Examples")
plt.ylabel("Distance")
plt.tight_layout()
plt.show()

# === KMeans Küme Profilleri (Ortalama) ===
kume_profilleri = cluster_df.groupby('kmeans_cluster')[crime_features].mean()
print("\n=== Crime Profiles of Clusters (Average Values) ===")
print(kume_profilleri)
for kume in kume_profilleri.index:
    print(f"\nKüme {kume} Özeti:")
    print(kume_profilleri.loc[kume].sort_values(ascending=False).head(5))

kume_etiketleri = {
    0: "0: Relatively Safe City",
    1: "1: High Crime Density City",
    2: "2: Medium Risk City",
    3: "3: Very Dangerous City"
}

print(kume_etiketleri)

# Pivot table ile şehirlerin yıllara göre küme değişimi (ortalama ile)
city_cluster_changes = cluster_df.pivot_table(index="year", columns="city", values="kmeans_cluster", aggfunc="mean")



# Isı haritası ile tüm şehirlerin yıllara göre ortalama kümeleri
plt.figure(figsize=(14, 8))
ax = sns.heatmap(
    city_cluster_changes.T,
    cmap="viridis",
    cbar_kws={'label': 'Average Cluster Number'},
    linewidths=0.5
)

# Colorbar nesnesini yakala ve özel etiketleri ata
colorbar = ax.collections[0].colorbar
colorbar.set_ticks([0, 1, 2, 3])
colorbar.set_ticklabels([kume_etiketleri[i] for i in range(4)])

plt.title("Average Cluster Change of Cities by Year")
plt.xlabel("Year")
plt.ylabel("City")
plt.yticks(rotation=0, fontsize=7)
plt.xticks(fontsize=8)
plt.tight_layout()
plt.show()




en_tehlikeli_kume = 3
tehlikeli_sehirler_yillar = cluster_df[cluster_df["kmeans_cluster"] == en_tehlikeli_kume][["city", "year"]]
print(tehlikeli_sehirler_yillar)
print(cluster_df[cluster_df["city"] == "İstanbul"][["year", "kmeans_cluster"]])


yillik_artis = cluster_df[cluster_df["kmeans_cluster"] == en_tehlikeli_kume]["year"].value_counts().sort_index()
plt.figure(figsize=(10,5))
plt.plot(yillik_artis.index, yillik_artis.values, marker='o')
plt.title("Number of Cities in the Most Dangerous Cluster (by Year)")
plt.xlabel("Year")
plt.ylabel("City Number")
plt.grid(True)
plt.show()




# Şehirlerin yıllara göre kümelerini sıralıyoruz
cluster_df_sorted = cluster_df.sort_values(by=['city', 'year'])

# Yıllar içinde kümelerin nasıl değiştiğini takip etmek için fark sütunu ekliyoruz
cluster_df_sorted['cluster_change'] = cluster_df_sorted.groupby('city')['kmeans_cluster'].diff()


# Zamanla daha tehlikeli kümelere geçen ve sonra o kümelerde kalmış şehirleri tespit ediyoruz
more_dangerous_cities = cluster_df_sorted[cluster_df_sorted['cluster_change'] > 0]

# Yalnızca zamanla daha tehlikeli kümelere geçmiş ve o kümelerde kalmış şehirleri seçiyoruz
# Yani, küme değişimi pozitif olan ve son küme numarası 2 veya 3 olan şehirleri filtreliyoruz
final_more_dangerous_cities = more_dangerous_cities[more_dangerous_cities['kmeans_cluster'] >= 2]

# Grafik için figür ve eksen oluşturuyoruz
plt.figure(figsize=(14, 8))

# Şehirlerin renklerini belirliyoruz
# Daha tehlikeli kümelere geçen şehirler farklı renklerde olacak
colors = ['red', 'blue', 'green', 'orange', 'purple', 'pink', 'brown', 'cyan', 'magenta', 'yellow']

# Tüm şehirlerin yıllarındaki kümelerini çiziyoruz
for i, city in enumerate(final_more_dangerous_cities['city'].unique()):
    city_data = cluster_df[cluster_df['city'] == city]  # Tüm yıllardaki verileri alıyoruz

    # Şehirlerin kümelerini çiziyoruz, her şehir farklı renkte
    plt.plot(city_data['year'], city_data['kmeans_cluster'], marker='o', label=city, color=colors[i % len(colors)],
             linestyle='-', markersize=8)

# Başlık ve etiketler
plt.title('Cities That Transition to More Dangerous Clusters Over Time and Remain in Those Clusters', fontsize=16)
plt.xlabel('Year', fontsize=14)
plt.ylabel('Cluster Number (Risk Level)', fontsize=14)

# Yandaki skalada yalnızca vurgulanan şehirlerin isimlerini göstereceğiz
plt.legend(title='Citys', bbox_to_anchor=(1.05, 1), loc='upper left')

# Grafik görünümünü sıkıştır
plt.tight_layout()

# Göster
plt.show()

# Zamanla daha tehlikeli kümelere geçmiş ve o kümelerde kalmış şehirlerin listesi
print("Zamanla Daha Tehlikeli Kümelere Geçiş Yapan ve O Kümelerde Kalan Şehirler:")
print(final_more_dangerous_cities[['city', 'year', 'kmeans_cluster']])



# Daha az tehlikeli kümelere geçen tüm şehirler
less_dangerous = cluster_df_sorted[cluster_df_sorted['cluster_change'] < 0]

# Bu şehirlerin isimlerini alalım
less_dangerous_city_names = less_dangerous['city'].unique()

# Bu şehirlerden yalnızca son küme değeri önceki yıllara göre daha düşük olanları alalım
final_less_dangerous_cities = []

for city in less_dangerous_city_names:
    city_data = cluster_df_sorted[cluster_df_sorted['city'] == city]
    if city_data['kmeans_cluster'].iloc[-1] < city_data['kmeans_cluster'].max():
        final_less_dangerous_cities.append(city)

print("Zamanla daha az tehlikeli kümeye geçip orada kalan şehirler:")
for city in final_less_dangerous_cities:
    print(city)

 #Grafik çizimi
plt.figure(figsize=(14, 8))
colors = plt.cm.tab10.colors  # 10 farklı renk

for i, city in enumerate(final_less_dangerous_cities):
    city_data = cluster_df_sorted[cluster_df_sorted['city'] == city]
    plt.plot(city_data['year'], city_data['kmeans_cluster'], marker='o', label=city, color=colors[i % 10], linestyle='-')

plt.title("Cities That Move to Less Dangerous Cluster Numbers Over Time and Stay There", fontsize=16)
plt.xlabel("Year", fontsize=14)
plt.ylabel("Cluster Numbers", fontsize=14)
plt.legend(title="Citys", bbox_to_anchor=(1.05, 1), loc='upper left')
plt.gca().invert_yaxis()  # Daha az tehlikeli kümeler yukarıda görünsün
plt.tight_layout()
plt.show()



# Toplam suç sayısını hesapla
cluster_df['total_crimes'] = cluster_df[crime_features].sum(axis=1)

# Yıllara ve kümelere göre ortalama suç oranını (toplam suç) hesapla
avg_crime_by_cluster_year = cluster_df.groupby(['year', 'kmeans_cluster'])['total_crimes'].mean().reset_index()

# Grafik çizimi
plt.figure(figsize=(12, 7))
for cluster_id in sorted(cluster_df['kmeans_cluster'].unique()):
    cluster_data = avg_crime_by_cluster_year[avg_crime_by_cluster_year['kmeans_cluster'] == cluster_id]
    plt.plot(cluster_data['year'], cluster_data['total_crimes'], marker='o', label=f"Küme {cluster_id}: {kume_etiketleri[cluster_id]}")

plt.title("Annual Average Crime Totals by Cluster", fontsize=16)
plt.xlabel("Year", fontsize=14)
plt.ylabel("Average Number of Crimes", fontsize=14)
plt.legend(title="Cluster", fontsize=10)
plt.grid(True)
plt.tight_layout()
plt.show()