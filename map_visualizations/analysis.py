import pandas as pd
import folium
from folium.plugins import HeatMap
from geopy.geocoders import Nominatim
from tqdm import tqdm
import plotly.express as px

# CSV dosyasını yükle
df = pd.read_csv("/Users/sudearici/Documents/crime_prediction/data/processed_crime.csv")

# Sadece kadın+erkek toplamına odaklanmak için gruplama
df_total = df.groupby(['year', 'city'], as_index=False).agg({'total_crime': 'sum'})

# Koordinatları bulmak için şehir listesi
cities = df_total['city'].unique()

# Koordinat sözlüğü oluştur
geolocator = Nominatim(user_agent="crime_mapper")
coordinates = {}

print("Şehir koordinatları alınıyor...")
for city in tqdm(cities):
    try:
        location = geolocator.geocode(f"{city}, Turkey")
        if location:
            coordinates[city] = (location.latitude, location.longitude)
    except:
        coordinates[city] = (None, None)

# Koordinatları dataframe'e ekle
df_total['lat'] = df_total['city'].map(lambda x: coordinates.get(x, (None, None))[0])
df_total['lon'] = df_total['city'].map(lambda x: coordinates.get(x, (None, None))[1])

# Geçersiz koordinatları temizle
df_total = df_total.dropna(subset=['lat', 'lon'])

# 2. Eksik verileri kontrol et (gerekirse temizlenebilir)
df = df.dropna(subset=['city', 'lat', 'lon', 'total_crime'])

# 3. Şehir bazlı toplam suçları grupla
df_total = df.groupby(['city', 'lat', 'lon'], as_index=False)['total_crime'].sum()

# 4. Harita Oluşturma
map_center = [39.0, 35.0]
crime_map = folium.Map(location=map_center, zoom_start=6)

# 5. Heatmap Katmanı
heat_data = [[row['lat'], row['lon'], row['total_crime']] for index, row in df_total.iterrows()]
HeatMap(heat_data, radius=25).add_to(crime_map)

# 6. Tooltip ile suç sayısı etiketleri
for _, row in df_total.iterrows():
    folium.CircleMarker(
        location=[row['lat'], row['lon']],
        radius=6,
        fill=True,
        fill_color='crimson',
        fill_opacity=0.7,
        color=None,
        tooltip=f"{row['city']}: {int(row['total_crime'])} suç"
    ).add_to(crime_map)

# 7. Haritayı HTML olarak kaydet
crime_map.save("crime_heatmap_with_tooltip.html")
print("✅ Harita başarıyla oluşturuldu: crime_heatmap_with_tooltip.html")