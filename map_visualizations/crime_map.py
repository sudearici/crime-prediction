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

### 1. Heatmap: Suç Yoğunluk Haritası (Folium)
map_center = [39.0, 35.0]
crime_map = folium.Map(location=map_center, zoom_start=6)

heat_data = [[row['lat'], row['lon'], row['total_crime']] for index, row in df_total.iterrows()]
HeatMap(heat_data).add_to(crime_map)

crime_map.save("crime_heatmap.html")
print("🔴 Heatmap kaydedildi: crime_heatmap.html")

### 2. Zaman İçinde Suç Değişimi Haritası (Plotly)
fig = px.scatter_geo(df_total,
                     lat='lat',
                     lon='lon',
                     color='total_crime',
                     size='total_crime',
                     animation_frame='year',
                     scope='europe',
                     hover_name='city',
                     title='Zaman İçinde Türkiye\'de Suç Yoğunluğu',
                     template='plotly_dark')

fig.write_html("animated_crime_map.html")
print("🟢 Animasyonlu harita kaydedildi: animated_crime_map.html")
