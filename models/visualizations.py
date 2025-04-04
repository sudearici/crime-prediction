import matplotlib.pyplot as plt
import numpy as np

# Modellerin isimleri
models = ['Lineer Regresyon', 'Ridge Regresyon', 'Lasso Regresyon',
          'Karar Ağacı', 'Random Forest', 'XGBoost']

# MSE ve R2 değerleri
mse_values = [8.064861841271969e-24, 33.33793244401248, 84.08883456649774,
              778377.6307692308, 239930.27043692302, 214777.609375]
r2_values = [1.0, 0.9999965759293673, 0.9999913634083499,
             0.9200544307560217, 0.9753572542546148, 0.9779406189918518]

# Grafik boyutunu ayarla
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 7))

# MSE'yi görselleştir
ax1.bar(models, mse_values, color='skyblue')
ax1.set_title('Model Performansları (MSE)')
ax1.set_xlabel('Modeller')
ax1.set_ylabel('MSE')
ax1.set_yscale('log')  # MSE'nin logaritmik ölçeği daha uygun olabilir çünkü bazı değerler çok küçük.

# R2'yi görselleştir
ax2.bar(models, r2_values, color='lightgreen')
ax2.set_title('Model Performansları (R2)')
ax2.set_xlabel('Modeller')
ax2.set_ylabel('R2')

# Grafikleri göster
plt.tight_layout()
plt.show()
