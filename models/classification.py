from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
import pandas as pd

#veri yükleme
df = pd.read_csv("/Users/sudearici/Documents/crime_prediction/data/processed_crime.csv")

df_clean = df.copy()

# total_crime değerini eşik değere göre sınıflandır (median üstü 1, altı 0)
threshold = df_clean['total_crime'].median()
df_clean['target'] = (df_clean['total_crime'] > threshold).astype(int)

# Kategorik değişkenleri sayısallaştır
df_clean['city'] = LabelEncoder().fit_transform(df_clean['city'])

# Hedef ve özellik sütunlarını ayır
X = df_clean.drop(columns=['total_crime', 'target'])
y = df_clean['target']

# Özellikleri ölçeklendir
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Eğitim ve test verisine ayır
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Modelleri eğit ve değerlendir
results = {}

# Naive Bayes
nb = GaussianNB()
nb.fit(X_train, y_train)
y_pred_nb = nb.predict(X_test)
results['Naive Bayes'] = {
    'accuracy': accuracy_score(y_test, y_pred_nb),
    'precision': precision_score(y_test, y_pred_nb),
    'recall': recall_score(y_test, y_pred_nb),
    'f1_score': f1_score(y_test, y_pred_nb),
    'roc_auc': roc_auc_score(y_test, y_pred_nb)
}

# SVM
svm = SVC(probability=True)
svm.fit(X_train, y_train)
y_pred_svm = svm.predict(X_test)
results['SVM'] = {
    'accuracy': accuracy_score(y_test, y_pred_svm),
    'precision': precision_score(y_test, y_pred_svm),
    'recall': recall_score(y_test, y_pred_svm),
    'f1_score': f1_score(y_test, y_pred_svm),
    'roc_auc': roc_auc_score(y_test, svm.predict_proba(X_test)[:,1])
}

# Random Forest
rf = RandomForestClassifier()
rf.fit(X_train, y_train)
y_pred_rf = rf.predict(X_test)
results['Random Forest'] = {
    'accuracy': accuracy_score(y_test, y_pred_rf),
    'precision': precision_score(y_test, y_pred_rf),
    'recall': recall_score(y_test, y_pred_rf),
    'f1_score': f1_score(y_test, y_pred_rf),
    'roc_auc': roc_auc_score(y_test, rf.predict_proba(X_test)[:,1])
}

# Logistic Regression
lr = LogisticRegression()
lr.fit(X_train, y_train)
y_pred_lr = lr.predict(X_test)
results['Logistic Regression'] = {
    'accuracy': accuracy_score(y_test, y_pred_lr),
    'precision': precision_score(y_test, y_pred_lr),
    'recall': recall_score(y_test, y_pred_lr),
    'f1_score': f1_score(y_test, y_pred_lr),
    'roc_auc': roc_auc_score(y_test, lr.predict_proba(X_test)[:,1])
}

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import InputLayer, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping

# Modeli oluştur
dnn_model = Sequential([
    InputLayer(input_shape=(X_train.shape[1],)),
    Dense(32, activation='relu'),
    Dropout(0.3),  # %30 dropout ile overfitting'i azalt
    Dense(16, activation='relu'),
    Dropout(0.2),
    Dense(1, activation='sigmoid')
])

# Modeli derle
dnn_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Early stopping tanımla (validation loss 5 epoch boyunca iyileşmezse durdur)
early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

# Modeli eğit (batch_size belirle, validation split ekle)
dnn_model.fit(
    X_train, y_train,
    epochs=50,
    batch_size=32,
    validation_split=0.2,
    callbacks=[early_stop],
    verbose=1
)

# Tahmin yap
y_pred_dnn_prob = dnn_model.predict(X_test).flatten()
y_pred_dnn = (y_pred_dnn_prob > 0.5).astype(int)

# Sonuçları kaydet
results['DNN (Keras)'] = {
    'accuracy': accuracy_score(y_test, y_pred_dnn),
    'precision': precision_score(y_test, y_pred_dnn),
    'recall': recall_score(y_test, y_pred_dnn),
    'f1_score': f1_score(y_test, y_pred_dnn),
    'roc_auc': roc_auc_score(y_test, y_pred_dnn_prob)
}


# Sonuçları tabloya dök
results_df = pd.DataFrame(results).T

# Sonuçları ekrana yazdır
print(results_df)

# Görselleştirme: Metriklerin her model için görselleştirilmesi
metrics = ['accuracy', 'precision', 'recall', 'f1_score', 'roc_auc']

# Sonuçları tabloya dök
results_df = pd.DataFrame(results).T

import seaborn as sns
import matplotlib.pyplot as plt

plt.style.use('seaborn-whitegrid')
sns.set_palette("Set2")

metrics = ['accuracy', 'precision', 'recall', 'f1_score', 'roc_auc']
ax = results_df[metrics].plot(kind='bar', figsize=(12, 6), width=0.75)

plt.title('Model Performance Matrices', fontsize=16, weight='bold')
plt.ylabel('Skor', fontsize=12)
plt.xlabel('Models', fontsize=12)
plt.xticks(rotation=45, ha='right', fontsize=10)
plt.yticks(fontsize=10)
plt.ylim(0, 1.05)

# Legend'ı sağa al
plt.legend(title='Matrices', fontsize=10, title_fontsize=11,
           bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0.)

# Barların üstüne skorları yaz
for p in ax.patches:
    ax.annotate(f'{p.get_height():.2f}',
                (p.get_x() + p.get_width() / 2., p.get_height()),
                ha='center', va='bottom', fontsize=9, color='black', xytext=(0, 5),
                textcoords='offset points')

plt.tight_layout()
plt.show()