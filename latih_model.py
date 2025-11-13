import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
import joblib

print("Sedang membuat data latih dan melatih model...")

# 1. Simulasi Data Latih (Dataset Historis Buatan)
# Kita buat 2000 sampel data kejadian banjir/tidak banjir
np.random.seed(42)
n_samples = 2000

data = {
    # Curah Hujan (mm/jam)
    'curah_hujan': np.random.uniform(0, 150, n_samples),
    # Pasang Laut (cm) - Diukur dari rata-rata muka laut
    'pasang_laut': np.random.uniform(100, 300, n_samples),
    # Penurunan Tanah (cm/tahun)
    'land_subsidence': np.random.uniform(2, 15, n_samples),
    # Elevasi Tanah (meter) - Ini nanti diambil dari Peta DEMNAS saat visualisasi
    'elevasi': np.random.uniform(-2, 10, n_samples) 
}

df = pd.DataFrame(data)

# 2. Logika Target (Tinggi Genangan)
# Rumus simulasi: Genangan terjadi jika akumulasi air (hujan + pasang) melebihi elevasi tanah
# Faktor konversi kasar untuk simulasi logika
def hitung_genangan(row):
    water_level_cm = (row['pasang_laut']) + (row['curah_hujan'] * 0.5) + (row['land_subsidence'] * 2)
    terrain_height_cm = row['elevasi'] * 100  # Konversi meter ke cm
    
    genangan = water_level_cm - terrain_height_cm
    
    # Jika genangan negatif, artinya tanah lebih tinggi dari air (kering/0)
    return max(0, genangan)

df['tinggi_genangan'] = df.apply(hitung_genangan, axis=1)

# 3. Training Model
X = df[['curah_hujan', 'pasang_laut', 'land_subsidence', 'elevasi']]
y = df['tinggi_genangan']

model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X, y)

# 4. Simpan Model
joblib.dump(model, 'model_banjir.pkl')
print("Model berhasil dilatih dan disimpan sebagai 'model_banjir.pkl'")