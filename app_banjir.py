import numpy as np
import matplotlib.pyplot as plt
import joblib
import rasterio
from rasterio.plot import show
import os

# Nama file model dan peta
MODEL_FILE = 'model_banjir.pkl'
MAP_FILE = 'DEMNAS_1409-22_v1.0 (1).tif'

def load_resources():
    if not os.path.exists(MODEL_FILE):
        print(f"Error: {MODEL_FILE} tidak ditemukan. Jalankan latih_model.py dulu.")
        exit()
    if not os.path.exists(MAP_FILE):
        print(f"Error: {MAP_FILE} tidak ditemukan. Pastikan file TIF ada di folder ini.")
        exit()
    
    print("Memuat model dan data peta...", end=" ")
    model = joblib.load(MODEL_FILE)
    print("Siap!")
    return model

def get_input_user():
    print("\n" + "="*60)
    print("   PREDIKSI & PEMETAAN WILAYAH BANJIR ROB KOTA SEMARANG")
    print("="*60)
    
    print("\n[1] DATA CURAH HUJAN (mm/jam)")
    print("    - Rendah (Gerimis/Hujan Ringan) : 0 - 20")
    print("    - Sedang (Hujan Normal)         : 20 - 50")
    print("    - Tinggi (Hujan Lebat)          : 50 - 100")
    print("    - Ekstrem (Badai)               : > 100")
    hujan = float(input(">>> Masukkan angka curah hujan: "))

    print("\n[2] DATA TINGGI PASANG LAUT (cm)")
    print("    - Surut Terendah                : 80 - 120")
    print("    - Normal                        : 120 - 180")
    print("    - Pasang Tinggi (Rob)           : 180 - 250")
    print("    - Pasang Ekstrem                : > 250")
    pasang = float(input(">>> Masukkan angka tinggi pasang: "))

    print("\n[3] DATA PENURUNAN TANAH (cm/tahun)")
    print("    - Wilayah Stabil (Semarang Atas): 0 - 2")
    print("    - Wilayah Rawan (Genuk/Kaligawe): 7 - 13")
    subsidence = float(input(">>> Masukkan rata-rata penurunan tanah: "))

    return hujan, pasang, subsidence

def generate_flood_map(model, hujan, pasang, subsidence):
    print("\n[INFO] Sedang memproses peta DEMNAS (Ini mungkin memakan waktu)...")
    
    # Membuka file DEMNAS
    with rasterio.open(MAP_FILE) as src:
        # Membaca data ketinggian (Band 1)
        elevation_data = src.read(1)
        # Masking nilai nodata (biasanya -9999 atau sangat kecil) agar tidak diproses
        elevation_data = np.where(elevation_data < -100, np.nan, elevation_data)
        
        # Transformasi koordinat untuk plotting
        transform = src.transform
        
        # Persiapan Data untuk Prediksi Massal
        # Kita meratakan (flatten) array 2D peta menjadi 1D list untuk dimasukkan ke model
        rows, cols = elevation_data.shape
        flat_elev = elevation_data.flatten()
        
        # Buat array input yang ukurannya sama dengan jumlah piksel peta
        # Isi kolom hujan, pasang, subsidence dengan nilai input user (sama rata)
        input_matrix = np.zeros((len(flat_elev), 4))
        input_matrix[:, 0] = hujan       # Kolom 0: Hujan
        input_matrix[:, 1] = pasang      # Kolom 1: Pasang
        input_matrix[:, 2] = subsidence  # Kolom 2: Subsidence
        input_matrix[:, 3] = flat_elev   # Kolom 3: Elevasi (bervariasi sesuai peta)
        
        # Bersihkan NaN (area kosong di peta) sebelum prediksi
        valid_idx = ~np.isnan(flat_elev)
        
        # Wadah hasil prediksi
        flood_map_flat = np.zeros(len(flat_elev))
        
        # Lakukan prediksi hanya pada piksel yang valid (ada datanya)
        if np.any(valid_idx):
            preds = model.predict(input_matrix[valid_idx])
            flood_map_flat[valid_idx] = preds
            
        # Kembalikan ke bentuk peta asli (2D)
        flood_map = flood_map_flat.reshape(rows, cols)
        
        # Masking: Ubah nilai 0 (tidak banjir) menjadi transparan agar peta dasar terlihat
        flood_map_masked = np.ma.masked_where(flood_map <= 0, flood_map)

        # --- VISUALISASI ---
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # 1. Plot Peta Dasar (DEMNAS - Topografi)
        # Menggunakan colormap 'terrain' (hijau/coklat)
        show(src, ax=ax, cmap='terrain', alpha=0.8, title="Peta Prediksi Genangan Banjir Rob")
        
        # 2. Plot Layer Banjir di atasnya
        # Menggunakan colormap 'Blues' (Biru air)
        # Extent menyesuaikan koordinat geografis peta asli
        img_extent = [src.bounds.left, src.bounds.right, src.bounds.bottom, src.bounds.top]
        
        im = ax.imshow(flood_map_masked, cmap='bwr_r', vmin=0, vmax=100, alpha=0.7, extent=img_extent)
        
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('Tinggi Genangan Air (cm)')
        
        ax.set_xlabel('Longitude')
        ax.set_ylabel('Latitude')
        
        print("[INFO] Peta berhasil dibuat. Tampilkan window...")
        plt.show()

# EKSEKUSI UTAMA 
if __name__ == "__main__":
    model_rf = load_resources()
    h_input, p_input, s_input = get_input_user()
    
    print(f"\nMemproses skenario: Hujan {h_input}mm | Pasang {p_input}cm | Subsidence {s_input}cm/thn")
    generate_flood_map(model_rf, h_input, p_input, s_input)