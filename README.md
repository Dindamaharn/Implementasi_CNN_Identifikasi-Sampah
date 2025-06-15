## ♻️ Klasifikasi Sampah Organik & Anorganik dengan CNN

Proyek ini adalah sistem klasifikasi citra sampah organik dan anorganik menggunakan Convolutional Neural Network (CNN) dan antarmuka web berbasis Streamlit.

---

### 📁 Struktur Folder

```
.
├── dataset/
│   ├── original/               # Dataset asli (organik & anorganik)
│   ├── datasplit/              # Dataset hasil split (train/val/test)
│   └── datatest_eksternal/    # Dataset untuk uji eksternal
├── preprocessing_data/        # Script untuk split data
├── train/                     # Script pelatihan CNN
├── models/                    # Model .h5 hasil training
├── evaluasi_model/            # Script evaluasi model
├── results/                   # Grafik akurasi, loss, dll
├── ui/                        # Aplikasi Streamlit
├── venv/                      # Virtual environment (opsional)
└── README.md
```

---

### 🔗 1. Download Dataset

Silakan download dataset dari link berikut:

> [⬇️ Download Dataset Sampah (Google Drive)](https://drive.google.com/drive/folders/1bQ0ssWYgL2p_xkg6L5Ihl_UIWHfZfPT8?usp=drive_link)

Setelah diunduh:

1. Ekstrak zip ke folder `dataset/original/`
2. Pastikan struktur folder-nya seperti ini:

   ```
   dataset/original/
   ├── organik/
   └── anorganik/
   ```

---

### ⚙️ 2. Persiapan Awal (Setup)

#### A. **Clone repository (jika dari GitHub)**

```bash
git clone https://github.com/namakamu/proyek-klasifikasi-sampah.git
cd proyek-klasifikasi-sampah
```

#### B. **Buat Virtual Environment**

```bash
python -m venv venv
```

#### C. **Aktifkan Virtual Environment**

```bash
.\venv\Scripts\activate         # Windows
# atau
source venv/bin/activate       # Mac/Linux
```

#### D. **Install Dependencies**

```bash
pip install -r requirements.txt
```

> Jika `requirements.txt` belum ada, kamu bisa install manual:

```bash
pip install tensorflow streamlit matplotlib split-folders pillow
```

---

### 💠 3. Preprocessing: Split Dataset

Jalankan script untuk membagi dataset menjadi train/val/test:

```bash
python preprocessing_data/split_data.py
```

---

### 🤖 4. Training Model CNN

#### A. Training Sebelum Optimasi

```bash
python train/train_before_optimasi.py
```

#### B. Training Setelah Optimasi

```bash
python train/train_after_optimasi.py
```

Model akan disimpan ke folder `models/` seperti:

```
models/model_before.h5
models/model_optimasi.h5
```

---

### 📊 5. Evaluasi Model

Jalankan script evaluasi untuk menampilkan akurasi, loss, dan grafik:

```bash
python evaluasi_model/evaluasi_model.py
```

Hasil evaluasi disimpan di folder `results/`.

---

### 🌐 6. Jalankan Aplikasi Web

```bash
streamlit run ui/app.py
```

Aplikasi akan terbuka otomatis di browser, dan kamu bisa mengunggah gambar sampah untuk diprediksi.

---

### ✅ Fitur

* Klasifikasi gambar sampah: Organik atau Anorganik
* Model sebelum & sesudah optimasi
* Visualisasi hasil training
* Antarmuka pengguna (UI) berbasis Streamlit

---

### 📌 Catatan Tambahan

* Gunakan Python versi 3.8 - 3.10 untuk kompatibilitas TensorFlow.
* Jangan upload folder `venv/` atau dataset ke GitHub.
* Simpan model `.h5` dalam folder `models/`.

