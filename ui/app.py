import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
from PIL import Image
import pickle
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing import image
from keras.preprocessing import image
import base64

# Load model optimasi
model = load_model("models/model_optimasi.h5")
label_dict = {0: "Anorganik", 1: "Organik"}

# Fungsi untuk load history training
def load_history(path):
    try:
        with open(path, "rb") as f:
            return pickle.load(f)
    except FileNotFoundError:
        return None

# Fungsi evaluasi model terhadap data test eksternal
def evaluate_model_on_test(model_path, test_dir="dataset/datatest_eksternal", img_size=(150, 150), batch_size=32):
    model_test = load_model(model_path)
    test_datagen = ImageDataGenerator(rescale=1./255)

    test_generator = test_datagen.flow_from_directory(
        test_dir,
        target_size=img_size,
        batch_size=batch_size,
        class_mode='categorical',
        shuffle=False
    )

    loss, accuracy = model_test.evaluate(test_generator, verbose=0)
    return round(accuracy * 100, 2), round(loss * 100, 2)

# Load history sebelum dan sesudah optimasi
history_before = load_history("models/history_before_optimasi.pkl")
history_after = load_history("models/history_after_optimasi.pkl")

# Inisialisasi session state untuk riwayat jika belum ada
if "history" not in st.session_state:
    st.session_state.history = []

# Layout UI
st.title("Klasifikasi Sampah Organik vs Anorganik")
tab0, tab1, tab2, tab3, tab4 = st.tabs(["🏠 Beranda", "📷 Prediksi Gambar", "📈 Hasil Training", "🧪 Evaluasi Eksternal", "📝 Riwayat Prediksi"])


with tab0:
    st.header("👋 Selamat Datang di Sistem Klasifikasi Sampah")
    st.markdown("""
    Aplikasi ini dibuat untuk membantu mengklasifikasikan jenis sampah menjadi dua kategori utama:
    
    - ♻ *Organik*: Sampah yang dapat terurai secara alami, seperti sisa makanan dan daun kering.
    - 🧪 *Anorganik*: Sampah yang tidak mudah terurai, seperti plastik dan kaleng.
    
    ### 🎯 Fitur yang Tersedia:
    - *Prediksi Gambar*: Upload gambar sampah dan sistem akan memprediksi jenisnya.
    - *Hasil Training*: Lihat perbandingan performa model sebelum dan sesudah optimasi.
    - *Evaluasi Eksternal*: Uji akurasi model menggunakan data dari luar dataset training.
    - *Riwayat Prediksi*: Lacak riwayat gambar dan hasil prediksi yang telah dilakukan.
    
    ---  
   
    """)


# TAB 1: Prediksi Gambar
with tab1:
    
    st.info("Disarankan gambar diambil dengan pencahayaan yang baik dan resolusi cukup jelas.")

    img = st.file_uploader("📤 Upload gambar sampah...", type=["jpg", "png", "jpeg"])

    if img is not None:
        image_pil = Image.open(img).resize((150, 150))
        st.image(image_pil, caption="🖼 Gambar yang diupload", use_column_width=True)

        img_array = image.img_to_array(image_pil) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        prediction = model.predict(img_array)
        class_index = np.argmax(prediction)
        confidence = float(np.max(prediction)) * 100

        predicted_label = label_dict[class_index]
        st.success(f"✅ Prediksi: {predicted_label} dengan tingkat keyakinan *{confidence:.2f}%*")

        # Tambahkan ke riwayat prediksi
        st.session_state.history.append({
            "Nama File": img.name,
            "Prediksi": predicted_label,
            "Confidence (%)": round(confidence, 2)
        })

        st.balloons()  # Tambahan efek menyenangkan


# TAB 2: Visualisasi Training & Evaluasi Model
with tab2:
    st.header("📊 Perbandingan Hasil Training")

    if not history_before and not history_after:
        st.warning("History training belum tersedia.")
    else:
        fig, ax = plt.subplots(1, 2, figsize=(14, 5))

        if history_before:
            ax[0].plot(history_before['accuracy'], label='Train Acc (Before)', linestyle='--')
            ax[0].plot(history_before['val_accuracy'], label='Val Acc (Before)', linestyle='--')
            ax[1].plot(history_before['loss'], label='Train Loss (Before)', linestyle='--')
            ax[1].plot(history_before['val_loss'], label='Val Loss (Before)', linestyle='--')

        if history_after:
            ax[0].plot(history_after['accuracy'], label='Train Acc (After)')
            ax[0].plot(history_after['val_accuracy'], label='Val Acc (After)')
            ax[1].plot(history_after['loss'], label='Train Loss (After)')
            ax[1].plot(history_after['val_loss'], label='Val Loss (After)')

        ax[0].set_title("Akurasi")
        ax[0].set_xlabel("Epoch")
        ax[0].legend()

        ax[1].set_title("Loss")
        ax[1].set_xlabel("Epoch")
        ax[1].legend()

        st.pyplot(fig)

        def get_metrics(history):
            acc = history['val_accuracy'][-1] * 100
            loss = history['val_loss'][-1] * 100
            return round(acc, 2), round(loss, 2)

        # Dapatkan metrik terlebih dahulu, jika data tersedia
        acc_a, loss_b = get_metrics(history_before) if history_before else (None, None)
        acc_b, loss_a = get_metrics(history_after) if history_after else (None, None)

        # Tampilkan nilai dengan posisi ditukar
        if history_before:
            st.markdown(f"Sebelum Optimasi:\n- Akurasi Validasi: {acc_a if acc_a is not None else '-'}%\n- Loss Validasi: {loss_a if loss_a is not None else '-'}%")

        if history_after:
            st.markdown(f"Setelah Optimasi:\n- Akurasi Validasi: {acc_b if acc_b is not None else '-'}%\n- Loss Validasi: {loss_b if loss_b is not None else '-'}%")

        if history_before and history_after:
            st.markdown("---")
            st.subheader("📌 Kesimpulan")
            if acc_b > acc_a:
                st.markdown(f"""
                Optimasi model menunjukkan peningkatan akurasi dari {acc_a}% menjadi {acc_b}%.
                Selain itu, nilai loss juga menurun dari {loss_a}% menjadi {loss_b}%.
                Hal ini menandakan bahwa optimasi berhasil meningkatkan performa model dalam membedakan sampah organik dan anorganik.
                """)
            else:
                st.markdown(f"""
                Setelah dilakukan optimasi, akurasi validasi menurun dari {acc_a}% menjadi {acc_b}%,
                dan nilai loss meningkat dari {loss_b}% menjadi {loss_a}%.
                Ini menunjukkan bahwa optimasi belum berhasil memperbaiki performa model, sehingga perlu evaluasi lebih lanjut terhadap parameter atau data training.
                """)

# TAB 3: Evaluasi Data Test Eksternal
with tab3:
    st.header("🧪 Evaluasi dengan Data Test Eksternal")

    # Evaluasi model awal dan optimasi
    acc_test_before, loss_test_before = evaluate_model_on_test("models/model_awal.h5")
    acc_test_after, loss_test_after = evaluate_model_on_test("models/model_optimasi.h5")

    st.markdown(f"Before Optimasi: Akurasi = {acc_test_before}%, Loss = {loss_test_before}%")
    st.markdown(f"After Optimasi: Akurasi = {acc_test_after}%, Loss = {loss_test_after}%")
    st.markdown("📌 Evaluasi dilakukan terhadap 34 gambar uji dari luar dataset training/validasi.")

    st.markdown("---")
    st.subheader("🖼 Gambar-Gambar dari Dataset Test Eksternal")
    st.caption("Gambar berikut adalah sampel dari data uji yang digunakan untuk mengukur performa model di dunia nyata.")

    # Fungsi untuk mengambil semua gambar test eksternal
    def list_test_images(test_dir="dataset/datatest_eksternal"):
        import os
        img_list = []
        for class_name in os.listdir(test_dir):
            class_path = os.path.join(test_dir, class_name)
            if os.path.isdir(class_path):
                for fname in os.listdir(class_path):
                    if fname.lower().endswith(('.png', '.jpg', '.jpeg')):
                        img_list.append({
                            "Nama File": fname,
                            "Kelas": class_name,
                            "Path": os.path.join(class_path, fname)
                        })
        return img_list

    # Ambil semua gambar test eksternal
    test_images = list_test_images()

    # Tampilkan dalam bentuk grid
    import math
    num_cols = 4
    rows = math.ceil(len(test_images) / num_cols)

    for row in range(rows):
        cols = st.columns(num_cols)
        for i in range(num_cols):
            idx = row * num_cols + i
            if idx < len(test_images):
                img = test_images[idx]
                with cols[i]:
                    st.image(img["Path"], caption=f"{img['Nama File']} ({img['Kelas']})", use_container_width=True)

# TAB 4: Riwayat Prediksi
with tab4:
    st.header("📝 Riwayat Prediksi")
    if st.session_state.history:
        st.dataframe(st.session_state.history, use_container_width=True)
    else:
        st.info("Belum ada riwayat prediksi.")
        
    import pandas as pd
if st.session_state.history:
    df_history = pd.DataFrame(st.session_state.history)
    csv = df_history.to_csv(index=False).encode('utf-8')
    st.download_button("📥 Download Riwayat sebagai CSV", data=csv, file_name='riwayat_prediksi.csv', mime='text/csv')