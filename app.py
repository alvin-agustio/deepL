import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image
from rembg import remove
import io

# Muat model yang sudah dilatih
model = load_model('asl_model.h5')  # Sesuaikan dengan nama file model Anda

# Definisikan kelas yang digunakan dalam model (misalnya, huruf ASL)
class_names = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z']

# Judul aplikasi
st.title('American Sign Language (ASL) Prediction')

# Instruksi untuk pengguna
st.write("Upload an image of an ASL sign and get the predicted letter!")

# Membuat widget untuk upload gambar
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

# Fungsi untuk menghapus latar belakang gambar
def remove_background(image_data):
    input_image = Image.open(io.BytesIO(image_data))
    output_image = remove(input_image)  # Proses penghapusan latar belakang
    output_pil_image = Image.open(io.BytesIO(output_image))
    return output_pil_image

# Jika ada gambar yang diunggah, tampilkan dan prediksi
if uploaded_file is not None:
    # Baca gambar dan tampilkan
    img = Image.open(uploaded_file)
    st.image(img, caption='Uploaded Image', use_column_width=True)

    # Menghapus latar belakang gambar
    img_without_bg = remove_background(uploaded_file.read())
    st.image(img_without_bg, caption='Image without Background', use_column_width=True)

    # Preprocessing gambar yang diunggah tanpa latar belakang
    img_without_bg = img_without_bg.resize((128, 128))  # Sesuaikan dengan ukuran yang digunakan model Anda
    img_array = np.array(img_without_bg)
    img_array = img_array / 255.0  # Normalisasi
    img_array = np.expand_dims(img_array, axis=0)  # Tambahkan dimensi batch

    # Prediksi menggunakan model
    prediction = model.predict(img_array)
    predicted_class = np.argmax(prediction, axis=1)

    # Menampilkan hasil prediksi
    predicted_letter = class_names[predicted_class[0]]
    st.markdown(f"<h3 style='text-align: center; color: blue;'>Predicted class: {predicted_letter}</h3>", unsafe_allow_html=True)
