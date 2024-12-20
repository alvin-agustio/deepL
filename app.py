import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# Fungsi untuk memuat model TFLite
def load_tflite_model(model_path):
    interpreter = tf.lite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()
    return interpreter

# Fungsi untuk melakukan prediksi menggunakan TFLite model
def predict_with_tflite(interpreter, img_array):
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    # Preprocessing gambar (normalisasi dan memastikan format yang benar)
    input_data = np.expand_dims(img_array, axis=0).astype(np.float32)
    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()

    # Ambil hasil prediksi
    output_data = interpreter.get_tensor(output_details[0]['index'])
    return output_data

# Muat model TFLite
interpreter = load_tflite_model('asl_model.tflite')

# Definisikan kelas yang digunakan dalam model (misalnya, huruf ASL)
class_names = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z']

# Judul aplikasi
st.title('American Sign Language (ASL) Prediction')

# Instruksi untuk pengguna
st.write("Upload an image of an ASL sign and get the predicted letter!")

# Membuat widget untuk upload gambar
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    # Baca gambar dan tampilkan
    img = Image.open(uploaded_file)
    st.image(img, caption='Uploaded Image', use_container_width=True)

    # Preprocessing gambar yang diunggah
    img = img.resize((128, 128))  # Sesuaikan dengan ukuran yang digunakan model Anda
    img_array = np.array(img)
    img_array = img_array / 255.0  # Normalisasi ke rentang 0-1

    # Cek dimensi gambar
    st.write(f"Gambar shape setelah preprocessing: {img_array.shape}")

    # Prediksi menggunakan model TFLite
    predictions = predict_with_tflite(interpreter, img_array)
    
    # Cek hasil output untuk debugging
    st.write(f"Predictions output shape: {predictions.shape}")
    
    # Ambil hasil kelas dengan probabilitas tertinggi
    predicted_class = np.argmax(predictions, axis=1)

    # Menampilkan hasil prediksi
    predicted_letter = class_names[predicted_class[0]]
    st.markdown(f"<h3 style='text-align: center; color: blue;'>Predicted class: {predicted_letter}</h3>", unsafe_allow_html=True)
