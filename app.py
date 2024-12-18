import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import load_model
import numpy as np
from PIL import Image
import cv2

# Fungsi untuk menghapus latar belakang
def remove_background_opencv(img):
    img = np.array(img)  # Convert PIL image to numpy array
    if len(img.shape) == 3 and img.shape[2] == 3:  # Ensure RGB
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        _, mask = cv2.threshold(gray, 240, 255, cv2.THRESH_BINARY_INV)
        mask = mask.astype(np.uint8)
        bg_removed = cv2.bitwise_and(img, img, mask=mask)
        return bg_removed
    else:
        st.error("Error: Uploaded image is not in RGB format. Please upload a valid image.")
        return None

# Load model dan daftar kelas
model = load_model('asl_model.h5')
class_names = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9',
               'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j',
               'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't',
               'u', 'v', 'w', 'x', 'y', 'z']

# Bagian UI untuk judul dan pengunggahan gambar
st.title('American Sign Language (ASL) Prediction')
st.write("Upload an image of an ASL sign and get the predicted letter!")
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    img = Image.open(uploaded_file)

    # Menampilkan gambar asli
    st.subheader("Original Uploaded Image")
    st.image(img, caption='Uploaded Image', use_container_width=True)

    # Proses remove background
    st.write("Processing: Removing background from the image...")
    img_no_bg = remove_background_opencv(img)

    if img_no_bg is not None:
        # Menampilkan gambar tanpa latar belakang
        st.subheader("Image After Background Removal")
        st.image(img_no_bg, caption='Image with Background Removed', use_column_width=True)

        # Button untuk melanjutkan prediksi setelah background removal
        if st.button("Predict ASL Sign"):
            # Preprocessing dari gambar dan prediksi
            img_resized = cv2.resize(img_no_bg, (128, 128))
            img_array = img_resized / 255.0
            img_array = np.expand_dims(img_array, axis=0)

            # Prediksi menggunakan model
            prediction = model.predict(img_array)
            predicted_class = np.argmax(prediction, axis=1)

            predicted_letter = class_names[predicted_class[0]]
            st.markdown(f"<h3 style='text-align: center; color: blue;'>Predicted class: {predicted_letter}</h3>",
                        unsafe_allow_html=True)
    else:
        st.error("Failed to process the image. Please upload another image.")
