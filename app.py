from flask import Flask, request, render_template
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from PIL import Image
import numpy as np
import base64
from io import BytesIO

app = Flask(__name__)
model = load_model("model/skin_cancer.h5")


# Label dan Deskripsi
labels = {
    'melanoma': (
        "Melanoma",
        "Melanoma adalah jenis kanker kulit yang paling berbahaya dan dapat menyebar jika tidak ditangani. "
        "Biasanya muncul sebagai tahi lalat baru atau perubahan pada tahi lalat lama."
    ),
    'bkl': (
        "BKL (Benign Keratosis-like Lesions)",
        "BKL adalah lesi jinak pada kulit seperti lentigo, keratosis seboroik, dan lainnya. "
        "Meskipun jinak, kadang terlihat mirip kanker kulit dan perlu evaluasi."
    ),
    'nv': (
        "NV (Melanocytic Nevi)",
        "Nevus melanocytic atau tahi lalat adalah pertumbuhan jinak dari sel penghasil pigmen. "
        "Biasanya aman, tapi perlu diperiksa jika ada perubahan warna, bentuk, atau ukuran."
    )
}

# =========================
# Fungsi Validasi Warna Kulit
# =========================
def is_probably_skin(img):
    img_np = np.array(img.resize((224, 224)))
    avg_color = img_np.mean(axis=(0, 1))  # [R, G, B]
    r, g, b = avg_color

    # Estimasi kasar rentang warna kulit manusia
    if r > 90 and g > 40 and b > 20 and max(r, g, b) - min(r, g, b) < 80:
        return True
    return False

# =========================
# ROUTES
# =========================
@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")

@app.route("/upload", methods=["GET"])
def upload():
    return render_template("upload.html")

@app.route("/predict", methods=["POST"])
def predict():
    file = request.files.get("image")
    if not file:
        return "Tidak ada file dikirim", 400

    try:
        img = Image.open(file.stream).convert("RGB")
    except Exception as e:
        return f"Gagal membuka gambar: {e}", 400

    # Validasi gambar apakah terlihat seperti kulit
    if not is_probably_skin(img):
        return render_template("result.html", 
                               prediction="Gambar Tidak Valid",
                               description="❌ Gambar yang diunggah tidak dikenali sebagai gambar kulit.",
                               confidence="N/A",
                               image_data=None)

    # Prediksi kanker kulit
    img_resized = img.resize((224, 224))
    img_array = img_to_array(img_resized) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    prediction = model.predict(img_array)[0]
    predicted_key = list(labels.keys())[np.argmax(prediction)]
    label_name, description = labels[predicted_key]
    confidence = round(100 * np.max(prediction), 2)

    # Encode gambar sebagai base64 untuk ditampilkan
    buffered = BytesIO()
    img_resized.save(buffered, format="JPEG")
    img_str = base64.b64encode(buffered.getvalue()).decode()

    return render_template(
        "result.html",
        prediction=label_name,
        description=description,
        confidence=confidence,
        image_data=img_str
    )

if __name__ == "__main__":
    app.run(debug=True)
