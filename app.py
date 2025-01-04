from flask import Flask, render_template, request
import io
import base64
from PIL import Image
from predict import predict


model_path = "lite_model.tflite" 

app = Flask(__name__)

@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None
    confidence = None
    img_base64 = None

    if request.method == "POST":
        if "image" not in request.files:
            return render_template("index.html", error="No file selected.")

        file = request.files["image"]
        if file.filename == "":
            return render_template("index.html", error="No file selected.")

        # Open the image and preprocess it
        image = Image.open(file).convert("RGB")
        img_resized = image.resize((224, 224))   
        prediction,confidence = predict(model_path,img_resized)

        # Convert the image to base64 to display on the webpage
        img_buffer = io.BytesIO()
        image.save(img_buffer, format="PNG")
        img_base64 = base64.b64encode(img_buffer.getvalue()).decode("utf-8")

    return render_template("index.html", prediction=prediction, confidence=confidence, img_base64=img_base64)

if __name__ == "__main__":
    app.run(host = "0.0.0.0",port=5000,debug=False)
