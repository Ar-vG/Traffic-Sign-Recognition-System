import os
from flask import Flask, request, jsonify, render_template
import tensorflow as tf
import numpy as np
import cv2

# load model once at startup
MODEL_PATH = os.path.join(os.getcwd(), "traffic_model.keras")
model = tf.keras.models.load_model(MODEL_PATH)

# label map (same as in recognition.py)
label_cod = {
    0: 'Speed limit (20km/h)', 1: 'Speed limit (30km/h)', 2: 'Speed limit (50km/h)',
    3: 'Speed limit (60km/h)', 4: 'Speed limit (70km/h)', 5: 'Speed limit (80km/h)',
    6: 'End of speed limit (80km/h)', 7: 'Speed limit (100km/h)', 8: 'Speed limit (120km/h)',
    9: 'No passing', 10: 'No passing veh over 3.5 tons', 11: 'Right-of-way at intersection',
    12: 'Priority road', 13: 'Yield', 14: 'Stop', 15: 'No vehicles', 16: 'Veh > 3.5 tons prohibited',
    17: 'No entry', 18: 'General caution', 19: 'Dangerous curve left', 20: 'Dangerous curve right',
    21: 'Double curve', 22: 'Bumpy road', 23: 'Slippery road', 24: 'Road narrows on the right',
    25: 'Road work', 26: 'Traffic signals', 27: 'Pedestrians', 28: 'Children crossing',
    29: 'Bicycles crossing', 30: 'Beware of ice/snow', 31: 'Wild animals crossing',
    32: 'End speed + passing limits', 33: 'Turn right ahead', 34: 'Turn left ahead', 35: 'Ahead only',
    36: 'Go straight or right', 37: 'Go straight or left', 38: 'Keep right', 39: 'Keep left',
    40: 'Roundabout mandatory', 41: 'End of no passing', 42: 'End no passing veh > 3.5 tons'
}


def preprocess_image(img_bytes):
    """Convert raw image bytes from upload to model input."""
    arr = np.frombuffer(img_bytes, np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        return None
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (30, 30))
    img = img / 255.0
    return img


app = Flask(__name__)


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    if 'file' not in request.files:
        return jsonify(error="no file part"), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify(error="no selected file"), 400
    img = preprocess_image(file.read())
    if img is None:
        return jsonify(error="could not decode image"), 400
    pred = model.predict(np.expand_dims(img, 0))
    class_id = int(np.argmax(pred, axis=1)[0])
    label = label_cod.get(class_id, f"Unknown ({class_id})")
    return jsonify(class_id=class_id, label=label)


if __name__ == "__main__":
    # ensure template folder is correct when run from workspace root
    app.run(debug=True)
