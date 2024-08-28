import ast
import pathlib
from io import BytesIO
from flask import Flask, request, jsonify
import tensorflow as tf


TRAINED_MODEL_PATH = pathlib.Path(f'models/model_2024-08-27_12-07-58').resolve()
TRAINED_MODEL_CLASSES_PATH = pathlib.Path(TRAINED_MODEL_PATH / 'classes.txt').resolve()

with open(TRAINED_MODEL_CLASSES_PATH, 'r') as f:
    content = f.read()
    TRAINED_MODEL_CLASSES = ast.literal_eval(content)

app = Flask(__name__)
model = tf.keras.models.load_model(TRAINED_MODEL_PATH)

@app.route('/v1/predict', methods=['POST'])
def predict():
    file = request.files['file']
    if not file:
        return 'No file uploaded', 400

    try:
        image = tf.keras.preprocessing.image.load_img(
            BytesIO(file.read()),
            target_size=(256, 256),
            color_mode='grayscale'
        )
        image_array = tf.keras.preprocessing.image.img_to_array(image)
        image_array = tf.expand_dims(image_array, axis=0)

        prediction = model.predict(image_array)
        prediction_dict = {class_name: float(pred) for class_name, pred in zip(TRAINED_MODEL_CLASSES, prediction[0])}

        return jsonify(prediction_dict)

    except Exception as e:
        return str(e), 500

if __name__ == '__main__':
    app.run(host='localhost', port=5000)
