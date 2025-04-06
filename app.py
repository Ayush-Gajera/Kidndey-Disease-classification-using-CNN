from flask import Flask, render_template, request
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
import os

app = Flask(__name__)

# Load your trained model
model = tf.keras.models.load_model('model/kidney_cnn_model.h5')

# Mapping prediction output
class_names = ['Cyst', 'Normal', 'Stone', 'Tumor']

# History list
history = []

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return "No file uploaded!", 400

    file = request.files['file']
    if file.filename == '':
        return "No file selected!", 400

    save_path = os.path.join('static', file.filename)
    file.save(save_path)

    # Preprocess the image
    img = image.load_img(save_path, target_size=(150, 150))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0

    # Predict
    predictions = model.predict(img_array)
    predicted_index = np.argmax(predictions[0])
    predicted_class = class_names[predicted_index]
    confidence = np.max(predictions[0]) * 100

    # Save only filename, not full path
    history.append({
        'img_filename': file.filename, 
        'prediction': predicted_class,
        'confidence': confidence
    })

    return render_template('index.html', 
                           prediction=predicted_class, 
                           confidence=round(confidence,2), 
                           img_path='static/' + file.filename)

@app.route('/history')
def show_history():
    return render_template('history.html', history=history)

@app.route('/about')
def about():
    return render_template('about.html')

if __name__ == '__main__':
    app.run(debug=True)
