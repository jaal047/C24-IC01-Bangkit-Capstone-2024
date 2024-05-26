from flask import Flask, request, jsonify
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from werkzeug.datastructures import FileStorage
import tempfile
from flask_restful import reqparse

app = Flask(__name__)

# Load your pre-trained model (replace with your model path)
model = load_model('IC-01\API\\tampak_depan2.h5')
model2 = load_model('IC-01\API\\tampak_depan2.h5')
model3 = load_model('IC-01\API\\tampak_depan2.h5')

# Define target class labels (replace with your actual labels)
labels = ['Bukan Gigi', 'Gigi berlubang', 'Gigi Sehat', 'Perubahan Warna Gigi', 'Radang Gusi']

def preprocess_image(image_data):
  # Replace with your specific preprocessing steps
  img = image.load_img(image_data, target_size=(128, 128))  # Example resizing
  x = image.img_to_array(img)
  x = np.expand_dims(x, axis=0)  # Add batch dimension
  x = x / 255.0  # Normalize
  return x

parser = reqparse.RequestParser()
parser.add_argument('file',
                    type=FileStorage,
                    location='files',
                    required=True,
                    help='provide a file')

@app.route('/predict/tampak-depan', methods=['POST'])
def predict():
  # Check if an image is included in the request
    args = parser.parse_args()
    the_file = args['file']
    # save a temporary copy of the file
    ofile, ofname = tempfile.mkstemp()
    the_file.save(ofname)
    

    # Preprocess the image
    preprocessed_image = preprocess_image(ofname)

    # Make prediction using the model
    prediction = model.predict(preprocessed_image)
    class_index = np.argmax(prediction)

    # Return the predicted class label
    return jsonify({'prediction': labels[class_index],
                    'Class probability': f'{prediction[0][class_index]:.2f}'})

@app.route('/predict/tampak-atas', methods=['POST'])
def predict():
  # Check if an image is included in the request
    args = parser.parse_args()
    the_file = args['file']
    # save a temporary copy of the file
    ofile, ofname = tempfile.mkstemp()
    the_file.save(ofname)
    

    # Preprocess the image
    preprocessed_image = preprocess_image(ofname)

    # Make prediction using the model
    prediction = model2.predict(preprocessed_image)
    class_index = np.argmax(prediction)

    # Return the predicted class label
    return jsonify({'prediction': labels[class_index],
                    'Class probability': f'{prediction[0][class_index]:.2f}'})


@app.route('/predict/tampak-bawah', methods=['POST'])
def predict():
  # Check if an image is included in the request
    args = parser.parse_args()
    the_file = args['file']
    # save a temporary copy of the file
    ofile, ofname = tempfile.mkstemp()
    the_file.save(ofname)
    

    # Preprocess the image
    preprocessed_image = preprocess_image(ofname)

    # Make prediction using the model
    prediction = model3.predict(preprocessed_image)
    class_index = np.argmax(prediction)

    # Return the predicted class label
    return jsonify({'prediction': labels[class_index],
                    'Class probability': f'{prediction[0][class_index]:.2f}'})
if __name__ == '__main__':
  app.run(debug=True)