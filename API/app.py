import uvicorn
from fastapi import FastAPI, File, UploadFile, Request
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.efficientnet import preprocess_input
import numpy as np
import io
from google.cloud import firestore, storage
import uuid
from datetime import datetime

# Initialize Firestore client
db = firestore.Client()

# Initialize Cloud Storage client
storage_client = storage.Client()

#Fastapi
app = FastAPI()

models = {
    "depan": load_model('EfficientNet_Tampak_Depan.h5'),
    "atas": load_model("EfficientNet_TampakAtas.h5"),
    "bawah": load_model('EfficientNet_TampakBawah_Fix_tf16.h5'),
}

# Define target class labels (replace with your actual labels)
labels = ['Bukan Gigi', 'Gigi berlubang', 'Gigi Sehat', 'Perubahan Warna Gigi', 'Radang Gusi']

# Function to preprocess the image (ensure compatibility with PIL format)
def preprocess_image(image_bytes):
    img = image.load_img(io.BytesIO(image_bytes), target_size=(224, 224))
    img_array = image.img_to_array(img)

    img_batch = np.expand_dims(img_array, axis=0)

    img_preprocessed = preprocess_input(img_batch)
    
    return img_preprocessed

# Single API endpoint for all views
@app.post("/predict/{view}", tags=["Predictions"])
async def predict(view: str, file: UploadFile = File(...)):
    # Check for valid image file
    if not file.content_type.startswith('image/'):
        return {"message": "Please upload an image file"}

    # Validate image size (less than 1 MB)
    if file.size > 1 * 1024 * 1024:  # 1 MB
        return {"message": "Image size cannot exceed 1 MB"}

    # Read image as bytes
    image_bytes = await file.read()

    # Preprocess the image
    preprocessed_image = preprocess_image(image_bytes)
    
    # Generate unique image name
    image_name = f"{uuid.uuid4()}.{file.content_type.split('/')[1]}"


    
    # Check for valid view and select corresponding model
    if view not in models:
        return {"message": "Invalid view. Supported views: depan, atas, bawah"}
    model = models[view]

    # Make prediction using the model
    prediction = model.predict(preprocessed_image)
    class_index = np.argmax(prediction)
    
    if labels[class_index] == 'Bukan Gigi':
        return "Input gambar bukan gambar gigi, tolong berikan input gambar yang benar."
    else :
        # Upload image to Cloud Storage
        bucket = storage_client.bucket("drsolution")  # Replace with your bucket name
        blob = bucket.blob(f'Data/{image_name}')
        blob.upload_from_string(image_bytes)

        # Prepare Firestore data
        image_url = f"gs://drsolution/Data/{image_name}"  # Replace with your bucket URL format
        
        prediction_data = {
            "view": view,
            "prediction": labels[class_index],
            "class_probability": f"{prediction[0][class_index]:.2f}",
            "image_url": image_url,
            "created_at": datetime.now(),  # Add server timestamp
        }

        # Save prediction to Firestore
        collection_ref = db.collection("predictions")  # Replace with your collection name
        collection_ref.add(prediction_data)



        return {"prediction": labels[class_index], "Class probability": f"{prediction[0][class_index]:.2f}"}

@app.get("/predictions", tags=["Predictions"])
async def get_predictions(limit: int = 10, skip: int = 0):
  """
  Retrieves predictions from Firestore with pagination options.

  Query parameters:
    - limit: Maximum number of predictions to return (default: 10).
    - skip: Number of predictions to skip (for pagination) (default: 0).
  """

  # Get predictions from Firestore
  collection_ref = db.collection("predictions")  # Replace with your collection name
  
#   docs = collection_ref.get()
  query = collection_ref.order_by("created_at").limit(limit)

  # Fetch data as a list of dictionaries
  predictions = [doc.to_dict() for doc in query.stream()]

  return predictions

# Enable CORS for all origins
@app.middleware("http")
async def add_cors_headers(request: Request, call_next):
    response = await call_next(request)
    response.headers["Access-Control-Allow-Origin"] = "*"
    response.headers["Access-Control-Allow-Headers"] = "Content-Type, Authorization"
    response.headers["Access-Control-Allow-Methods"] = "POST, OPTIONS"
    return response

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)  # Start the API server