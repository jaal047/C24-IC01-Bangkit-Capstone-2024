# Image Classification API with FastAPI

This repository contains a FastAPI application for classifying dental images using pre-trained EfficientNet models. The app supports three views: `depan` (front), `atas` (top), and `bawah` (bottom), and saves the classification results to Google Cloud Firestore and uploads the images to Google Cloud Storage.

## Requirements

- Python 3.7+
- FastAPI
- Uvicorn
- TensorFlow
- Google Cloud Firestore
- Google Cloud Storage

## Setup

1. **Clone the repository:**
   ```sh
   git clone https://github.com/pt-periksa-gigi-indonesia/C24IC-Bangkit-Capstone-2024/tree/main/IC-01.git
   cd C24IC-Bangkit-Capstone-2024/tree/main/IC-01
   ```

2. **Install dependencies:**
   ```sh
   pip install -r requirements.txt
   ```

3. **Set up Google Cloud credentials:**
   - Ensure you have a service account with the necessary permissions for Firestore and Storage.
   - Set the `GOOGLE_APPLICATION_CREDENTIALS` environment variable to the path of your service account key file.

   ```sh
   export GOOGLE_APPLICATION_CREDENTIALS="/path/to/your/service-account-file.json"
   ```

4. **Run the application:**
   ```sh
   uvicorn main:app --host 0.0.0.0 --port 8000
   ```

## API Endpoints

### Predict Image Classification

- **Endpoint:** `POST /predict/{view}`
- **Description:** Classifies an uploaded image using the specified view's model.
- **Parameters:**
  - `view` (path parameter): One of `depan`, `atas`, `bawah`.
  - `file` (form data): The image file to be classified.
- **Response:**
  - `prediction`: The predicted label of the image.
  - `class_probability`: The probability of the predicted class.

```sh
curl -X 'POST' \
  'http://localhost:8000/predict/depan' \
  -H 'accept: application/json' \
  -H 'Content-Type: multipart/form-data' \
  -F 'file=@/path/to/your/image.jpg'
```

### Get Predictions

- **Endpoint:** `GET /predictions`
- **Description:** Retrieves the classification results from Firestore with pagination.
- **Parameters:**
  - `limit` (query parameter): Maximum number of predictions to return (default: 10).
  - `skip` (query parameter): Number of predictions to skip (for pagination) (default: 0).
- **Response:** A list of prediction records.

```sh
curl -X 'GET' \
  'http://localhost:8000/predictions?limit=10&skip=0' \
  -H 'accept: application/json'
```

## File Structure

- `app.py`: The main FastAPI application file.
- `requirements.txt`: Python dependencies.

## Additional Information

### Image Preprocessing

The uploaded images are preprocessed to match the input requirements of the EfficientNet models. The images are resized to 224x224 pixels and normalized.

### Cloud Storage

Images are uploaded to a specified Google Cloud Storage bucket. The bucket name is hardcoded in the script and should be replaced with your actual bucket name.

### Firestore

Classification results are stored in a Firestore collection. The collection name is hardcoded in the script and should be replaced with your actual collection name.

### CORS

CORS is enabled for all origins to allow interaction from any frontend application.

Please replace `your-bucket-name` and `your-collection-name` with the actual bucket name and Firestore collection name respectively.
