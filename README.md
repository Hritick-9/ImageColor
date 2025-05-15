# Image Enhancement API with Android Client

This repository contains a Flask-based API server that provides image enhancement capabilities through a machine learning model, along with an Android client application for easy access.

## Project Structure

```
.
├── app.py              # Flask server implementation
├── model.pkl           # Trained ML model for image enhancement
├── Procfile           # Configuration for deployment
├── requirements.txt    # Python dependencies

```

## Server Implementation

The server is built using Flask and provides a REST API endpoint for image enhancement. It uses a pre-trained machine learning model to process grayscale images and enhance them.

### Features

- Image enhancement API endpoint
- Health check endpoint
- Automatic model loading
- Image preprocessing and postprocessing
- Support for grayscale to RGB conversion

### API Endpoints

1. **Image Enhancement**
   - Endpoint: `/predict`
   - Method: POST
   - Input: Image file in the request body
   - Response: Colored image in PNG format

### Technical Details

- Input image size: 128x128 pixels
- Output: Enhanced RGB image
- Model: Custom ML model (stored in model.pkl)
- Framework: Flask with TensorFlow/Keras backend

## Setup and Installation

1. Clone the repository:
   ```bash
   git clone <repository-url>
   cd <repository-name>
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Run the server:
   ```bash
   python app.py
   ```
   The server will start on port 10000 by default.

## Deployment

The application is configured for deployment on platforms like Render using the provided Procfile. The server uses Gunicorn as the WSGI server.

## Android Client

The Android application provides a user-friendly interface to interact with the image enhancement API. Users can:

- Select images from their device
- Send images to the server for enhancement
- View and save enhanced images
- Manage their enhanced image gallery

## Requirements

### Server Requirements
- Python 3.x
- Flask
- TensorFlow
- Keras
- NumPy
- Pillow
- Gunicorn
- Joblib

### Android Requirements
- Android 5.0 (API level 21) or higher
- Internet permission
- Storage permission
link :-https://drive.google.com/drive/folders/1_cJPe52_3qidQBxI4CYKpU6CL-s6705q
## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.



 