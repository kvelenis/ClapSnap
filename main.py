# Import necessary modules and classes from FastAPI and other libraries
from fastapi import FastAPI, HTTPException, Request, File, UploadFile, Form
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse, JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from starlette.responses import RedirectResponse
from sklearn.manifold import TSNE
import numpy as np
import os
import pandas as pd
import uuid
from datetime import datetime
from typing import List
import time
import torch
import librosa
from transformers import pipeline

# Define paths for uploading and static files
UPLOAD_DIR = os.path.join(os.path.dirname(__file__), "user_data")
STATIC_UPLOAD_DIR = os.path.join(os.path.dirname(__file__), "static_uploads")

# Create directories if they do not exist
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(STATIC_UPLOAD_DIR, exist_ok=True)

# Initialize FastAPI application
app = FastAPI()

# Initialize Jinja2Templates to render HTML templates
templates = Jinja2Templates(directory="templates")

# Mount static directories to serve static files and uploaded files
app.mount("/static", StaticFiles(directory="static"), name="static")
app.mount("/static_uploads", StaticFiles(directory=STATIC_UPLOAD_DIR), name="static_uploads")
app.mount("/user_data", StaticFiles(directory=UPLOAD_DIR), name="uploads")

def create_unique_folder():
    """
    Create a unique folder name based on the current timestamp and a random string.
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")  # Current timestamp
    random_string = str(uuid.uuid4())[:8]  # First 8 characters of a UUID for uniqueness
    return f"{timestamp}_{random_string}"  # Return formatted unique folder name

@app.get("/")
async def index(request: Request):
    """
    Handle GET requests to the root URL by rendering the index.html template.
    """
    return templates.TemplateResponse("index.html", {"request": request})

@app.get("/favicon.ico", include_in_schema=False)
async def favicon():
    """
    Handle GET requests for the favicon by redirecting to the favicon.ico file.
    """
    return RedirectResponse(url="/static/favicon.ico")

@app.post("/analyze_wav", response_class=JSONResponse)
async def analyze_wav(request: Request, wav_file: UploadFile = File(...), candidate_labels: str = Form(...)):
    """
    Handle POST requests to analyze a WAV file and classify audio segments.
    
    Parameters:
    - wav_file: The uploaded WAV file.
    - candidate_labels: Comma-separated string of labels for classification.
    
    Returns:
    - JSONResponse with classification scores over time.
    """
    # Create a unique folder to store the uploaded file
    unique_folder = create_unique_folder()
    folder_path = os.path.join(UPLOAD_DIR, unique_folder)
    os.makedirs(folder_path)  # Create the directory

    # Define paths for saving the uploaded WAV file
    input_wav_dir = os.path.join(folder_path, "input_wav")
    os.makedirs(input_wav_dir)  # Create the directory
    input_wav_path = os.path.join(input_wav_dir, wav_file.filename)

    # Save the uploaded WAV file to the server
    with open(input_wav_path, "wb") as buffer:
        buffer.write(await wav_file.read())

    # Convert the candidate labels from a comma-separated string to a list
    candidate_labels_list = [label.strip() for label in candidate_labels.split(",")]

    # Load the audio file and convert it to mono
    audio, sr = librosa.load(input_wav_path, sr=None, mono=True)

    # Initialize the zero-shot audio classification pipeline
    audio_classifier = pipeline(task="zero-shot-audio-classification", model="laion/larger_clap_general")

    # Define parameters for audio segmentation
    window_size = 2 * sr  # 2-second window size
    overlap = 1 * sr  # 1-second overlap
    step = window_size - overlap  # Step size for sliding window

    # Initialize a dictionary to store classification scores
    scores_dict = {label: [] for label in candidate_labels_list}
    time_points = []

    # Iterate through the audio with overlapping windows
    for start in range(0, len(audio) - window_size + 1, step):
        window = audio[start:start + window_size]  # Extract the audio segment
        output = audio_classifier(window, candidate_labels=candidate_labels_list)  # Classify the segment
        
        # Store classification scores in the dictionary
        for result in output:
            scores_dict[result['label']].append(result['score'])
        
        # Store the time point corresponding to the start of the window
        time_points.append(start / sr)

    # Prepare the plot data for visualization
    plot_data = {}
    for label in scores_dict:
        plot_data[label] = {
            "time_points": time_points,  # Time points of each segment
            "scores": scores_dict[label]  # Classification scores for each label
        }

    # Return the plot data as a JSON response
    return JSONResponse(content=plot_data)

# Run the application with Uvicorn if this script is executed directly
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=5002)
