import os
import torch
import pickle
import json
import logging
from flask import Flask, request, jsonify
from diffusers import StableDiffusionPipeline
import clip
from segment_anything import SamPredictor, sam_model_registry
import base64
import io
import signal
from datetime import datetime

app = Flask(__name__)

# Configure logging
logging.basicConfig(filename='app.log', level=logging.INFO, format='%(asctime)s %(message)s')

# Paths to the model and data files
MODEL_FOLDER = os.path.join(os.path.dirname(__file__), '../model/')
PICKLE_PATH = os.path.join(MODEL_FOLDER, 'stble_clif_sam2.pickle')
DATA_JSON_PATH = os.path.join(MODEL_FOLDER, 'stble_clif_sam2.json')
SAM_CHECKPOINT = os.path.join(MODEL_FOLDER, 'checkpoints/sam_vit_h_4b8939.pth')

# Load the machine learning model from pickle
with open(PICKLE_PATH, 'rb') as model_file:
    model_data = pickle.load(model_file)

# Load data from JSON file
with open(DATA_JSON_PATH, 'r') as data_file:
    data_json = json.load(data_file)

# Initialize the models
device = "cuda" if torch.cuda.is_available() else "cpu"
stable_diffusion_pipe = StableDiffusionPipeline.from_pretrained("CompVis/stable-diffusion-v1-4").to(device)
clip_model, preprocess = clip.load("ViT-B/32", device=device)
sam_model = sam_model_registry["vit_h"](checkpoint=SAM_CHECKPOINT).to(device)
sam_predictor = SamPredictor(sam_model)


# Timeout handler
def timeout_handler(signum, frame):
    raise TimeoutError("The request took too long to process.")


# Helper function to convert image to base64
def image_to_base64(image):
    buffered = io.BytesIO()
    image.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode("utf-8")


@app.route('/')
def index():
    return "Welcome to the Deep Edge API!"


@app.route('/generate', methods=['POST'])
def generate_image():
    # Log the start time
    start_time = datetime.now()

    # Set timeout
    signal.signal(signal.SIGALRM, timeout_handler)
    signal.alarm(30)  # 30 seconds timeout

    # Validate input
    if not request.is_json:
        logging.error("Invalid input format. JSON expected.")
        return jsonify({"error": "Invalid input format. JSON expected."}), 400

    data = request.get_json()
    if 'prompt' not in data or not isinstance(data['prompt'], str):
        logging.error("'prompt' field is required and must be a string.")
        return jsonify({"error": "'prompt' field is required and must be a string."}), 400

    prompt = data['prompt']

    try:
        # Log the request
        logging.info(f"Received request with prompt: {prompt}")

        # Generate image using Stable Diffusion
        image = stable_diffusion_pipe(prompt).images[0]

        # Preprocess and generate response...

        signal.alarm(0)  # Disable timeout after success
    except TimeoutError:
        logging.error("Request timed out.")
        return jsonify({"error": "Request timed out."}), 408
    except FileNotFoundError as e:
        logging.error(f"Model file not found: {str(e)}")
        return jsonify({"error": "Model file not found."}), 500
    except torch.cuda.OutOfMemoryError:
        logging.error("CUDA out of memory.")
        return jsonify({"error": "CUDA out of memory."}), 500
