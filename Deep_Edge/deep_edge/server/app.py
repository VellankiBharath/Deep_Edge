import os
import torch
from flask import Flask, request, jsonify, send_from_directory
from diffusers import StableDiffusionPipeline
from segment_anything import build_sam, SamPredictor, sam_model_registry
import clip
import cv2
import numpy as np
import base64
import io
from flask_cors import CORS

app = Flask(__name__, static_folder='../client', static_url_path='/client')
CORS(app)  # Enable CORS for all routes


# Paths to model files
MODEL_FOLDER = os.path.join(os.path.dirname(__file__), '../model/')
SAM_CHECKPOINT_PATH = os.path.join(MODEL_FOLDER, 'sam_vit_h_4b8939.pth')
STABLE_DIFFUSION_MODEL = "CompVis/stable-diffusion-v1-4"

# Initialize models
device = "cuda" if torch.cuda.is_available() else "cpu"
stable_diffusion_pipe = StableDiffusionPipeline.from_pretrained(STABLE_DIFFUSION_MODEL).to(device)

# Helper function to convert image to base64
def image_to_base64(image):
    buffered = io.BytesIO()
    image.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode("utf-8")
# Serve the HTML page
@app.route('/')
def index():
    return send_from_directory(app.static_folder, 'index.html')

@app.route('/generate', methods=['POST'])
def generate_image_with_clip_and_sam():
    pass
    # Get the text prompt from the request
    data = request.get_json()
    prompt = data.get('prompt', 'a beautiful landscape with mountains')  # Default prompt if none provided

    try:
        # Step 1: Generate the image using Stable Diffusion
        generated_image = stable_diffusion_pipe(prompt).images[0]
        image_base64 = image_to_base64(generated_image)

        # Step 2: Run CLIP analysis after generating the image
        clip_model, preprocess = clip.load("ViT-B/32", device=device)
        image_clip = preprocess(generated_image).unsqueeze(0).to(device)
        text_prompts = ["landscape", "mountains", "sunset", "river", "sky"]
        text_inputs = clip.tokenize(text_prompts).to(device)

        # Perform CLIP analysis
        with torch.no_grad():
            image_features = clip_model.encode_image(image_clip)
            text_features = clip_model.encode_text(text_inputs)
            logits_per_image = image_features @ text_features.T
            probs = logits_per_image.softmax(dim=-1).cpu().numpy()

        # Collect CLIP results
        clip_results = {text: float(probs[0][i]) for i, text in enumerate(text_prompts)}

        # Step 3: Run SAM segmentation after CLIP analysis
        sam_model = sam_model_registry["vit_h"](checkpoint=SAM_CHECKPOINT_PATH).to(device)
        sam_predictor = SamPredictor(sam_model)

        # Convert image to format required by SAM
        sam_image = cv2.cvtColor(np.array(generated_image), cv2.COLOR_RGB2BGR)
        sam_predictor.set_image(sam_image)

        # Use a point for segmentation
        input_point = np.array([[500, 375]])
        input_label = np.array([1])

        # Predict masks with SAM
        masks, scores, logits = sam_predictor.predict(
            point_coords=input_point,
            point_labels=input_label,
            multimask_output=True
        )

        # Convert masks and scores to JSON format
        masks_list = [mask.tolist() for mask in masks]
        scores_list = scores.tolist()

        # Return all results (image, CLIP, SAM)
        return jsonify({
            "prompt": prompt,
            "generated_image": image_base64,
            "clip_results": clip_results,
            "sam_masks": masks_list,
            "sam_scores": scores_list
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)

