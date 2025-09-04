import streamlit as st
import requests
import numpy as np
from PIL import Image, ImageOps
import warnings
from transformers import ViTImageProcessor
import time

# --- 1. CONFIGURATION (from your notebook) ---

# Suppress insecure request warnings
warnings.filterwarnings("ignore", message="Unverified HTTPS request is being made.*")

# The full URL for your model's inference endpoint
INFERENCE_URL = 'https://famodel-sandbox.apps.rhoai.sandbox2386.opentlc.com/v2/models/famodel/versions/1/infer'

# Dictionary to map the model's output index to the age group label
#ID2LABEL = {0: '0-2', 1: '3-5', 2: '6-9', 3: '10-14', 4: '15-19', 5: '20-29', 6: '30-39', 7: '40-49', 8: '50-59', 9: '60-69', 10: '70+'}
ID2LABEL = {0: '01', 1: '02', 2: '03', 3: '04', 4: '05', 5: '06-07', 6: '08-09', 7: '10-12', 8: '13-15', 9: '16-20', 10: '21-25', 11: '26-30', 12: '31-35', 13: '36-40', 14: '41-45', 15: '46-50', 16: '51-55', 17: '56-60', 18: '61-65', 19: '66-70', 20: '71-80', 21: '81-90', 22: '90+'}

# Note: I've updated your id2label mapping to a more standard one based on the model card.
# If your fine-tuned model uses the 23-category mapping, please replace the dictionary above with the one from your notebook.

# The Hugging Face path for the required image processor
PROCESSOR_PATH = "dima806/facial_age_image_detection"

# --- 2. LOAD THE PROCESSOR (with caching) ---

# @st.cache_resource is a Streamlit decorator that runs this function only once,
# preventing the model from being reloaded every time you upload an image.
@st.cache_resource
def load_processor():
    """Loads the ViTImageProcessor from Hugging Face."""
    try:
        return ViTImageProcessor.from_pretrained(PROCESSOR_PATH)
    except Exception as e:
        st.error(f"Failed to load image processor: {e}")
        return None

processor = load_processor()

# --- 3. STREAMLIT USER INTERFACE ---

st.set_page_config(page_title="Age Recognition AI", layout="centered")
st.title("🤖 Facial Age Group Prediction")
st.write("Upload an image of a face, and the AI model hosted on OpenShift will predict the age group.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None and processor is not None:
    # Display the uploaded image
   # image = Image.open(uploaded_file).convert("RGB")
    image_to_fix = Image.open(uploaded_file)
    image = ImageOps.exif_transpose(image_to_fix).convert("RGB")
    st.image(image, caption='Uploaded Image', use_container_width=True)

    # Add a button to trigger the prediction
    if st.button('Predict Age Group'):
        with st.spinner('Preprocessing image and sending to model...'):
            try:
                # --- 4. PREPARE IMAGE AND PAYLOAD (from your notebook) ---

                # The processor handles resizing, normalization, etc., to match the model's requirements
                inputs = processor(images=image, return_tensors="np")
                image_data = inputs['pixel_values'].astype(np.float32)

                # Build the payload in the exact format your model expects
                payload = {
                    "inputs": [
                        {
                            "name": "pixel_values",
                            "shape": image_data.shape,
                            "datatype": "FP32",
                            "data": image_data.tolist()
                        }
                    ]
                }

                # --- 5. CALL THE MODEL API (from your notebook) ---

                st.write(f"Sending request to the model server...")
                response = requests.post(INFERENCE_URL, json=payload, verify=False)
                response.raise_for_status() # Raise an error for bad status codes (4xx or 5xx)

                result = response.json()

                # --- 6. PARSE THE RESPONSE & DISPLAY (from your notebook) ---

                output_data = np.array(result['outputs'][0]['data'])
                predicted_class_index = np.argmax(output_data)
#                predicted_age_group = ID2LABEL.get(predicted_class_index, "Unknown")
                predicted_age_group = ID2LABEL.get(predicted_class_index)

                st.success(f"**Predicted Age Group:** {predicted_age_group}")

            except requests.exceptions.RequestException as e:
                st.error(f"Connection Error: Could not connect to the model API. Details: {e}")
            except (KeyError, IndexError) as e:
                st.error(f"Error: Could not parse the model's response. The format may have changed. Details: {e}")
                st.json(result) # Display the raw JSON response for debugging
            except Exception as e:
                st.error(f"An unexpected error occurred: {e}")

elif uploaded_file is not None and processor is None:
    st.error("The image processor could not be loaded. Please check the application logs or network connection.")
