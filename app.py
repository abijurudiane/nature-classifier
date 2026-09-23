import streamlit as st
from fastai.vision.all import load_learner, PILImage
import PIL
import os
import urllib.request

# --- Configuration ---
MODEL_FILENAME = "nature_model.pkl"
# Replace this URL with your actual Hugging Face model URL
HF_MODEL_URL = "https://huggingface.co/diana1space/nature-classifier-model/commit/fa57e4bad0570ce94ae185e482c6fb1496ddb888"

# --- Download the model if it doesn't exist ---
if not os.path.exists(MODEL_FILENAME):
    with st.spinner("Downloading model from Hugging Face... this may take a moment."):
        urllib.request.urlretrieve(HF_MODEL_URL, MODEL_FILENAME)

# --- Load the model ---
# You can add st.cache_resource here if you want to cache it in memory, but for a simple app, this is fine.
learn = load_learner(MODEL_FILENAME)

# --- The rest of your Streamlit code follows ---
# (Your `predict` function and UI code go here) 