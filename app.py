import streamlit as st
from fastai.vision.all import load_learner, PILImage
import PIL
import os
import urllib.request

# --- Configuration ---
MODEL_FILENAME = "nature_model.pkl"
HF_MODEL_URL = "https://huggingface.co/diana1space/nature-classifier-model/resolve/main/nature_model.pkl"

# --- Delete any old (possibly broken) file ---
if os.path.exists(MODEL_FILENAME):
    os.remove(MODEL_FILENAME)

# --- Download the model fresh ---
with st.spinner("Downloading model from Hugging Face... this may take a moment."):
    try:
        urllib.request.urlretrieve(HF_MODEL_URL, MODEL_FILENAME)
    except Exception as e:
        st.error(f"Failed to download the model: {e}")
        st.stop()

# --- Safety check: is it a real pickle or HTML? ---
with open(MODEL_FILENAME, "rb") as f:
    first_bytes = f.read(4)

if first_bytes.startswith(b"<"):
    st.error(
        "Downloaded file is HTML, not a model file. "
        "This usually means the Hugging Face repo is private, "
        "or the URL is wrong."
    )
    st.write(f"First bytes received: {first_bytes}")
    st.write(f"URL was: {HF_MODEL_URL}")
    st.stop()

# --- Load the model ---
try:
    learn = load_learner(MODEL_FILENAME)
except Exception as e:
    st.error(f"Failed to load the model: {e}")
    st.stop()

# --- UI ---
st.title("Nature Classifier 🐦🐼🐨")
st.write("Upload an image and I'll classify it as a bird, reptile, mammal, forest, or aquatic scene.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = PIL.Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", width="content")

    # Run prediction
    pil_img = PILImage.create(image)
    pred_class, pred_idx, probs = learn.predict(pil_img)

    # Show result
    st.subheader("Prediction")
    st.write(f"**{pred_class}**  ({probs[pred_idx]:.4f})")

    # Show full breakdown
    st.subheader("Full Breakdown")
    for cls, prob in zip(learn.dls.vocab, probs):
        st.write(f"- **{cls}**: {prob:.4f}")
