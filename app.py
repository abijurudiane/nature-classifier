import streamlit as st
from fastai.vision.all import load_learner, PILImage
import PIL
import os
import urllib.request

MODEL_FILENAME = "nature_model.pkl"
HF_MODEL_URL = "https://huggingface.co/diana1space/nature-classifier-model/resolve/main/nature_model.pkl"

if not os.path.exists(MODEL_FILENAME):
    with st.spinner("Downloading model..."):
        try:
            urllib.request.urlretrieve(HF_MODEL_URL, MODEL_FILENAME)
        except Exception as e:
            st.error(f"Failed to download the model: {e}")
            st.stop()

with open(MODEL_FILENAME, "rb") as f:
    first_bytes = f.read(4)

if first_bytes.startswith(b"<"):
    st.error("Downloaded file is HTML, not a model.")
    st.stop()

@st.cache_resource
def load_my_model():
    return load_learner(MODEL_FILENAME)

try:
    learn = load_my_model()
except Exception as e:
    st.error(f"Failed to load the model: {e}")
    st.stop()

st.title("Nature Classifier 🐦🐼🐨")
st.write("Upload an image and I'll classify it as a bird, reptile, mammal, forest, or aquatic scene.")

uploaded_file = st.file_uploader(
    "Choose an image...",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:
    image_bytes = uploaded_file.getvalue()

    # Show the uploaded image in the UI
    preview_image = PIL.Image.open(uploaded_file).convert("RGB")
    preview_image.thumbnail((400, 400))
    st.image(preview_image, caption="Uploaded Image", width="stretch")

    # Predict using raw bytes — fastai accepts bytes directly
    pred_class, pred_idx, probs = learn.predict(image_bytes)

    st.subheader("Prediction")
    st.write(f"**{pred_class}**  ({probs[pred_idx]:.4f})")

    st.subheader("Full Breakdown")
    for cls, prob in zip(learn.dls.vocab, probs):
        st.write(f"- **{cls}**: {prob:.4f}")
