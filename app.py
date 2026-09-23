import streamlit as st
from fastai.vision.all import load_learner, PILImage
import PIL

# 1. Load your trained model
# Make sure the filename matches exactly
LEARNER_PATH = "nature_model.pkl"
learn = load_learner(LEARNER_PATH)

# 2. Define the prediction function
def predict(img):
    # Convert the Streamlit image to a fastai PILImage
    pil_img = PILImage.create(img)
    
    # Get the prediction
    pred, pred_idx, probs = learn.predict(pil_img)
    
    # Display the image
    st.image(img, caption="Uploaded Image", use_column_width=True)
    
    # Display the top prediction and probability
    st.write(f"**Prediction:** {pred}")
    st.write(f"**Probability:** {probs[pred_idx]:.4f}")
    
    # Optional: Show probabilities for all classes
    st.subheader("Full Breakdown")
    for cls, prob in zip(learn.dls.vocab, probs):
        st.write(f"- {cls}: {prob:.4f}")

# 3. Create the Streamlit interface
st.title("Nature Classifier 🐦🐼🐨")
st.write("Upload an image to classify it as a bird, reptile, mammal, forest, or aquatic scene.")

# File uploader widget
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

# If a file is uploaded, process it
if uploaded_file is not None:
    # Convert uploaded file to a PIL Image that fastai can handle
    image = PIL.Image.open(uploaded_file)
    predict(image)