# =============================================================================
# app.py
# =============================================================================
# This is the Streamlit web application.
# It loads the trained CNN model and lets users upload an image of a
# handwritten digit, then displays the predicted digit (0–9).
#
# Run with:
#   streamlit run app.py
# =============================================================================

import numpy as np
import streamlit as st
import tensorflow as tf
from PIL import Image, ImageOps

# -----------------------------------------------------------------------------
# Page Configuration
# -----------------------------------------------------------------------------
# This must be the FIRST Streamlit call in the script

st.set_page_config(
    page_title="Digit Recognizer",
    page_icon="🔢",
    layout="centered"
)

# -----------------------------------------------------------------------------
# Custom CSS Styling
# -----------------------------------------------------------------------------
st.markdown("""
    <style>
        /* Main background and font */
        .main { background-color: #0f0f0f; }

        /* Title styling */
        h1 {
            font-family: 'Courier New', monospace;
            color: #00ff88;
            text-align: center;
            letter-spacing: 4px;
            font-size: 2.4rem;
        }

        /* Subtitle */
        .subtitle {
            text-align: center;
            color: #888;
            font-family: 'Courier New', monospace;
            font-size: 0.9rem;
            margin-top: -10px;
            margin-bottom: 30px;
        }

        /* Prediction result box */
        .result-box {
            background: linear-gradient(135deg, #1a1a2e, #16213e);
            border: 2px solid #00ff88;
            border-radius: 12px;
            padding: 30px;
            text-align: center;
            margin: 20px 0;
        }

        .result-label {
            color: #888;
            font-family: 'Courier New', monospace;
            font-size: 1rem;
            letter-spacing: 3px;
            text-transform: uppercase;
        }

        .result-digit {
            color: #00ff88;
            font-family: 'Courier New', monospace;
            font-size: 6rem;
            font-weight: bold;
            line-height: 1.1;
        }

        .confidence-text {
            color: #aaa;
            font-family: 'Courier New', monospace;
            font-size: 0.85rem;
            margin-top: 8px;
        }

        /* Info box */
        .info-box {
            background: #1a1a1a;
            border-left: 4px solid #00ff88;
            padding: 12px 16px;
            border-radius: 4px;
            font-family: 'Courier New', monospace;
            font-size: 0.85rem;
            color: #aaa;
            margin-bottom: 20px;
        }

        /* Hide Streamlit branding */
        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}
    </style>
""", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# Load the Trained Model
# -----------------------------------------------------------------------------
# @st.cache_resource means the model is loaded ONCE and reused across
# every user interaction — this is important for performance

@st.cache_resource
def load_model():
    """Load the pre-trained CNN model from disk."""
    try:
        model = tf.keras.models.load_model("model/mnist_cnn.h5")
        return model
    except Exception as e:
        return None

model = load_model()

# -----------------------------------------------------------------------------
# Helper Function: Preprocess Uploaded Image
# -----------------------------------------------------------------------------

def preprocess_image(image: Image.Image) -> np.ndarray:
    """
    Prepare a user-uploaded image for CNN prediction.

    Steps:
      1. Convert to grayscale (MNIST is single-channel)
      2. Invert if background is dark (MNIST has white digits on black)
      3. Resize to 28x28 (MNIST image size)
      4. Normalize pixels to [0.0, 1.0]
      5. Reshape to (1, 28, 28, 1) — batch of 1 for the model

    Args:
        image: A PIL Image object (any size, any color)

    Returns:
        A NumPy array of shape (1, 28, 28, 1) ready for prediction
    """
    # Step 1: Convert to grayscale
    image = image.convert("L")

    # Step 2: Resize to 28x28 using high-quality downsampling
    image = image.resize((28, 28), Image.LANCZOS)

    # Step 3: Convert to numpy array
    img_array = np.array(image, dtype="float32")

    # Step 4: Determine if inversion is needed.
    # MNIST digits are WHITE on BLACK background.
    # If the uploaded image has a light background (mean > 127),
    # invert it so the digit is white on black.
    if img_array.mean() > 127:
        img_array = 255.0 - img_array

    # Step 5: Normalize to [0.0, 1.0]
    img_array = img_array / 255.0

    # Step 6: Reshape to (1, 28, 28, 1)
    img_array = img_array.reshape(1, 28, 28, 1)

    return img_array

# -----------------------------------------------------------------------------
# App UI
# -----------------------------------------------------------------------------

# Title
st.markdown("<h1> DIGIT RECOGNIZER</h1>", unsafe_allow_html=True)
st.markdown(
    '<p class="subtitle">CNN · MNIST · TensorFlow/Keras</p>',
    unsafe_allow_html=True
)

# Model status check
if model is None:
    st.error(
        " Model not found. "
    )
    st.stop()  # Stop the app here if no model is available

# Info tip
st.markdown("""
<div class="info-box">
     <strong>Tips for best results:</strong> Use a white or light background with
    a dark pen. Center the digit in the image. Accepted formats: .png, .jpg, .jpeg
</div>
""", unsafe_allow_html=True)

# File uploader widget
uploaded_file = st.file_uploader(
    label="Upload a handwritten digit image",
    type=["png", "jpg", "jpeg"],
    help="Upload a clear image of a single handwritten digit (0–9)"
)

# -----------------------------------------------------------------------------
# Prediction Flow
# -----------------------------------------------------------------------------

if uploaded_file is not None:

    # Open the uploaded image with PIL
    image = Image.open(uploaded_file)

    # Layout: two columns — left for image, right for result
    col1, col2 = st.columns(2, gap="large")

    with col1:
        st.markdown("**Uploaded Image**")
        # Display the original uploaded image
        st.image(image, use_column_width=True, caption="Your image")

    with col2:
        st.markdown("**28×28 Preview (model input)**")
        # Show the preprocessed grayscale version the model actually sees
        preview = image.convert("L").resize((28, 28), Image.LANCZOS)
        st.image(preview, use_column_width=True, caption="Preprocessed")

    # Preprocess and predict
    with st.spinner("Analyzing digit..."):
        processed = preprocess_image(image)

        # model.predict returns an array of shape (1, 10)
        # Each value is the probability for digits 0–9
        predictions = model.predict(processed, verbose=0)

        predicted_digit = int(np.argmax(predictions[0]))   # digit with highest prob
        confidence      = float(np.max(predictions[0]))    # that probability

    # Display result
    st.markdown(f"""
        <div class="result-box">
            <div class="result-label">Predicted Digit</div>
            <div class="result-digit">{predicted_digit}</div>
            <div class="confidence-text">Confidence: {confidence * 100:.1f}%</div>
        </div>
    """, unsafe_allow_html=True)

    # Show full probability breakdown
    st.markdown("**Confidence scores for all digits:**")
    for digit, prob in enumerate(predictions[0]):
        bar_color = "#00ff88" if digit == predicted_digit else "#444"
        st.markdown(
            f"<div style='display:flex; align-items:center; gap:10px; "
            f"margin:3px 0; font-family:monospace;'>"
            f"<span style='color:#aaa; width:15px;'>{digit}</span>"
            f"<div style='flex:1; background:#222; border-radius:4px; height:18px;'>"
            f"<div style='width:{prob*100:.1f}%; background:{bar_color}; "
            f"height:100%; border-radius:4px;'></div></div>"
            f"<span style='color:#aaa; width:50px;'>{prob*100:.1f}%</span>"
            f"</div>",
            unsafe_allow_html=True
        )

else:
    # Placeholder shown before any image is uploaded
    st.markdown("""
        <div style='text-align:center; padding:60px 20px; color:#555;
                    font-family:monospace; border:2px dashed #333; border-radius:12px;'>
            <div style='font-size:3rem;'>📁</div>
            <div style='margin-top:12px;'>Upload an image above to get started</div>
        </div>
    """, unsafe_allow_html=True)

# Footer
st.markdown("---")
st.markdown(
    "<p style='text-align:center; color:#444; font-family:monospace; font-size:0.8rem;'>"
    "CNN trained on MNIST · 99%+ test accuracy · Built with TensorFlow & Streamlit"
    "</p>",
    unsafe_allow_html=True
)