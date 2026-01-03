import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import json
import gdown
import os

# Page config
st.set_page_config(
    page_title="Autism & Emotion Detection",
    page_icon="🧠",
    layout="wide"
)

# =============================
# GOOGLE DRIVE MODEL LOADING
# =============================

@st.cache_resource
def download_models_from_gdrive():
    """
    Download models from Google Drive if not present locally
    Replace these IDs with your own Google Drive file IDs
    """
    
    # YOUR GOOGLE DRIVE FILE IDs (Replace these!)
    AUTISM_MODEL_ID = "1bWg9-F6dgPDirLsQxZkrSfl7ImygC7My"
    EMOTION_MODEL_ID = "1Xk4LymTvup2ipZbTnarq95O_otO_dIb9"
    EMOTION_CLASSES_ID = "1Er_XFyn7Jk3ikC4AXXmOUNLG5i4h4Y9g"
    
    models_dir = "models"
    os.makedirs(models_dir, exist_ok=True)
    
    autism_path = os.path.join(models_dir, "autism_model.keras")
    emotion_path = os.path.join(models_dir, "emotion_model.keras")
    classes_path = os.path.join(models_dir, "emotion_classes.json")
    
    # Download autism model
    if not os.path.exists(autism_path):
        with st.spinner("⬇️ Downloading Autism Detection Model (first time only)..."):
            try:
                gdown.download(
                    f"https://drive.google.com/uc?id={AUTISM_MODEL_ID}",
                    autism_path,
                    quiet=False
                )
                st.success("✅ Autism model downloaded!")
            except Exception as e:
                st.error(f"❌ Error downloading autism model: {e}")
                return None, None, None
    
    # Download emotion model
    if not os.path.exists(emotion_path):
        with st.spinner("⬇️ Downloading Emotion Recognition Model (first time only)..."):
            try:
                gdown.download(
                    f"https://drive.google.com/uc?id={EMOTION_MODEL_ID}",
                    emotion_path,
                    quiet=False
                )
                st.success("✅ Emotion model downloaded!")
            except Exception as e:
                st.error(f"❌ Error downloading emotion model: {e}")
                return None, None, None
    
    # Download emotion classes JSON
    if not os.path.exists(classes_path):
        with st.spinner("⬇️ Downloading emotion classes..."):
            try:
                gdown.download(
                    f"https://drive.google.com/uc?id={EMOTION_CLASSES_ID}",
                    classes_path,
                    quiet=False
                )
                st.success("✅ Classes downloaded!")
            except Exception as e:
                st.error(f"❌ Error downloading classes: {e}")
                return None, None, None
    
    # Load models
    try:
        autism_model = tf.keras.models.load_model(autism_path)
        emotion_model = tf.keras.models.load_model(emotion_path)
        
        with open(classes_path, 'r') as f:
            emotion_classes = json.load(f)
        emotion_classes = {v: k for k, v in emotion_classes.items()}
        
        return autism_model, emotion_model, emotion_classes
    except Exception as e:
        st.error(f"❌ Error loading models: {e}")
        return None, None, None

# Preprocess image
def preprocess_image(image, target_size=(150, 150)):
    """Preprocess uploaded image for model prediction"""
    img = image.resize(target_size)
    img_array = np.array(img)
    
    if img_array.shape[-1] == 4:  # RGBA
        img_array = img_array[:, :, :3]
    elif len(img_array.shape) == 2:  # Grayscale
        img_array = np.stack([img_array] * 3, axis=-1)
    
    img_array = img_array / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

# Main app
def main():
    st.title("🧠 Autism & Emotion Detection System")
    st.markdown("---")
    
    # Info about first-time download
    st.info("ℹ️ **First-time users:** Models will be downloaded automatically (may take a few minutes). Subsequent loads will be instant!")
    
    # Load models
    with st.spinner("🔄 Loading models..."):
        autism_model, emotion_model, emotion_classes = download_models_from_gdrive()
    
    if autism_model is None or emotion_model is None:
        st.error("⚠️ Failed to load models. Please check your Google Drive file IDs in the code.")
        st.markdown("""
        ### 📝 Setup Instructions:
        1. Upload your models to Google Drive
        2. Make them publicly accessible (Anyone with link can view)
        3. Get the file IDs from the sharing links
        4. Replace the IDs in the code
        """)
        return
    
    st.success("✅ Models loaded successfully!")
    
    # Sidebar
    st.sidebar.header("🎯 Detection Options")
    detection_type = st.sidebar.selectbox(
        "Choose Detection Type:",
        ["Autism Detection", "Emotion Recognition", "Both"]
    )
    
    st.sidebar.markdown("---")
    st.sidebar.info("""
    **How to use:**
    1. Upload a face image
    2. Select detection type
    3. Click 'Analyze'
    4. View results
    """)
    
    # Main content
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.header("📤 Upload Image")
        uploaded_file = st.file_uploader(
            "Choose a face image...",
            type=['jpg', 'jpeg', 'png'],
            help="Upload a clear face image for best results"
        )
        
        if uploaded_file is not None:
            image = Image.open(uploaded_file)
            st.image(image, caption="Uploaded Image", use_container_width=True)
    
    with col2:
        st.header("📊 Analysis Results")
        
        if uploaded_file is not None:
            if st.button("🔍 Analyze Image", type="primary"):
                with st.spinner("Analyzing..."):
                    processed_img = preprocess_image(image)
                    
                    # Autism Detection
                    if detection_type in ["Autism Detection", "Both"]:
                        st.subheader("🧩 Autism Detection")
                        autism_pred = autism_model.predict(processed_img, verbose=0)[0][0]
                        autism_label = "Autistic" if autism_pred > 0.5 else "Non-Autistic"
                        autism_confidence = autism_pred if autism_pred > 0.5 else 1 - autism_pred
                        
                        if autism_label == "Autistic":
                            st.error(f"**Prediction:** {autism_label}")
                        else:
                            st.success(f"**Prediction:** {autism_label}")
                        
                        st.metric("Confidence", f"{autism_confidence * 100:.2f}%")
                        st.progress(float(autism_confidence))
                        st.markdown("---")
                    
                    # Emotion Recognition
                    if detection_type in ["Emotion Recognition", "Both"]:
                        st.subheader("😊 Emotion Recognition")
                        emotion_pred = emotion_model.predict(processed_img, verbose=0)[0]
                        emotion_idx = np.argmax(emotion_pred)
                        emotion_label = emotion_classes.get(emotion_idx, "Unknown")
                        emotion_confidence = emotion_pred[emotion_idx]
                        
                        st.info(f"**Detected Emotion:** {emotion_label}")
                        st.metric("Confidence", f"{emotion_confidence * 100:.2f}%")
                        
                        st.write("**All Emotion Probabilities:**")
                        emotion_df = {
                            emotion_classes.get(i, f"Class_{i}"): f"{prob * 100:.2f}%"
                            for i, prob in enumerate(emotion_pred)
                        }
                        for emotion, prob in sorted(emotion_df.items(), 
                                                   key=lambda x: float(x[1].strip('%')), 
                                                   reverse=True):
                            st.write(f"- **{emotion}:** {prob}")
        else:
            st.info("👆 Please upload an image to begin analysis")
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center'>
        <p>🔬 Powered by TensorFlow & VGG16 | Built with Streamlit</p>
        <p style='font-size: 12px;'>Models loaded from Google Drive</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()