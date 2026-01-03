import streamlit as st
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import VGG16
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
# MODEL BUILDER FUNCTIONS
# =============================

def build_autism_model(img_rows=150, img_cols=150, channels=3):
    """Rebuild autism model architecture"""
    base = VGG16(
        include_top=False,
        weights='imagenet',
        input_shape=(img_rows, img_cols, channels)
    )
    
    for layer in base.layers:
        layer.trainable = False
    
    model = models.Sequential([
        base,
        layers.GlobalAveragePooling2D(),
        layers.Dense(256, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(1, activation='sigmoid')
    ])
    
    return model

def build_emotion_model(img_rows=150, img_cols=150, channels=3, num_classes=6):
    """Rebuild emotion model architecture"""
    base = VGG16(
        include_top=False,
        weights='imagenet',
        input_shape=(img_rows, img_cols, channels)
    )
    
    # Freeze most layers
    for layer in base.layers[:-4]:
        layer.trainable = False
    
    # Fine-tune top layers
    for layer in base.layers[-4:]:
        layer.trainable = True
    
    model = models.Sequential([
        base,
        layers.GlobalAveragePooling2D(),
        layers.Dense(256, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(num_classes, activation='softmax')
    ])
    
    return model

# =============================
# KERAS 3: LOAD WEIGHTS METHOD
# =============================

@st.cache_resource
def load_models_from_weights():
    """
    Load models using weights files (Keras 3 compatible)
    REPLACE FILE IDs WITH YOUR OWN!
    """
    
    # ⚠️ REPLACE THESE WITH YOUR GOOGLE DRIVE FILE IDs
    FILE_IDS = {
        'autism_weights': 'YOUR_AUTISM_WEIGHTS_FILE_ID',
        'emotion_weights': 'YOUR_EMOTION_WEIGHTS_FILE_ID',
        'emotion_classes': 'YOUR_EMOTION_CLASSES_FILE_ID',
        'model_config': 'YOUR_MODEL_CONFIG_FILE_ID'
    }
    
    models_dir = "models"
    os.makedirs(models_dir, exist_ok=True)
    
    files_to_download = {
        'autism_model.weights.h5': FILE_IDS['autism_weights'],
        'emotion_model.weights.h5': FILE_IDS['emotion_weights'],
        'emotion_classes.json': FILE_IDS['emotion_classes'],
        'model_config.json': FILE_IDS['model_config']
    }
    
    # Download files
    for filename, file_id in files_to_download.items():
        filepath = os.path.join(models_dir, filename)
        
        if not os.path.exists(filepath):
            if 'YOUR_' in file_id:
                st.error(f"❌ Please update FILE_IDS with your actual Google Drive file IDs!")
                return None, None, None
                
            with st.spinner(f"⬇️ Downloading {filename}..."):
                try:
                    gdown.download(
                        f"https://drive.google.com/uc?id={file_id}",
                        filepath,
                        quiet=False
                    )
                    st.success(f"✅ {filename} downloaded!")
                except Exception as e:
                    st.error(f"❌ Error downloading {filename}: {e}")
                    return None, None, None
    
    # Load configuration
    config_path = os.path.join(models_dir, 'model_config.json')
    if os.path.exists(config_path):
        with open(config_path, 'r') as f:
            config = json.load(f)
        st.info(f"📦 Model trained with TensorFlow {config.get('tensorflow_version', 'unknown')}")
    else:
        config = {
            'img_rows': 150,
            'img_cols': 150,
            'channels': 3,
            'num_emotion_classes': 6
        }
    
    # Load emotion classes
    classes_path = os.path.join(models_dir, 'emotion_classes.json')
    try:
        with open(classes_path, 'r') as f:
            emotion_classes = json.load(f)
        emotion_classes = {v: k for k, v in emotion_classes.items()}
    except Exception as e:
        st.error(f"❌ Error loading emotion classes: {e}")
        return None, None, None
    
    # Build and load models
    try:
        with st.spinner("🏗️ Building model architectures..."):
            autism_model = build_autism_model(
                config['img_rows'], 
                config['img_cols'], 
                config['channels']
            )
            
            emotion_model = build_emotion_model(
                config['img_rows'],
                config['img_cols'],
                config['channels'],
                config['num_emotion_classes']
            )
        
        with st.spinner("⚙️ Loading model weights..."):
            autism_weights_path = os.path.join(models_dir, 'autism_model.weights.h5')
            emotion_weights_path = os.path.join(models_dir, 'emotion_model.weights.h5')
            
            autism_model.load_weights(autism_weights_path)
            emotion_model.load_weights(emotion_weights_path)
        
        st.success("✅ Models loaded successfully!")
        return autism_model, emotion_model, emotion_classes
        
    except Exception as e:
        st.error(f"❌ Error building/loading models: {e}")
        st.exception(e)
        return None, None, None

# =============================
# KERAS 3: DIRECT .keras LOADING
# =============================

@st.cache_resource
def load_models_keras_format():
    """
    Load models directly from .keras files (Keras 3 native format)
    """
    
    # ⚠️ REPLACE WITH YOUR FILE IDs
    FILE_IDS = {
        'autism_model': '1q3LqM-BOm7YbYhCw1yBL1YB5fNTu9Df4',
        'emotion_model': '1Xk4LymTvup2ipZbTnarq95O_otO_dIb9',
        'emotion_classes': '1Er_XFyn7Jk3ikC4AXXmOUNLG5i4h4Y9g'
    }
    
    models_dir = "models"
    os.makedirs(models_dir, exist_ok=True)
    
    files_to_download = {
        'autism_model.keras': FILE_IDS['autism_model'],
        'emotion_model.keras': FILE_IDS['emotion_model'],
        'emotion_classes.json': FILE_IDS['emotion_classes']
    }
    
    # Download files
    for filename, file_id in files_to_download.items():
        filepath = os.path.join(models_dir, filename)
        
        if not os.path.exists(filepath):
            if 'YOUR_' in file_id:
                st.error(f"❌ Please update FILE_IDS with your actual Google Drive file IDs!")
                return None, None, None
                
            with st.spinner(f"⬇️ Downloading {filename}..."):
                try:
                    gdown.download(
                        f"https://drive.google.com/uc?id={file_id}",
                        filepath,
                        quiet=False
                    )
                    st.success(f"✅ {filename} downloaded!")
                except Exception as e:
                    st.error(f"❌ Error downloading {filename}: {e}")
                    return None, None, None
    
    # Load models directly
    try:
        with st.spinner("🔄 Loading models from .keras files..."):
            autism_model = tf.keras.models.load_model(
                os.path.join(models_dir, 'autism_model.keras')
            )
            emotion_model = tf.keras.models.load_model(
                os.path.join(models_dir, 'emotion_model.keras')
            )
        
        # Load emotion classes
        with open(os.path.join(models_dir, 'emotion_classes.json'), 'r') as f:
            emotion_classes = json.load(f)
        emotion_classes = {v: k for k, v in emotion_classes.items()}
        
        st.success("✅ Models loaded successfully!")
        return autism_model, emotion_model, emotion_classes
        
    except Exception as e:
        st.error(f"❌ Error loading models: {e}")
        st.exception(e)
        st.warning("💡 Try using 'Weights Only' method instead")
        return None, None, None

# =============================
# IMAGE PREPROCESSING
# =============================

def preprocess_image(image, target_size=(150, 150)):
    """Preprocess uploaded image for model prediction"""
    img = image.resize(target_size)
    img_array = np.array(img)
    
    # Handle different image formats
    if img_array.shape[-1] == 4:  # RGBA
        img_array = img_array[:, :, :3]
    elif len(img_array.shape) == 2:  # Grayscale
        img_array = np.stack([img_array] * 3, axis=-1)
    
    # Normalize to [0, 1]
    img_array = img_array / 255.0
    
    # Add batch dimension
    img_array = np.expand_dims(img_array, axis=0)
    
    return img_array

# =============================
# MAIN APP
# =============================

def main():
    st.title("🧠 Autism & Emotion Detection System")
    st.markdown("---")
    
    # Sidebar settings
    st.sidebar.header("⚙️ Settings")
    
    loading_method = st.sidebar.radio(
        "Model Loading Method:",
        ["Weights Only (Recommended)", "Direct .keras Loading"],
        help="Choose 'Weights Only' for maximum compatibility"
    )
    
    st.sidebar.markdown("---")
    
    # Display TensorFlow version
    st.sidebar.info(f"🔧 TensorFlow: {tf.__version__}")
    st.sidebar.info(f"🔧 Keras: {tf.keras.__version__}")
    
    # Info about first-time download
    st.info("ℹ️ **First-time users:** Models will be downloaded automatically (may take a few minutes). Subsequent loads will be instant!")
    
    # Load models based on method
    with st.spinner("🔄 Loading models..."):
        if loading_method == "Weights Only (Recommended)":
            autism_model, emotion_model, emotion_classes = load_models_from_weights()
        else:
            autism_model, emotion_model, emotion_classes = load_models_keras_format()
    
    if autism_model is None or emotion_model is None:
        st.error("⚠️ Failed to load models.")
        
        with st.expander("📝 Setup Instructions", expanded=True):
            st.markdown("""
            ### Step 1: Save Models in Kaggle
            
            Add this code at the end of your training notebook:
            
            ```python
            # Save weights (Keras 3 format)
            autism_model.save_weights('autism_model.weights.h5')
            emotion_model.save_weights('emotion_model.weights.h5')
            
            # Save complete models
            autism_model.save('autism_model.keras')
            emotion_model.save('emotion_model.keras')
            
            # Save classes and config
            import json
            with open('emotion_classes.json', 'w') as f:
                json.dump(emotion_train_gen.class_indices, f)
            
            config = {'img_rows': 150, 'img_cols': 150, 'channels': 3, 'num_emotion_classes': 6}
            with open('model_config.json', 'w') as f:
                json.dump(config, f)
            ```
            
            ### Step 2: Upload to Google Drive
            
            1. Download files from Kaggle Output section
            2. Upload to Google Drive
            3. Make publicly accessible (Anyone with link → Viewer)
            4. Get file IDs from sharing links
            
            ### Step 3: Update This App
            
            Replace FILE_IDS in the code with your actual Google Drive file IDs
            """)
        return
    
    # Sidebar - Detection options
    st.sidebar.markdown("---")
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
            if st.button("🔍 Analyze Image", type="primary", use_container_width=True):
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
        <p style='font-size: 12px;'>Compatible with Keras 3</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()

