import streamlit as st
import seaborn as sns
import tensorflow as tf
import numpy as np
import os
import joblib
import hashlib
import matplotlib.pyplot as plt
import cv2
import json
from PIL import Image
from openai import OpenAI
from tensorflow.keras.preprocessing import image
from skimage.feature import hog
from medscan_db_setup import get_db, Physician, PatientRecord, PatientPrediction
from sqlalchemy.orm import sessionmaker
from sqlalchemy import create_engine, update
import requests
from io import BytesIO
from skimage.measure import find_contours
from tensorflow.keras.models import Model
import io
import pydicom

# %matplotlib inline

# ✅ PostgreSQL Database Connection
engine = create_engine("postgresql://postgres:postgres@localhost:5432/medscan_app_db")
Session = sessionmaker(bind=engine)
session = Session()

# ✅ Hash Password Function
def hash_password(password):
    return hashlib.sha256(password.encode()).hexdigest()

# ✅ OpenAI API Key
client = OpenAI(api_key="*******")  # Replace with your actual API key

# ✅ Increase overall app width
st.set_page_config(layout="wide")

# ✅ Custom CSS to make all elements wider
st.markdown(
    """
    <style>
    /* Increase main content width */
    .stApp {
        max-width: 95% !important;
        padding-left: 2%;
        padding-right: 2%;
    }

    /* Increase sidebar width */
    .css-1d391kg {
        width: 300px !important;
    }

    /* Increase text input box width */
    .stTextInput, .stNumberInput, .stSelectbox, .stTextArea {
        width: 100% !important;
    }

    /* Make uploaded images full-width */
    .element-container img {
        max-width: 100% !important;
        height: auto !important;
    }

    /* Increase font size for headings */
    .stMarkdown h1, .stMarkdown h2, .stMarkdown h3 {
        font-size: 24px !important;
    }

    /* Increase tab font size */
    div[data-testid="stTabs"] button {
        font-size: 18px !important;
        font-weight: bold !important;
    }
    </style>
    """,
    unsafe_allow_html=True
)



# ✅ Dynamic Background Image for Signup, Login, and Main Application
def add_background():
    # Ensure session state exists before using it
    if "page" not in st.session_state:
        st.session_state["page"] = "Signup"  # Default to Login

    # Background image logic
    if st.session_state["page"] in ["Signup", "Login"]:
        background_image_url = "https://images.pexels.com/photos/5407212/pexels-photo-5407212.jpeg"  # Image for Login & Signup
    else:
        # background_image_url = "https://mnoncology.com/application/files/cache/thumbnails/gettyimages-1214718462-58629ce6e9ef8b7692ecec881ce1ce8e.jpg"  # New Image for Main App
        background_image_url = "https://images.pexels.com/photos/5912576/pexels-photo-5912576.jpeg"

    # Apply background style
    st.markdown(
        f"""
        <style>
        .stApp {{
            background: url({background_image_url});
            background-size: cover;
            background-position: center;
            background-repeat: no-repeat;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )

# ✅ Apply Background
add_background()


# add_background(background_image_url)

# ✅ Fetch Patient Details by ID
def get_patient_details(patient_id):
    return session.query(PatientRecord).filter_by(patient_id=patient_id).first()

# ✅ Fetch Patient Predictions by ID
def get_patient_predictions(patient_id):
    return session.query(PatientPrediction).filter_by(patient_id=patient_id).order_by(PatientPrediction.created_at.desc()).first()

# ✅ Load AI Model for Disease Detection
@st.cache_resource
def load_disease_model():
    # return tf.keras.models.load_model("NIH_EfficientNetV2S_model.keras")
    return tf.keras.models.load_model("NIH_EfficientNetV2S_model_14k.h5", safe_mode=False)

model = load_disease_model()

# ✅ Load X-ray Validation Model
@st.cache_resource
def load_xray_validation_model():
    return joblib.load("chest_xray_rf_model_hybrid.pkl")

xray_validator = load_xray_validation_model()

CLASS_NAMES = ['No Finding', 'Cardiomegaly', 'Hernia', 'Infiltration', 'Nodule', 'Emphysema','Effusion', 'Atelectasis', 'Pleural_Thickening', 'Pneumothorax','Mass', 'Fibrosis', 'Consolidation', 'Edema', 'Pneumonia']

auc_scores = {
    "No Finding": 0.8401473942796982,
    "Cardiomegaly": 0.9564229249011857,
    "Hernia": 0.8723808445459292,
    "Infiltration": 0.7733112101659854,
    "Nodule": 0.8536433492315845,
    "Emphysema": 0.9427499721634562,
    "Effusion" : 0.8891934099347406,
    "Atelectasis": 0.8682137117010666,
    "Pleural_Thickening": 0.8720200507142087,
    "Pneumothorax": 0.9232435359415188,
    "Mass": 0.898240935100785,
    "Fibrosis": 0.8576267207699659,
    "Consolidation": 0.8366648137347816,
    "Edema": 0.9308000200130084,
    "Pneumonia": 0.8394213595974418
}


IMG_SIZE = 600

def convert_dicom_to_image(dicom_path):
    """Convert DICOM (.dcm) file to a standard image format (JPG/PNG)."""
    try:
        dicom_data = pydicom.dcmread(dicom_path)  # Read DICOM file
        image_array = dicom_data.pixel_array  # Extract pixel data

        # Normalize pixel values (DICOMs often have high bit depths)
        image_array = (image_array - np.min(image_array)) / (np.max(image_array) - np.min(image_array))
        image_array = (image_array * 255).astype(np.uint8)  # Convert to 8-bit image

        # Save as PNG
        img_path = dicom_path.replace(".dcm", ".jpg")
        cv2.imwrite(img_path, image_array)
        print(f"✅ DICOM converted to image: {img_path}")

        return img_path
    except Exception as e:
        print(f"⚠️ Error converting DICOM to image: {str(e)}")
        return None
    
def preprocess_xray(image_path):
    """Preprocess the X-ray image for model input."""
    img = Image.open(image_path).convert("RGB")  # Ensure RGB format
    img = np.array(img)  # Convert to NumPy array

    # Ensure correct color channels
    if len(img.shape) == 2:  # If still grayscale, force it to RGB
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    elif img.shape[-1] == 4:  # If RGBA, convert to RGB
        img = cv2.cvtColor(img, cv2.COLOR_RGBA2RGB)

    img_size = 600  # Resize to model input size
    img_resized = cv2.resize(img, (img_size, img_size)) / 255.0
    img_array = np.expand_dims(img_resized, axis=0)  # Add batch dimension

    return img_array

def convert_google_drive_link(image_path):
    """Convert a Google Drive shareable link to a direct download link."""
    if isinstance(image_path, str) and "drive.google.com" in image_path:
        file_id = image_path.split("id=")[-1]
        return f"https://drive.google.com/uc?export=download&id={file_id}"
    return image_path  # Return unchanged if it's already a local path or Image object

def fetch_image(image_path):
    """Fetch image from Google Drive, HTTP URL, or local path."""
    if isinstance(image_path, str):  # Only convert if it's a string (URL or local path)
        image_path = convert_google_drive_link(image_path)  

        if image_path.startswith("http"):  # If it's a URL, download the image
            try:
                response = requests.get(image_path, stream=True)
                response.raise_for_status()  # Raise error for bad status codes (e.g., 403, 404)
                return Image.open(BytesIO(response.content))
            except requests.exceptions.RequestException as e:
                st.error(f"❌ Failed to fetch image. Error: {e}")
                return None
        else:
            try:
                return Image.open(image_path)  # Local file path
            except FileNotFoundError:
                st.error("❌ Image file not found. Check the file path.")
                return None
    elif isinstance(image_path, Image.Image):  
        return image_path  # Already an Image object, return as is
    else:
        st.error("❌ Invalid image path. Expected a URL or file path.")
        return None


def predict_disease(image_input):
    """Run AI model and return predictions. Accepts both file paths and PIL Images."""
    
    # ✅ Handle case where input is a PIL Image
    if isinstance(image_input, Image.Image):
        img = image_input.convert("RGB")  # Ensure RGB format
    else:
        # ✅ Load image from file path
        img = Image.open(image_input).convert("RGB")

    img = img.resize((600, 600))  # Resize to model input size

    # ✅ Convert to NumPy array
    img_array = image.img_to_array(img) / 255.0  # Normalize to [0,1]
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension

    # ✅ Ensure TensorFlow model input is correct
    img_array = tf.convert_to_tensor(img_array, dtype=tf.float32)

    # ✅ Run AI model
    with st.spinner("🩺 Our cutting-edge Meta-EfficientNetV2S AI model is examining your X-ray for potential diseases. This may take a few seconds…"):
        probabilities = model.predict(img_array)[0]

    # ✅ Filter diseases based on probability thresholds
    detected_diseases = {
        CLASS_NAMES[i]: round(prob, 4) for i, prob in enumerate(probabilities) if prob > 0.5
    }
    pred_diseases = {
        CLASS_NAMES[i]: round(prob, 4) for i, prob in enumerate(probabilities) if prob > 0.2
    }
    
    all_predictions = {
        CLASS_NAMES[i]: round(prob, 4) for i, prob in enumerate(probabilities)
    }
    

    return probabilities, detected_diseases, pred_diseases, all_predictions



def extract_hog_canny_features(img):
    """Extract HOG + Canny features for X-ray validation."""
    img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2GRAY)
    img = cv2.resize(img, (224, 224))

    # Extract HOG features
    hog_features, _ = hog(img, orientations=12, pixels_per_cell=(4, 4),
                           cells_per_block=(2, 2), visualize=True, feature_vector=True)

    # Extract Canny edges
    canny_edges = cv2.Canny(img, threshold1=30, threshold2=100)
    canny_features = canny_edges.flatten()

    # Combine HOG + Canny features
    combined_features = np.concatenate((hog_features, canny_features))

    return combined_features

def validate_xray(img):
    """Run the X-ray validation model and return a confidence score."""
    features = extract_hog_canny_features(img)
    features = np.array(features).reshape(1, -1)

    prediction = xray_validator.predict(features)
    confidence = xray_validator.predict_proba(features)[0][1]  # Confidence score for "Chest X-ray"

    return "Chest X-ray" if prediction[0] == 1 else "Non-Chest X-ray", confidence


def generate_prescription_report(predictions_dict, auc_scores, symptoms):
    """Generate a structured medical prescription report using OpenAI's GPT API, considering AUC scores for confidence levels and patients symptoms."""

    # Convert float32 values to standard Python float
    predictions_dict = {key: float(value) for key, value in predictions_dict.items()}

    # Categorize predictions based on probability & AUC confidence
    categorized_diseases = {"High Confidence": [], "Medium Confidence": [], "Low Confidence": []}
    
    for disease, prob in predictions_dict.items():
        auc_score = auc_scores.get(disease, 0.75)  # Default AUC to 0.75 if missing

        if prob >= 0.50 and auc_score > 0.85:
            categorized_diseases["High Confidence"].append((disease, round(prob * 100, 2), round(auc_score, 3)))
        elif 0.30 <= prob < 0.50 and 0.75 <= auc_score <= 0.85:
            categorized_diseases["Medium Confidence"].append((disease, round(prob * 100, 2), round(auc_score, 3)))
        else:
            categorized_diseases["Low Confidence"].append((disease, round(prob * 100, 2), round(auc_score, 3)))

    # Generate structured output
    def format_disease_list(disease_list):
        return "\n".join([f"- {disease}: {prob}% (AUC: {auc})" for disease, prob, auc in disease_list]) if disease_list else "None"

    high_conf = format_disease_list(categorized_diseases["High Confidence"])
    med_conf = format_disease_list(categorized_diseases["Medium Confidence"])
    low_conf = format_disease_list(categorized_diseases["Low Confidence"])
    patient_symptoms = symptoms

    prompt = f"""
    **Patient X-ray Analysis Report**  
    The AI model has analyzed the patient's chest X-ray and categorized the following diseases based on probability and model confidence (AUC scores):

    🔹 **High Confidence (Strong Model Performance & High Probability)**  
    {high_conf}

    🟡 **Medium Confidence (Moderate Probability or Model Uncertainty)**  
    {med_conf}

    ⚠️ **Low Confidence (Low Probability or Weak Model Performance)**  
    {low_conf}

    🩺 **Analysis & Considerations:**  
    - High confidence diseases (above 50% with strong AUC) indicate a strong likelihood of presence.  
    - Medium confidence diseases (30-50%) require additional tests for validation.  
    - Low confidence diseases may be model misclassifications and should be interpreted cautiously.
    - Consider into account of the patient symptoms as well {patient_symptoms} and analyze highlighting this with the scores.

    🔬 **Next Steps & Recommendations:**  
    - 🏥 **Recommended Medical Tests:** (Based on high-confidence diseases)  
    - 💊 **Potential Treatment Options:** (If applicable)  
    - 📅 **Follow-up Consultations:** (Suggest relevant specialists)  
    - ⚡ **Lifestyle Modifications:** (If applicable)  

    Please provide a formal **structured prescription report of around 700 words** including:
    1️⃣ **Diagnosis Summary**  
    2️⃣ **Recommended Tests**  
    3️⃣ **Suggested Medications**  
    4️⃣ **Consultation Advice (Specialist Referrals)**  
    5️⃣ **Lifestyle & Health Recommendations**  

    Keep the response **clear, concise, and medically relevant**.
    """

    response = client.chat.completions.create(
        # model="gpt-3.5-turbo",
        # model="gpt-4o",
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}]
    )

    return response.choices[0].message.content

def plot_disease_probabilities(probabilities):
    """Generate a visually appealing bar chart for disease probabilities (excluding 'No Finding')."""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Keep original indices from CLASS_NAMES while filtering out "No Finding"
    disease_probs = {CLASS_NAMES[i]: probabilities[i] * 100 for i in range(len(CLASS_NAMES)) if CLASS_NAMES[i] != "No Finding"}
    
    # Sort diseases by probability in descending order
    sorted_disease_probs = dict(sorted(disease_probs.items(), key=lambda item: item[1], reverse=True))
    
    # Generate colors for the 14 disease classes
    colors = sns.color_palette("husl", len(sorted_disease_probs))

    # Create the bar plot
    sns.barplot(
        y=list(sorted_disease_probs.keys()),
        x=list(sorted_disease_probs.values()),
        palette=colors,
        ax=ax
    )

    ax.set_xlabel("Probability (%)", fontsize=14, fontweight='bold')
    ax.set_ylabel("Diseases", fontsize=14, fontweight='bold')
    ax.set_title("Predicted Disease Probabilities", fontsize=16, fontweight='bold')

    # Annotate bars with values
    for i, (disease, prob) in enumerate(sorted_disease_probs.items()):
        ax.text(prob + 1, i, f"{prob:.2f}%", va='center', fontsize=12, fontweight='bold')

    plt.xlim(0, 100)
    st.pyplot(fig)  # Display the plot in Streamlit



def gradcam_with_contour(image_path, class_name, model):
    all_classes = ['No Finding', 'Cardiomegaly', 'Hernia', 'Infiltration', 'Nodule', 'Emphysema',
                   'Effusion', 'Atelectasis', 'Pleural_Thickening', 'Pneumothorax',
                   'Mass', 'Fibrosis', 'Consolidation', 'Edema', 'Pneumonia']
    
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_size = 600
    img_resized = cv2.resize(img, (img_size, img_size)) / 255.0
    img_array = np.expand_dims(img_resized, axis=0)
    
    preds = model.predict(img_array)[0]
    class_idx = all_classes.index(class_name)
    class_prob = preds[class_idx]
    
    # Auto-detect last convolutional layer
    for layer in reversed(model.layers):
        if isinstance(layer, tf.keras.layers.Conv2D):
            last_conv_layer = layer
            break
    
    grad_model = Model(inputs=model.input, outputs=[last_conv_layer.output, model.output])
    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        loss = predictions[:, class_idx]
    grads = tape.gradient(loss, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    conv_outputs = conv_outputs[0] * pooled_grads
    heatmap = tf.reduce_mean(conv_outputs, axis=-1).numpy()
    heatmap = np.maximum(heatmap, 0)
    heatmap = heatmap / np.max(heatmap)
    
    # Generate contours
    heatmap_resized = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
    contours = find_contours(heatmap_resized, 0.5)
    
    # Plot image with contour overlay
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.imshow(img)
    for contour in contours:
        ax.plot(contour[:, 1], contour[:, 0], linestyle='dotted', color='red', linewidth=2)
    
    ax.add_patch(plt.Rectangle((0, 0), img.shape[1], 40, color='grey', alpha=0.8))
    ax.text(10, 25, f"{class_name}: {class_prob:.2f}", fontsize=14, color='white', weight='bold')
    ax.axis('off')

    if class_name not in ['No Finding']:
        # Convert figure to image for Streamlit
        img_buf = io.BytesIO()
        plt.savefig(img_buf, format="png", bbox_inches='tight', pad_inches=0)
        img_buf.seek(0)
        st.image(img_buf, caption=f"Grad-CAM Contour for {class_name}", width=500)
        plt.close(fig)


def gradcam_with_heatmap(image_path, class_name, model):
    all_classes = ['No Finding', 'Cardiomegaly', 'Hernia', 'Infiltration', 'Nodule', 'Emphysema',
                   'Effusion', 'Atelectasis', 'Pleural_Thickening', 'Pneumothorax',
                   'Mass', 'Fibrosis', 'Consolidation', 'Edema', 'Pneumonia']
    
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_size = 600
    img_resized = cv2.resize(img, (img_size, img_size)) / 255.0
    img_array = np.expand_dims(img_resized, axis=0)
    
    preds = model.predict(img_array)[0]
    class_idx = all_classes.index(class_name)
    class_prob = preds[class_idx]
    
    # Auto-detect last convolutional layer
    for layer in reversed(model.layers):
        if isinstance(layer, tf.keras.layers.Conv2D):
            last_conv_layer = layer
            break
    
    grad_model = Model(inputs=model.input, outputs=[last_conv_layer.output, model.output])
    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        loss = predictions[:, class_idx]
    grads = tape.gradient(loss, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    conv_outputs = conv_outputs[0] * pooled_grads
    heatmap = tf.reduce_mean(conv_outputs, axis=-1).numpy()
    heatmap = np.maximum(heatmap, 0)
    heatmap = heatmap / np.max(heatmap)
    
    # Resize heatmap to match the original image
    heatmap_resized = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
    heatmap_colored = cv2.applyColorMap(np.uint8(255 * heatmap_resized), cv2.COLORMAP_JET)
    superimposed_img = cv2.addWeighted(img, 0.6, heatmap_colored, 0.4, 0)
    
    # Plot image with heatmap overlay
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.imshow(superimposed_img)
    
    ax.add_patch(plt.Rectangle((0, 0), img.shape[1], 60, color='navy', alpha=0.8))
    ax.text(10, 25, f"{class_name}: {class_prob:.2f}", fontsize=14, color='white', weight='bold')
    ax.axis('off')
    
    if class_name not in ['No Finding']:
        # Convert figure to image for Streamlit
        img_buf = io.BytesIO()
        plt.savefig(img_buf, format="png", bbox_inches='tight', pad_inches=0)
        img_buf.seek(0)
        st.image(img_buf, caption=f"Grad-CAM Heatmap for {class_name}", width=500)
        plt.close(fig)


def plot_disease_confidence_bubble(pred_diseases):
    """Enhanced visualization of AI probability vs. model confidence for all 14 diseases with better readability."""
    
    # ✅ Ensure all 14 diseases are included, even if not detected
    all_diseases = CLASS_NAMES  # List of all 14 diseases
    probabilities = np.array([pred_diseases.get(d, 0.01) for d in all_diseases])  # Ensure no zero probabilities
    confidences = np.array([auc_scores.get(d, 0.75) for d in all_diseases])  # Assign default AUC=0.75 if missing

    # ✅ Adjust probabilities so they are never exactly zero (prevents getting stuck on x-axis)
    probabilities = np.clip(probabilities, 0.01, 1.0)  # Set a minimum of 1% probability

    # ✅ Define bubble size dynamically based on probability & confidence
    sizes = (probabilities * confidences + 0.2) * 2000  # Add 0.2 to ensure visibility

    # ✅ Color Mapping
    colors = ["red" if prob >= 0.5 else "orange" if 0.3 <= prob < 0.5 else "gray" for prob in probabilities]  
    # High Confidence = Red, Medium = Orange, Low = Gray

    # ✅ Create figure & axis
    fig, ax = plt.subplots(figsize=(12, 7))

    # ✅ Scatter plot with better color contrast
    scatter = ax.scatter(probabilities, confidences, s=sizes, c=colors, alpha=0.8, edgecolors="black")
    
    # ✅ Add color bar for confidence interpretation
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label("Model Confidence (AUC)", fontsize=12)

    # ✅ Confidence Level Bands (Horizontal)
    ax.axhline(y=0.75, color="gray", linestyle="dashed", alpha=0.6, label="Medium Confidence (AUC ≥ 0.75)")
    ax.axhline(y=0.85, color="black", linestyle="dotted", alpha=0.6, label="High Confidence (AUC ≥ 0.85)")
    ax.axhline(y=0.65, color="lightgray", linestyle="dotted", alpha=0.6, label="Low Confidence (AUC ≤ 0.75)")

    # ✅ Probability Level Bands (Vertical)
    ax.axvline(x=0.3, color="gray", linestyle="dashed", alpha=0.6, label="Medium Probability (≥ 30%)")
    ax.axvline(x=0.5, color="black", linestyle="dotted", alpha=0.6, label="High Probability (≥ 50%)")

    # ✅ Spread out text labels to avoid clutter
    for i, disease in enumerate(all_diseases):
        y_offset = -0.03 if i % 2 == 0 else 0.03  # Spread text up and down
        x_offset = -0.03 if i % 3 == 0 else 0.03  # Spread text left and right
        ax.text(probabilities[i] + x_offset, confidences[i] + y_offset, f"{disease} ({probabilities[i]*100:.1f}%)",
                fontsize=11, ha="center", va="bottom", color="black", bbox=dict(facecolor="white", alpha=0.7, edgecolor="gray", boxstyle="round,pad=0.3"))

    # ✅ Set labels & title
    ax.set_xlabel("AI Disease Probability Score", fontsize=14, fontweight="bold")
    ax.set_ylabel("Model Confidence (AUC Score)", fontsize=14, fontweight="bold")
    ax.set_title("🔬 AI Confidence vs. Disease Detection Probability (All 14 Diseases)", fontsize=16, fontweight="bold")

    # ✅ Adjust axis limits & grid for clarity
    ax.set_xlim(0, 1.0)
    ax.set_ylim(0.6, 1.0)  # Extend range slightly to prevent label cutoff
    ax.grid(True, linestyle="--", alpha=0.5)

    # ✅ Show legend for confidence & probability bands
    ax.legend(loc="lower right", fontsize=10)

    # ✅ Show plot in Streamlit
    st.pyplot(fig)

    # ✅ Categorize diseases based on confidence level

    high_confidence_diseases = [
        (d, pred_diseases.get(d, 0), auc_scores.get(d, 0.75))
        for d in all_diseases if pred_diseases.get(d, 0) >= 0.5 and auc_scores.get(d, 0.75) >= 0.85
    ]

    medium_confidence_diseases = [
        (d, pred_diseases.get(d, 0), auc_scores.get(d, 0.75))
        for d in all_diseases if (
            (0.3 <= pred_diseases.get(d, 0) < 0.5 and auc_scores.get(d, 0.75) >= 0.75)  # Prob 30-50% but strong AUC
            or (pred_diseases.get(d, 0) >= 0.5 and auc_scores.get(d, 0.75) < 0.85)  # Prob > 50% but weaker AUC
        )
    ]

    low_confidence_diseases = [
        (d, pred_diseases.get(d, 0), auc_scores.get(d, 0.75))
        for d in all_diseases if (
            (0.25 <= pred_diseases.get(d, 0) < 0.5 and auc_scores.get(d, 0.75) < 0.75)  # Prob 25-50% but weak AUC
            or (pred_diseases.get(d, 0) < 0.3 and auc_scores.get(d, 0.75) >= 0.85)  # Prob < 30% but strong AUC
        )
    ]



    # ✅ Generate final summary message
    summary_message = f"Out of 14 diseases, "
    summary_parts = []

    if high_confidence_diseases:
        summary_parts.append(f"{len(high_confidence_diseases)} detected with high confidence")
    if medium_confidence_diseases:
        summary_parts.append(f"{len(medium_confidence_diseases)} detected with medium confidence")
    if low_confidence_diseases:
        summary_parts.append(f"{len(low_confidence_diseases)} detected with low confidence")

    summary_message += " and ".join(summary_parts) + "."

    # ✅ Display the summary in Streamlit
    if high_confidence_diseases:
        st.success(f"✅ {summary_message}")
    elif medium_confidence_diseases:
        st.warning(f"⚠️ {summary_message}")
    else:
        st.info(f"ℹ️ {summary_message}")
        
    # ✅ Define Border Styling
    border_style = """
        <style>
        .bordered-box {
            border: 2px solid #4CAF50; 
            padding: 15px;
            margin: 10px 0;
            border-radius: 10px;
            background-color: #f9f9f9;
            box-shadow: 2px 2px 10px rgba(0,0,0,0.1);
        }
        .box-title {
            font-size: 18px;
            font-weight: bold;
            text-align: center;
            color: #2E7D32;
            margin-bottom: 10px;
        }
        </style>
    """
    st.markdown(border_style, unsafe_allow_html=True)
    
    st.markdown(f'<div class="box-title">📊 Confidence Buckets for Disease Detection</div>', unsafe_allow_html=True)

    # ✅ Display Confidence Buckets with Borders
    def display_confidence_bucket(title, diseases, color):
        if diseases:
            st.markdown(f'<div class="bordered-box" style="border-color: {color};">', unsafe_allow_html=True)
            st.markdown(f"### {title}")
            for disease, prob, auc in diseases:
                st.write(f"- **{disease}** → Probability: {prob*100:.2f}%, AUC Score: {auc:.3f}")
            st.markdown("</div>", unsafe_allow_html=True)

    # ✅ Render Sections
    display_confidence_bucket("✅ High Confidence Diseases (AUC ≥ 0.85 and Probability ≥ 50%)", high_confidence_diseases, "#4CAF50")
    display_confidence_bucket("⚠️ Medium Confidence Diseases (AUC ≥ 0.75 and Probability between 30-50%) OR (AUC < 0.85 and Probability ≥ 50%)", medium_confidence_diseases, "#FFC107")
    display_confidence_bucket("ℹ️ Low Confidence Diseases (AUC < 0.75 and Probability between 25-50%) OR (AUC ≥ 0.85 but Probability < 30%)", low_confidence_diseases, "#2196F3")
        

        
# ✅ Authentication System
if "logged_in" not in st.session_state:
    st.session_state["logged_in"] = False

if "physician_info" not in st.session_state:
    st.session_state["physician_info"] = None

if "page" not in st.session_state:
    st.session_state["page"] = "Signup"



def signup():
    # st.title("Physician Signup")
    # ✅ Centered Title at the Top
    st.markdown(
        """
        <h1 style='text-align: center; font-size: 160px; font-weight: bold;
                color: black; font-family: Arial, sans-serif;
                letter-spacing: 2px; text-shadow: 3px 3px 5px rgba(0,0,0,0.2);'>
            MedScan AI
        </h1>
        <h2 style='text-align: center; font-size: 60px; font-weight: bold;
                color: blue; font-family: Arial, sans-serif;
                letter-spacing: 1px; text-shadow: 2px 2px 4px rgba(0,0,0,0.1);'>
            Smart AI for Chest X-ray Diagnosis
        </h2>
        """,
        unsafe_allow_html=True
    )


    # ✅ Make "Physician Signup" Smaller
    st.markdown(
        """
        <h3 style='text-align: left; font-size: 22px; font-weight: bold;'>
            Physician Signup
        </h3>
        """,
        unsafe_allow_html=True
    )


    physician_name = st.text_input("Lab Physician Name")
    medical_title = st.text_input("Medical Title")
    phy_hospital_clinic = st.text_input("Clinic Name")
    email = st.text_input("Email")
    password = st.text_input("Password", type="password")

    if st.button("Sign Up"):
        if session.query(Physician).filter_by(email=email).first():
            st.error("❌ Email already registered. Please log in.")
        else:
            try:
                # ✅ Ensure all fields are passed correctly
                new_physician = Physician(
                    physician_name=physician_name.strip(),
                    medical_title=medical_title.strip(),  # ✅ Fix: Ensure non-empty values
                    phy_hospital_clinic=phy_hospital_clinic.strip(),
                    email=email.strip(),  # ✅ Fix: Strip extra spaces
                    password=hash_password(password.strip())  # ✅ Ensure password is hashed
                )

                session.add(new_physician)
                session.commit()  # ✅ Commit to database

                st.success("✅ Signup successful! You can now log in.")

                # ✅ Redirect to Login Page
                st.session_state["page"] = "Login"
                st.rerun()

            except Exception as e:
                session.rollback()  # ✅ Rollback if error occurs
                st.error(f"❌ Signup failed. Error: {str(e)}")

    if st.button("Already have an account? Log in here."):
        st.session_state["page"] = "Login"
        st.rerun()



# ✅ Login Page
def login():
    st.title("Physician Login")

    email = st.text_input("Email")
    password = st.text_input("Password", type="password")

    if st.button("Login"):
        physician = session.query(Physician).filter_by(email=email, password=hash_password(password)).first()
        if physician:
            st.session_state["logged_in"] = True
            st.session_state["physician_info"] = {
                "name": physician.physician_name,
                "title": physician.medical_title,
                "hospital": physician.phy_hospital_clinic,
                "email": physician.email
            }
            st.success(f"✅ Welcome, Dr. {physician.physician_name}!")
            st.session_state["page"] = "Main"
            st.rerun()
        else:
            st.error("❌ Invalid email or password.")

# ✅ Sign-Off Function (Updates Database)


def sign_off_patient(patient_id, physician_info, ai_report, physician_report):
    """Updates patient_predictions with physician details and marks case as processed."""

    try:
        patient = get_patient_details(patient_id)
        if not patient:
            st.error("❌ Patient ID not found.")
            return

        prediction_record = get_patient_predictions(patient_id)

        if prediction_record:
            print("✅ Existing prediction found. Updating record...")
            # ✅ Update existing prediction record
            prediction_record.physician_name = physician_info["name"]
            prediction_record.physician_email = physician_info["email"]
            prediction_record.medical_title = physician_info["title"]
            prediction_record.phy_hospital_clinic = physician_info["hospital"]
            prediction_record.prescription_report_ai_gen = ai_report
            prediction_record.prescription_report_physician = physician_report
            prediction_record.case_processed = True

        else:
            print("⚠️ No existing prediction found. Creating new record...")
            # ✅ Insert new record if it doesn't exist
            new_prediction = PatientPrediction(
                patient_id=patient_id,
                physician_name=physician_info["name"],
                physician_email=physician_info["email"],
                medical_title=physician_info["title"],
                phy_hospital_clinic=physician_info["hospital"],
                prediction="Completed",
                prescription_report_ai_gen=ai_report,
                prescription_report_physician=physician_report,
                case_processed=True
            )
            session.add(new_prediction)

        # ✅ Update `case_processed` in `patient_records`
        session.execute(
            update(PatientRecord)
            .where(PatientRecord.patient_id == patient_id)
            .values(case_processed=True)
        )

        session.commit()  # ✅ Commit all at once

    except Exception as e:
        session.rollback()
        st.error(f"❌ Database update failed: {e}")
        print(f"❌ Database update failed: {e}")  # Debugging log


# ✅ Main Application
def main_app():
    st.sidebar.write(f"👨‍⚕️ **{st.session_state['physician_info']['name']}**, {st.session_state['physician_info']['title']}")
    st.sidebar.write(f"🏥 **{st.session_state['physician_info']['hospital']}**")

    if st.sidebar.button("Logout"):
        st.session_state["logged_in"] = False
        st.session_state["physician_info"] = None
        st.session_state["page"] = "Login"
        st.rerun()
        

    st.title("MedScan AI - Physician Dashboard")
    patient_id = st.text_input("Enter Patient ID")
    
    # ✅ Ensure session state variables exist before use
    if "ai_report" not in st.session_state:
        st.session_state["ai_report"] = ""
    
    if "physician_report" not in st.session_state:
        st.session_state["physician_report"] = ""
    

    if st.button("Fetch Details"):
        patient = get_patient_details(patient_id)
        
        if patient:
            st.success(f"✅ Patient Found: {patient.patient_name}, {patient.age} years old, {patient.gender}")
            
            final_image_path = patient.image_path
            
            # Step 1: Determine file type
            file_extension = os.path.splitext(final_image_path)[-1].lower()

            # Convert image type from dcm to jpg
            if file_extension == ".dcm":
                # Convert DICOM to standard image
                final_image_path = convert_dicom_to_image(final_image_path)
                st.success("✅ **DICOM image detected and processed successfully.**")
        
            st.image(final_image_path, caption="Uploaded X-ray", width=500)  # Adjust width as needed
            
            # ✅ Validate X-ray
            # validation_result, confidence = validate_xray(Image.open(patient.image_path))
            validation_result, confidence = validate_xray(Image.open(final_image_path).convert("RGB"))

            
            if validation_result == "Chest X-ray":
                st.success(f"✅ Valid Chest X-ray (AI Confidence Score: {confidence:.2%})")

                # ✅ Predict Disease
                probabilities, detected_diseases, pred_diseases, all_predictions = predict_disease(Image.open(final_image_path))
                # Display Results

                tab1, tab2, tab3, tab4 = st.tabs(["🩺 Results", "🧠 AI Confidence vs. Disease Severity", "🔍 Disease Detection", "📄 Prescription Report"])
                
                with tab1:
                        
                    st.subheader("📊 Prediction Probabilities")
                    plot_disease_probabilities(probabilities)

                    # ✅ Get "No Finding" probability
                    no_finding_prob = detected_diseases.get("No Finding", 0)

                    # ✅ Get the 14 disease probabilities
                    disease_probs = {disease: prob for disease, prob in detected_diseases.items() if disease != "No Finding"}

                    # ✅ Check if any disease is ≥ 0.5
                    diseases_above_50 = {disease: prob for disease, prob in disease_probs.items() if prob >= 0.5}

                    # ✅ Case 1: If any disease (excluding "No Finding") has prob ≥ 0.5 → Print all 14 classes
                    if diseases_above_50:
                        st.markdown("### ✅ **Detected Diseases (Including All 14 Classes)**")
                        for disease in disease_probs.keys():  # Print all 14 classes
                            prob = detected_diseases[disease]
                            st.markdown(f"**🩺 {disease}:** {prob * 100:.2f}%")
                        st.markdown("---")  # Separator
                        st.markdown("For an in-depth analysis of model confidence scores across all diseases, visit the '🧠 AI Confidence vs. Disease Severity' tab.")
                        st.markdown("---")  # Separator

                    # ✅ Case 2: If "No Finding" is ≥ 0.5 but some other disease is also ≥ 0.5 → Ignore "No Finding" & print all 14 classes
                    elif no_finding_prob >= 0.5 and diseases_above_50:
                        st.markdown("### ✅ **Detected Diseases (Ignoring 'No Finding')**")
                        for disease in disease_probs.keys():  # Print all 14 classes
                            prob = detected_diseases[disease]
                            st.markdown(f"**🩺 {disease}:** {prob * 100:.2f}%")
                        st.markdown("---")  # Separator
                        st.markdown("For an in-depth analysis of model confidence scores across all diseases, visit the '🧠 AI Confidence vs. Disease Severity' tab.")
                        st.markdown("---")  # Separator

                    # ✅ Case 3: If "No Finding" is ≥ 0.5 and no disease is ≥ 0.5 → Print "No Disease Detected"
                    elif no_finding_prob >= 0.5 and not diseases_above_50:
                        st.markdown("### ✅ **No Disease Detected**")
                        st.markdown(f"🟢 **No Finding:** {no_finding_prob * 100:.2f}%")
                        st.markdown("The model is confident that no disease is present.")
                        st.markdown("---")  # Separator
                        st.markdown("For an in-depth analysis of model confidence scores across all diseases, visit the '🧠 AI Confidence vs. Disease Severity' tab.")
                        st.markdown("---")  # Separator

                    # ✅ Case 4: If everything (including "No Finding") is below 0.5 → Print "Uncertain Prediction"
                    else:
                        st.markdown("### ⚠️ **Uncertain Prediction**")
                        st.markdown(
                            "The model is uncertain about the results, as no disease or 'No Finding' has a strong confidence score."
                        )
                        st.markdown("Consider additional testing for a more confident diagnosis.")
                        st.markdown("---")  # Separator
                        st.markdown("For an in-depth analysis of model confidence scores across all diseases, visit the '🧠 AI Confidence vs. Disease Severity' tab.")
                        st.markdown("---")  # Separator
                        
                with tab2:
                    st.subheader("🚦 AI Confidence & Disease Risk: What the Model Sees 🤖")
                    st.write("**This plot visualizes AI's confidence in detecting diseases based on model probability and AUC scores. Larger, red bubbles indicate highly probable and confident detections, while smaller, gray/orange bubbles represent lower confidence or borderline cases.**")

                    # Filter diseases with probability > 0.2 (to avoid noise)
                    prediction_of_disease = {disease: prob for disease, prob in all_predictions.items()}
                
                    plot_disease_confidence_bubble(prediction_of_disease)


                with tab3:
                    
                    st.write("Disease localization analysis is only generated for diseases detected with a high model confidence score.")
                    st.markdown("---")  # Separator
                    st.subheader("🔬 Disease Localization (Contour-based)")

                    image_path = patient.image_path

                    # Filter diseases with probability > 0.5
                    prediction_of_disease = {disease: prob for disease, prob in detected_diseases.items() if prob > 0.5}

                    if prediction_of_disease:
                        for class_name, class_prob in prediction_of_disease.items():  
                            # Display subheading for each detected disease
                            st.write(f"**Disease:** {class_name} ({class_prob:.2f})") 
                            
                            # Generate and display Grad-CAM Contour
                            gradcam_with_contour(image_path, class_name, model)
                            st.divider()  # Divider for separation

                        st.subheader("🔥 Disease Localization (Heatmap-based)")

                        for class_name, class_prob in prediction_of_disease.items():
                            # Display subheading for each detected disease
                            st.write(f"**Disease:** {class_name} ({class_prob:.2f})")  
                            
                            # Generate and display Grad-CAM Heatmap
                            gradcam_with_heatmap(image_path, class_name, model)
                            st.divider()  # Divider for separation

                    else:
                        st.write("✅ No disease detected above the 0.5 probability threshold.")

                with tab4:
                    
                    # ✅ AI-Generated Prescription
                    st.subheader("📄 AI-Powered Prescription Report (Generated using OpenAI GPT-4o-mini)")
                    st.write(
                        "**This is an AI-generated report created after feeding the model probability scores to OpenAI. This is not a final diagnosis.**"
                    )
                    st.session_state["ai_report"] = generate_prescription_report(pred_diseases, auc_scores, patient.symptoms)
                    st.markdown(f"```\n{st.session_state['ai_report']}\n```")

                    st.divider()  # Add a horizontal divider for separation

                    # ✅ Physician's Input for Manual Prescription (Displayed Below AI Report)
                    def update_physician_report():
                        st.session_state["physician_report"] = st.session_state["physician_input"]

                    st.subheader("Physician’s Prescription")
                    st.text_area(
                        "Enter Physician’s Report",
                        value=st.session_state["physician_report"],
                        key="physician_input",
                        on_change=update_physician_report
                    )  # ✅ Now updates session state dynamically

                    
            else:
                st.error(f"❌ Invalid Image: Not a Chest X-ray (Confidence: {confidence:.2%})")
                st.warning("Please upload a valid Chest X-ray image.")
        else:
            st.error("❌ Patient ID not found.")
    
    if st.sidebar.button("Sign Off Case"):
        print("✅ Button Clicked in Terminal!")  # Debugging

        # Retrieve stored AI and physician reports
        ai_report = st.session_state["ai_report"]
        physician_report = st.session_state["physician_report"]

        # ✅ Ensure reports exist before calling function
        if not ai_report:
            st.error("❌ No AI report generated. Please fetch details first.")
        elif not physician_report:
            st.warning("⚠️ No physician report provided. Signing off without physician input.")

        sign_off_patient(patient_id, st.session_state["physician_info"], ai_report, physician_report)
        st.success("✅ Case signed off successfully!")
        
    if st.sidebar.button("Email Report to the doctor"):
        st.markdown("✅ Report emailed to the doctor")  # Debugging
    if st.sidebar.button("Email Report to the patient"):
        st.markdown("✅ Report emailed to the patient")  # Debugging
        
    st.sidebar.warning("⚠️ **Disclaimer:** MedScan AI provides an AI-powered preliminary analysis of chest X-rays. While it can assist in identifying potential conditions, it should be considered **only as the first step of diagnosis**. A certified medical professional should be consulted for a comprehensive evaluation and final assessment.")

# ✅ Page Navigation Logic
if st.session_state["page"] == "Signup":
    signup()
elif st.session_state["page"] == "Login":
    login()
elif st.session_state["logged_in"]:
    main_app()
else:
    st.session_state["page"] = "Login"
    st.rerun()
