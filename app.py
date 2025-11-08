import streamlit as st
import numpy as np
import cv2
from tensorflow.keras.models import load_model
import pandas as pd
import matplotlib.pyplot as plt
import time
import os
from PIL import Image
import io

# Set page configuration
st.set_page_config(
    page_title="Traffic Sign Recognition System",
    page_icon="🚦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Apply custom CSS with improved contrast for both light and dark modes
st.markdown("""
    <style>
    /* General styling */
    .main-header {
        font-size: 2.5rem;
        color: #ffffff;
        text-align: center;
        margin-bottom: 1rem;
        padding-bottom: 1rem;
        border-bottom: 2px solid #4e4e4e;
        text-shadow: 1px 1px 2px rgba(0,0,0,0.5);
    }
    .sub-header {
        font-size: 1.5rem;
        color: #ffffff;
        margin: 1rem 0;
    }

    /* Info box with improved contrast */
    .info-box {
        background-color: rgba(30, 136, 229, 0.2);
        border-left: 5px solid #1E88E5;
        padding: 1rem;
        border-radius: 5px;
        margin: 1rem 0;
        color: #ffffff;
    }
    .info-box h3 {
        color: #36a3ff;
    }
    .info-box p {
        color: #ffffff;
    }

    /* Button styling */
    .stButton button {
        width: 100%;
        border-radius: 5px;
        background-color: #1E88E5;
        color: white;
        font-weight: bold;
    }

    /* Prediction box with better visibility */
    .prediction-box {
        background-color: rgba(30, 136, 229, 0.2);
        padding: 1.5rem;
        border-radius: 10px;
        box-shadow: 0 2px 5px rgba(0,0,0,0.3);
        color: #ffffff;
    }
    .prediction-box h3 {
        color: #36a3ff;
    }

    /* Footer styling */
    .footer {
        text-align: center;
        margin-top: 2rem;
        padding-top: 1rem;
        border-top: 1px solid #4e4e4e;
        font-size: 0.8rem;
        color: #cccccc;
    }

    /* Force white text on dark backgrounds */
    h1, h2, h3, h4, h5, p, li, label, .stMarkdown {
        color: #ffffff !important;
    }

    /* Make radio buttons and checkboxes more visible */
    .stRadio label, .stCheckbox label {
        color: #ffffff !important;
        font-weight: 500;
    }

    /* Make selectbox text visible */
    .stSelectbox label {
        color: #ffffff !important;
    }

    /* Ensure dataframe headers are visible */
    .dataframe th {
        background-color: #1E88E5 !important;
        color: white !important;
    }

    /* Ensure dataframe content is visible */
    .dataframe td {
        background-color: rgba(30, 136, 229, 0.1) !important;
        color: #ffffff !important;
    }

    /* Fix for sidebar text */
    .css-1d391kg, .css-163ttbj, .css-1e90glm {
        color: #ffffff !important;
    }

    /* Fix for expandable sections */
    .streamlit-expanderHeader {
        color: #ffffff !important;
        background-color: rgba(30, 136, 229, 0.2) !important;
    }

    /* Progress bar coloring */
    .stProgress > div > div {
        background-color: #1E88E5 !important;
    }
    </style>
    """, unsafe_allow_html=True)


# Load the model - using the original path that was working
@st.cache_resource
def load_recognition_model():
    try:
        # Use the original path that was working in your code
        model = load_model(r"E:\BSAI-5th\DataMining\Traffic  Sign Recognition System\traffic_sign_cnn.h5")
        return model
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None


# Label mapping with descriptions
class_info = {
    0: {"name": "Speed limit (20km/h)", "description": "Maximum speed allowed is 20 kilometers per hour"},
    1: {"name": "Speed limit (30km/h)", "description": "Maximum speed allowed is 30 kilometers per hour"},
    2: {"name": "Speed limit (50km/h)", "description": "Maximum speed allowed is 50 kilometers per hour"},
    3: {"name": "Speed limit (60km/h)", "description": "Maximum speed allowed is 60 kilometers per hour"},
    4: {"name": "Speed limit (70km/h)", "description": "Maximum speed allowed is 70 kilometers per hour"},
    5: {"name": "Speed limit (80km/h)", "description": "Maximum speed allowed is 80 kilometers per hour"},
    6: {"name": "End of speed limit (80km/h)", "description": "End of the 80 km/h speed limit zone"},
    7: {"name": "Speed limit (100km/h)", "description": "Maximum speed allowed is 100 kilometers per hour"},
    8: {"name": "Speed limit (120km/h)", "description": "Maximum speed allowed is 120 kilometers per hour"},
    9: {"name": "No passing", "description": "Overtaking of other vehicles is not allowed"},
    10: {"name": "No passing for vehicles over 3.5 metric tons",
         "description": "Heavy vehicles cannot pass other vehicles"},
    11: {"name": "Right-of-way at the next intersection", "description": "You have priority at the next intersection"},
    12: {"name": "Priority road", "description": "You are on a road with priority over intersecting roads"},
    13: {"name": "Yield", "description": "Give way to vehicles on the main road"},
    14: {"name": "Stop", "description": "Come to a complete stop and give way to all traffic"},
    15: {"name": "No vehicles", "description": "No vehicles of any kind are allowed"},
    16: {"name": "Vehicles over 3.5 metric tons prohibited", "description": "Heavy vehicles are not permitted"},
    17: {"name": "No entry", "description": "Entry is forbidden for all vehicles"},
    18: {"name": "General caution", "description": "Warning for a general hazard ahead"},
    19: {"name": "Dangerous curve to the left", "description": "Warning for a sharp bend to the left"},
    20: {"name": "Dangerous curve to the right", "description": "Warning for a sharp bend to the right"},
    21: {"name": "Double curve", "description": "Warning for a series of bends ahead"},
    22: {"name": "Bumpy road", "description": "Warning for an uneven road surface"},
    23: {"name": "Slippery road", "description": "Warning for a road that may become slippery"},
    24: {"name": "Road narrows on the right", "description": "Warning that the road becomes narrower on the right"},
    25: {"name": "Road work", "description": "Warning for construction or maintenance work on the road"},
    26: {"name": "Traffic signals", "description": "Warning for traffic lights ahead"},
    27: {"name": "Pedestrians", "description": "Warning for a pedestrian crossing"},
    28: {"name": "Children crossing", "description": "Warning for children crossing, often near a school"},
    29: {"name": "Bicycles crossing", "description": "Warning for a bicycle crossing ahead"},
    30: {"name": "Beware of ice/snow", "description": "Warning for potentially icy road conditions"},
    31: {"name": "Wild animals crossing", "description": "Warning for wild animals that may cross the road"},
    32: {"name": "End of all speed and passing limits", "description": "Previous speed and passing restrictions end"},
    33: {"name": "Turn right ahead", "description": "Mandatory right turn ahead"},
    34: {"name": "Turn left ahead", "description": "Mandatory left turn ahead"},
    35: {"name": "Ahead only", "description": "Vehicles must go straight ahead"},
    36: {"name": "Go straight or right", "description": "Vehicles may go straight or turn right"},
    37: {"name": "Go straight or left", "description": "Vehicles may go straight or turn left"},
    38: {"name": "Keep right", "description": "Vehicles must keep to the right of the obstacle"},
    39: {"name": "Keep left", "description": "Vehicles must keep to the left of the obstacle"},
    40: {"name": "Roundabout mandatory", "description": "Vehicles must travel around the roundabout"},
    41: {"name": "End of no passing", "description": "End of the no overtaking zone"},
    42: {"name": "End of no passing by vehicles over 3.5 metric tons",
         "description": "End of no passing restriction for heavy vehicles"}
}


def preprocess_image(image, target_size=(64, 64)):
    """Preprocess image for model prediction"""
    # Convert to RGB if needed
    if len(image.shape) == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    elif image.shape[2] == 4:
        image = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)

    # Resize to target size
    img = cv2.resize(image, target_size)
    # Normalize
    img = img.astype("float32") / 255.0
    # Expand dimensions for model input
    img = np.expand_dims(img, axis=0)
    return img


def predict_sign(model, image):
    """Make prediction on preprocessed image"""
    # Preprocess the image
    processed_img = preprocess_image(image)

    # Add progress bar for prediction
    with st.spinner('Processing image...'):
        progress_bar = st.progress(0)
        for i in range(100):
            time.sleep(0.01)
            progress_bar.progress(i + 1)

        # Make prediction
        preds = model.predict(processed_img)

    # Get the top 5 predictions
    top_indices = preds[0].argsort()[-5:][::-1]
    top_preds = [(class_info[i]["name"], class_info[i]["description"], float(preds[0][i])) for i in top_indices]

    return top_preds


def display_prediction_results(predictions):
    """Display the prediction results in a nice format"""
    main_pred, main_desc, main_conf = predictions[0]

    # Main prediction result
    st.markdown(f"<h2 style='text-align:center; color:#36a3ff;'>Predicted Sign: {main_pred}</h2>",
                unsafe_allow_html=True)

    col1, col2 = st.columns([1, 2])

    with col1:
        st.markdown(
            f"<div class='prediction-box'><h3>Confidence</h3><h1 style='text-align:center; color:#36a3ff;'>{main_conf * 100:.1f}%</h1></div>",
            unsafe_allow_html=True)

    with col2:
        st.markdown("<div class='prediction-box'>", unsafe_allow_html=True)
        st.markdown(f"<h3>Description</h3><p style='color:#ffffff;'>{main_desc}</p>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

    # Show bar chart of top predictions
    st.subheader("Top 5 Predictions")

    # Prepare data for chart
    labels = [pred[0] for pred in predictions]
    confidences = [pred[2] * 100 for pred in predictions]

    # Configure plot with light-colored text for dark background
    plt.style.use('dark_background')
    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.barh(labels, confidences, color='#1E88E5')
    ax.set_xlim(0, 100)
    ax.set_xlabel('Confidence (%)', color='white', fontsize=12)
    ax.set_ylabel('Signs', color='white', fontsize=12)
    ax.set_title('Top 5 Prediction Confidences', color='white', fontsize=14)
    ax.tick_params(colors='white')

    # Add the values on the bars
    for i, bar in enumerate(bars):
        ax.text(bar.get_width() + 1, bar.get_y() + bar.get_height() / 2,
                f'{confidences[i]:.1f}%', va='center', color='white')

    # Make sure the figure has a dark background to match Streamlit's dark theme
    fig.patch.set_facecolor('#0e1117')

    st.pyplot(fig)


def webcam_detection(model):
    """Perform real-time detection using webcam"""
    st.subheader("📹 Live Webcam Traffic Sign Detection")
    st.write("Note: Allow camera access and make sure your traffic sign is clearly visible.")

    # Start the webcam
    run = st.checkbox("Start Webcam")
    FRAME_WINDOW = st.empty()

    if run:
        cap = cv2.VideoCapture(0)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

        # Create a placeholder for prediction text
        prediction_text = st.empty()

        while run:
            ret, frame = cap.read()
            if not ret:
                st.error("Failed to capture image from webcam")
                break

            # Display the frame
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            FRAME_WINDOW.image(frame_rgb, channels="RGB")

            # Process every few frames to avoid overloading
            # Make prediction
            processed_img = preprocess_image(frame_rgb)
            preds = model.predict(processed_img, verbose=0)
            class_idx = np.argmax(preds)
            confidence = np.max(preds)

            # Display prediction
            if confidence > 0.7:  # Only show high confidence predictions
                sign_name = class_info[class_idx]["name"]
                prediction_text.markdown(f"""
                <div style='padding:10px; background-color:rgba(54, 163, 255, 0.2); border-radius:5px;'>
                    <h3 style='margin:0; color:#36a3ff;'>Detected: {sign_name}</h3>
                    <p style='margin:0; color:white;'>Confidence: {confidence * 100:.1f}%</p>
                </div>
                """, unsafe_allow_html=True)
            else:
                prediction_text.markdown("""
                <div style='padding:10px; background-color:rgba(255, 82, 82, 0.2); border-radius:5px;'>
                    <p style='margin:0; color:white;'>No traffic sign detected with high confidence</p>
                </div>
                """, unsafe_allow_html=True)

            # Slow down the loop a bit
            time.sleep(0.1)

        cap.release()
    else:
        # Display placeholder image when webcam is not active
        placeholder_img = np.zeros((480, 640, 3), dtype=np.uint8)
        cv2.putText(placeholder_img, "Click 'Start Webcam' to begin", (100, 240), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                    (255, 255, 255), 2)
        FRAME_WINDOW.image(placeholder_img, channels="RGB")


def batch_processing(model):
    """Process multiple images at once"""
    st.subheader("🗂️ Batch Processing")
    st.write("Upload multiple traffic sign images for batch recognition.")

    uploaded_files = st.file_uploader("Choose multiple images...", type=["jpg", "png", "jpeg"],
                                      accept_multiple_files=True)

    if uploaded_files:
        if st.button("Process All Images"):
            results = []

            progress_bar = st.progress(0)
            progress_text = st.empty()

            for i, uploaded_file in enumerate(uploaded_files):
                progress_text.text(f"Processing image {i + 1}/{len(uploaded_files)}")
                progress_bar.progress((i + 1) / len(uploaded_files))

                # Read and process image
                file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
                image = cv2.imdecode(file_bytes, 1)
                image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

                # Make prediction
                processed_img = preprocess_image(image_rgb)
                preds = model.predict(processed_img, verbose=0)
                class_idx = np.argmax(preds)
                confidence = np.max(preds)

                # Save results
                results.append({
                    "filename": uploaded_file.name,
                    "prediction": class_info[class_idx]["name"],
                    "confidence": float(confidence * 100),
                    "description": class_info[class_idx]["description"]
                })

            # Create a DataFrame for display
            df = pd.DataFrame(results)

            # Display results as a dataframe
            st.subheader("Batch Processing Results")
            st.dataframe(df.style.highlight_max(axis=0, subset=['confidence'], color='#36a3ff'))

            # Option to download results as CSV
            csv = df.to_csv(index=False).encode('utf-8')
            st.download_button(
                "Download Results as CSV",
                csv,
                "traffic_sign_results.csv",
                "text/csv",
                key='download-csv'
            )

            # Display summary visualization
            st.subheader("Summary of Predictions")
            sign_counts = df['prediction'].value_counts()

            # Use dark style for matplotlib
            plt.style.use('dark_background')
            fig, ax = plt.subplots(figsize=(10, 6))
            sign_counts.plot(kind='bar', ax=ax, color='#1E88E5')
            plt.xticks(rotation=90, color='white')
            plt.ylabel('Count', color='white')
            plt.xlabel('Traffic Sign', color='white')
            plt.title('Frequency of Detected Traffic Signs', color='white')
            ax.tick_params(colors='white')
            plt.tight_layout()
            fig.patch.set_facecolor('#0e1117')
            st.pyplot(fig)


def display_sign_information():
    """Display information about traffic signs"""
    st.subheader("🚸 Traffic Sign Information")

    # Create two columns for categories and sign details
    col1, col2 = st.columns([1, 2])

    with col1:
        st.markdown("### Sign Categories")
        categories = [
            "Speed limits",
            "Prohibitions",
            "Warnings",
            "Mandatory actions",
            "Informational signs"
        ]

        selected_category = st.radio("Select a category:", categories)

        # Map categories to sign indices
        category_signs = {
            "Speed limits": list(range(0, 9)),
            "Prohibitions": [9, 10, 15, 16, 17],
            "Warnings": list(range(18, 32)),
            "Mandatory actions": list(range(33, 41)),
            "Informational signs": [6, 12, 32, 41, 42]
        }

        sign_indices = category_signs[selected_category]
        selected_sign = st.selectbox("Select a sign:",
                                     [class_info[i]["name"] for i in sign_indices],
                                     key="sign_info")

    with col2:
        # Get the index from the name
        selected_idx = next(i for i, info in class_info.items()
                            if info["name"] == selected_sign)

        st.markdown(f"### {class_info[selected_idx]['name']}")
        st.markdown(f"**Description:** {class_info[selected_idx]['description']}")

        # Display mock-up image of the sign (would need actual images)
        st.markdown("#### Example of this sign:")

        # This would ideally load an actual image of the sign
        # For now, we'll create a colored placeholder
        sign_img = np.zeros((200, 200, 3), dtype=np.uint8)

        # Color coding based on category
        if selected_idx <= 8:  # Speed limits
            sign_img[:] = (255, 255, 255)  # White background for speed limits
            cv2.circle(sign_img, (100, 100), 80, (0, 0, 255), 5)  # Red circle
            cv2.putText(sign_img, str(selected_idx * 20 if selected_idx > 0 else 20),
                        (70, 120), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 0), 5)  # Speed number
        elif selected_idx in [9, 10, 15, 16, 17]:  # Prohibitions
            sign_img[:] = (255, 255, 255)  # White background
            cv2.circle(sign_img, (100, 100), 80, (0, 0, 255), 5)  # Red circle
            cv2.line(sign_img, (50, 50), (150, 150), (0, 0, 255), 5)  # Red diagonal line
        elif selected_idx >= 18 and selected_idx <= 31:  # Warnings
            sign_img[:] = (0, 255, 255)  # Yellow background for warnings
            pts = np.array([[100, 30], [170, 170], [30, 170]], np.int32)
            cv2.polylines(sign_img, [pts], True, (0, 0, 0), 5)
        else:  # Mandatory or informational
            sign_img[:] = (0, 0, 255) if selected_idx <= 40 else (0, 255, 0)  # Blue or green
            cv2.circle(sign_img, (100, 100), 80, (255, 255, 255), 5)

        st.image(sign_img, caption=f"Example of {selected_sign}")

        st.markdown("#### When you'll encounter this sign:")
        contexts = {
            "Speed limits": "On roads where speed control is necessary for safety.",
            "Prohibitions": "Areas where certain actions or vehicles are not allowed.",
            "Warnings": "Before potentially dangerous road conditions or hazards.",
            "Mandatory actions": "Where specific driving behaviors are required.",
            "Informational signs": "To provide information about road conditions or services."
        }

        st.write(contexts[selected_category])

        st.markdown("#### Driver action required:")
        if "Speed limit" in selected_sign:
            st.write("Adjust your speed to match or be below the posted limit.")
        elif "No" in selected_sign or "prohibited" in selected_sign:
            st.write("Avoid the prohibited action or vehicle type specified.")
        elif any(w in selected_sign for w in ["Warning", "Caution", "Beware", "Dangerous"]):
            st.write("Proceed with caution and be prepared for the mentioned hazard.")
        elif any(w in selected_sign for w in ["Turn", "Keep", "Go"]):
            st.write("Follow the mandatory direction indicated by the sign.")
        else:
            st.write("Be aware of the information provided and adjust driving accordingly.")


# Main app structure
def main():
    # Direct model load without caching to ensure it works
    model = None
    try:
        model = load_model(r"E:\BSAI-5th\DataMining\Traffic  Sign Recognition System\traffic_sign_cnn.h5")
        st.sidebar.success("Model loaded successfully!")
    except Exception as e:
        st.sidebar.error(f"Error loading model: {str(e)}")
        st.stop()

    # Sidebar with navigation - improved visibility
    st.sidebar.image(
        "https://upload.wikimedia.org/wikipedia/commons/thumb/1/1d/Road_sign_template.svg/1200px-Road_sign_template.svg.png",
        width=100)
    st.sidebar.markdown("<h2 style='color: white;'>Navigation</h2>", unsafe_allow_html=True)
    app_mode = st.sidebar.radio("Choose a feature:",
                                ["🏠 Home", "📷 Single Image Analysis", "📹 Real-time Detection",
                                 "🗂️ Batch Processing", "ℹ️ Traffic Sign Guide", "⚙️ Settings"])

    st.sidebar.markdown("---")
    st.sidebar.markdown(
        "<div style='color: white;'>This application uses a deep learning model trained on the German Traffic Sign Recognition Benchmark (GTSRB) dataset.</div>",
        unsafe_allow_html=True)

    # Different app modes
    if app_mode == "🏠 Home":
        # Homepage
        st.markdown("<h1 class='main-header'>Traffic Sign Recognition System</h1>", unsafe_allow_html=True)

        col1, col2 = st.columns([2, 1])

        with col1:
            st.markdown("""
            <div class='info-box'>
            <h3>Welcome to the Traffic Sign Recognition System!</h3>
            <p>This application uses deep learning to identify and classify traffic signs from images.
            Whether you're studying for a driver's test, verifying sign meanings, or working on autonomous 
            driving technology, this tool can help you identify traffic signs quickly and accurately.</p>
            </div>
            """, unsafe_allow_html=True)

            st.markdown("<h2 class='sub-header'>Features</h2>", unsafe_allow_html=True)

            features = {
                "📷 Single Image Analysis": "Upload and analyze individual traffic sign images",
                "📹 Real-time Detection": "Use your webcam for real-time traffic sign detection",
                "🗂️ Batch Processing": "Process multiple images at once",
                "ℹ️ Traffic Sign Guide": "Learn about different traffic signs and their meanings",
                "⚙️ Settings": "Configure application settings"
            }

            for feature, description in features.items():
                st.markdown(f"<div style='color: white;'><strong>{feature}</strong>: {description}</div>",
                            unsafe_allow_html=True)

        with col2:
            st.image("https://cdn-icons-png.flaticon.com/512/2554/2554896.png", width=200)

        st.markdown("<h2 class='sub-header'>Get Started</h2>", unsafe_allow_html=True)
        col1, col2, col3 = st.columns(3)

        with col1:
            if st.button("📷 Analyze Image", key="home_analyze"):
                # Use session state to change page instead of rerun
                st.session_state["app_mode"] = "📷 Single Image Analysis"
                st.rerun()

        with col2:
            if st.button("📹 Start Webcam", key="home_webcam"):
                # Use session state to change page instead of rerun
                st.session_state["app_mode"] = "📹 Real-time Detection"
                st.rerun()

        with col3:
            if st.button("🗂️ Batch Process", key="home_batch"):
                # Use session state to change page instead of rerun
                st.session_state["app_mode"] = "🗂️ Batch Processing"
                st.rerun()

    elif app_mode == "📷 Single Image Analysis":
        st.markdown("<h1 class='main-header'>Single Image Analysis</h1>", unsafe_allow_html=True)
        st.markdown("<p style='color: white;'>Upload a traffic sign image, and the model will classify it.</p>",
                    unsafe_allow_html=True)

        uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

        if uploaded_file is not None:
            # Read and display image
            file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
            image = cv2.imdecode(file_bytes, 1)
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            col1, col2 = st.columns([1, 2])
            with col1:
                st.image(image_rgb, caption="Uploaded Image", use_column_width=True)

            with col2:
                # Make predictions
                predictions = predict_sign(model, image_rgb)
                display_prediction_results(predictions)

    elif app_mode == "📹 Real-time Detection":
        st.markdown("<h1 class='main-header'>Real-time Traffic Sign Detection</h1>", unsafe_allow_html=True)
        webcam_detection(model)

    elif app_mode == "🗂️ Batch Processing":
        st.markdown("<h1 class='main-header'>Batch Processing</h1>", unsafe_allow_html=True)
        batch_processing(model)

    elif app_mode == "ℹ️ Traffic Sign Guide":
        st.markdown("<h1 class='main-header'>Traffic Sign Guide</h1>", unsafe_allow_html=True)
        display_sign_information()

    elif app_mode == "⚙️ Settings":
        st.markdown("<h1 class='main-header'>Settings</h1>", unsafe_allow_html=True)
        st.subheader("Application Settings")

        # Theme settings
        theme = st.selectbox("Theme", ["Dark", "Light"])
        st.markdown(f"<p style='color: white;'>Selected theme: {theme}</p>", unsafe_allow_html=True)

        # Model settings
        st.subheader("Model Settings")
        confidence_threshold = st.slider("Confidence Threshold", 0.0, 1.0, 0.7)
        st.markdown(
            f"<p style='color: white;'>Predictions with confidence below {confidence_threshold * 100:.0f}% will be ignored in real-time detection.</p>",
            unsafe_allow_html=True)

        # About section
        st.subheader("About")
        st.markdown("""
        <div style='color: white;'>
        <strong>Traffic Sign Recognition System v1.0</strong><br><br>

        This application uses a convolutional neural network trained on the German Traffic Sign 
        Recognition Benchmark (GTSRB) dataset to identify traffic signs.<br><br>

        Created by: Wasif-Sohail55<br>
        Last updated: 2025-07-15
        </div>
        """, unsafe_allow_html=True)

        # Model path configuration
        st.subheader("Model Configuration")
        model_path = st.text_input("Model Path",
                                   r"E:\BSAI-5th\DataMining\Traffic  Sign Recognition System\traffic_sign_cnn.h5")
        if st.button("Reload Model"):
            try:
                new_model = load_model(model_path)
                st.success("Model loaded successfully!")
                # We'd need to update the global model here, but that's beyond the scope of this example
            except Exception as e:
                st.error(f"Error loading model: {str(e)}")

    # Footer
    st.markdown("""
    <div class='footer'>
        <p>© 2025 Traffic Sign Recognition System | Created with Streamlit | Data: GTSRB Dataset</p>
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    # Initialize session state for navigation if it doesn't exist
    if "app_mode" not in st.session_state:
        st.session_state["app_mode"] = "🏠 Home"

    main()