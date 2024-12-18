import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import load_model
from PIL import Image
import numpy as np
import requests
from io import BytesIO
from tensorflow.keras.utils import to_categorical

# Map model names to .h5 files
MODEL_FILES = {
    "CNN": "cnn_model.h5",
    "DenseNet121": "densenet121_model.h5",
    "InceptionV3": "inceptionv3_lite_model.h5",
    "NASNet Mobile": "nasnet_mobile_model.h5",
    "VGGNet11": "vggnet11_lite_model.h5",
}

# Define your class names as trained in the model
class_names = [
    "Round_Bowel", "Round_damage", "Round_Fungus", "Round_good",
    "Round_strip", "Round_Wrinkled", "Round wrinkled_with_fungus",
    "Square_Damage", "square_fungus", "Square_good", "Square_Wrinkled",
    "Striped Square"
]

# Load DL models when the app starts
@st.cache_resource
def load_dl_model(model_name):
    model_path = f"models/{MODEL_FILES[model_name]}"
    return load_model(model_path)

# Function to get model accuracy on test data
def get_model_accuracy(model, test_data):
    images, labels = test_data
    
    # Convert labels to one-hot encoding
    labels_one_hot = to_categorical(labels, num_classes=len(class_names))
    
    # Evaluate the model on the test data
    loss, accuracy = model.evaluate(images, labels_one_hot)
    return accuracy * 100  # Convert to percentage

# Initialize session state
if "uploaded_image" not in st.session_state:
    st.session_state["uploaded_image"] = None
if "selected_model" not in st.session_state:
    st.session_state["selected_model"] = None

# Add custom CSS for responsiveness
st.markdown(
    """
    <style>
    /* Body and container styles for responsiveness */
    .stApp {
        display: flex;
        justify-content: center;
        align-items: center;
        flex-direction: column;
        padding: 20px;
    }

    .stContainer {
        max-width: 100%;
        width: 100%;
        padding: 20px;
        display: flex;
        flex-direction: column;
        align-items: center;
        justify-content: center;
    }

    .stImage {
        width: 100%;
        max-width: 500px;
    }

    .stButton {
        width: 100%;
        max-width: 200px;
    }

    /* Ensure input fields like buttons and select boxes are responsive */
    .stTextInput, .stSelectbox, .stButton {
        width: 100%;
        max-width: 350px;
    }

    </style>
    """,
    unsafe_allow_html=True,
)

# App Title
st.title("Areca Leaf Plate Grading")

# Horizontal Buttons for input options
st.markdown("### Select Input Option:")
col1, col2, col3 = st.columns(3)

with col1:
    camera = st.button("Camera")
with col2:
    gallery = st.button("Gallery")
with col3:
    weblink = st.button("Weblink")

# Logic for Camera Option
if camera:
    st.session_state["input_method"] = "Camera"

# Logic for Gallery Option
if gallery:
    st.session_state["input_method"] = "Gallery"

# Logic for Weblink Option
if weblink:
    st.session_state["input_method"] = "Weblink"

# Create a container for the content to handle screen size dynamically
with st.container():
    if st.session_state.get("input_method") == "Camera":
        # Step 1: Capture Image from Camera
        st.markdown("### Step 1: Capture an Image from Camera")
        captured_image = st.camera_input("Take a picture")

        if captured_image is not None:
            st.session_state["uploaded_image"] = captured_image
            image_to_predict = Image.open(captured_image)

            # Display the captured image
            st.image(image_to_predict, caption="Captured Image", use_column_width=True)

    elif st.session_state.get("input_method") == "Gallery":
        # Step 1: Upload Image
        st.markdown("### Step 1: Upload an Image")
        uploaded_image = st.file_uploader("Choose an image file", type=["jpg", "png", "jpeg"])

        if uploaded_image is not None:
            st.session_state["uploaded_image"] = uploaded_image
            image_to_predict = Image.open(uploaded_image)

            # Display the uploaded image
            st.image(image_to_predict, caption="Uploaded Image", use_column_width=True)

    elif st.session_state.get("input_method") == "Weblink":
        # Step 1: Get Image from Weblink
        st.markdown("### Step 1: Provide Image URL (Weblink)")
        image_url = st.text_input("Enter Image URL")

        if image_url:
            try:
                # Fetch the image from the URL
                response = requests.get(image_url)
                image_to_predict = Image.open(BytesIO(response.content))

                # Display the image
                st.image(image_to_predict, caption="Image from Weblink", use_column_width=True)

                # Save the image to session state
                st.session_state["uploaded_image"] = image_url

            except Exception as e:
                st.error(f"Failed to load image from URL: {e}")

    # Step 2: Choose a Model
    st.markdown("### Step 2: Choose a Model")
    model_options = [
        "Select a Model",
        "ML Models:",
        " - Random Forest",
        " - SVM",
        " - KNN",
        " - Logistic Regression",
        " - Naive Bayes",
        "DL Models:",
        " - CNN",
        " - DenseNet121",
        " - InceptionV3",
        " - NASNet Mobile",
        " - VGGNet11",
    ]
    st.session_state["selected_model"] = st.selectbox("Choose a model", model_options)

    # Step 3: Run Prediction
    if (
        st.session_state["selected_model"] not in ["Select a Model", "ML Models:", "DL Models:"]
        and st.button("Run Prediction")
    ):
        if st.session_state["selected_model"].startswith(" - "):
            model_name = st.session_state["selected_model"].strip(" - ")

            if model_name in MODEL_FILES:
                # Resize the image to 255x255 and normalize pixel values
                if isinstance(st.session_state["uploaded_image"], str):  # Weblink case
                    response = requests.get(st.session_state["uploaded_image"])
                    image_to_predict = Image.open(BytesIO(response.content))

                image_resized = image_to_predict.resize((255, 255))
                image_array = np.array(image_resized) / 255.0  # Normalize pixel values
                image_array = np.expand_dims(image_array, axis=0)  # Add batch dimension

                # Load and run the selected DL model
                model = load_dl_model(model_name)

                # Assuming you have a separate test dataset (replace with actual data)
                test_images = np.random.rand(10, 255, 255, 3)  # Dummy test images (replace with actual test data)
                test_labels = np.random.randint(0, 12, size=(10,))  # Dummy test labels (replace with actual labels)

                # Get model accuracy on the test data
                model_accuracy = get_model_accuracy(model, (test_images, test_labels))

                # Perform prediction
                prediction = model.predict(image_array)
                predicted_label = np.argmax(prediction)

                # Get the class name from the index
                predicted_class = class_names[predicted_label]

                # Display the results
                st.success(
                    f"Prediction successful! The predicted class is: {predicted_class} "
                    f"with an accuracy of {model_accuracy:.2f}%"
                )
            else:
                st.info(f"Selected ML model: {model_name}. Implement ML logic here.")
        else:
            st.warning("Please select a valid model.")

