import streamlit as st
from PIL import Image
import numpy as np
import cv2
import os
from pathlib import Path
from ultralytics import YOLO

# Configure the Streamlit App page
st.set_page_config(
    page_title="YOLOv8 Bone Fracture Detection",
    page_icon="🔍",
    layout="wide"
)

# Title and description of the Streamlit App
st.title("YOLOv8 Bone Fracture Detection")
st.markdown("Upload an image to detect bone fracture in your X-ray image.")

# Function to load our trained YOLOv8 model
@st.cache_resource
def load_model():
    # Set up model path, in this case we are using the trained model we downloaded from Google Colab named "best.pt"
    model_path = "best.pt" 
    
    # Load the model
    try:
        model = YOLO(model_path)
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

# Function to perform detection
def detect_objects(image, model):
    # Perform inference with YOLOv8
    results = model.predict(image, conf=0.25)  # Confidence Threshold is chosen based on the F1 Confidence curve peak
    
    # Making sure there is no error if the model does not come up with any results
    return results[0] if results else None

# Function to display results
def display_results(result, orig_img):
    if result is None:
        st.warning("No detection results to display")
        return
        
    # Plot bounding boxes directly on the result image
    rendered_img = result.plot()
    rendered_img = cv2.cvtColor(rendered_img, cv2.COLOR_BGR2RGB)
    
    # Display results as a comparison between original and result images
    col1, col2 = st.columns(2)
    
    # 1st column to display the original image
    with col1:
        st.subheader("Original Image")
        st.image(orig_img, use_column_width=True)
    
    # 2nd column to display the resulted image
    with col2:
        st.subheader("Detection Results")
        st.image(rendered_img, use_column_width=True)
    
    # Display detection information
    st.subheader("Detection Details")
    
    # To display the information, we need to convert the bounding boxes into dataframes
    if len(result.boxes) > 0:
        # Extract details from result
        boxes = result.boxes.cpu().numpy()
        
        # Create data array to display in the table
        data = []
        for i, box in enumerate(boxes):
            # Get the class name
            if result.names and int(box.cls[0]) in result.names:
                class_name = result.names[int(box.cls[0])]
            else:
                class_name = f"Class {int(box.cls[0])}"
                
            # Get coordinates of the bounding box and confidence threshold of the result
            x1, y1, x2, y2 = box.xyxy[0]
            confidence = box.conf[0]
            
            data.append({
                "ID": i,
                "Class": class_name,
                "Confidence": f"{confidence:.2f}",
                "Coordinates": f"({x1:.1f}, {y1:.1f}), ({x2:.1f}, {y2:.1f})"
            })
        
        # Display as dataframe
        import pandas as pd
        st.dataframe(pd.DataFrame(data))
    else:
        st.info("No fractures detected")

# Main function
def main():
    # Load our model
    model = load_model()
    
    if model is None:
        st.warning("Ensure your model file is in the correct location and format.")
        return
    
    # Upload the image that you want to detect
    uploaded_file = st.file_uploader("Choose a X-ray image...", type=["jpg", "jpeg", "png"])
    
    if uploaded_file is not None:
        # Read and display the image
        image = Image.open(uploaded_file)
        if image.mode == "RGBA":
            image = image.convert("RGB")
        
        # Convert PIL Image to numpy array for detection*
        img_array = np.array(image)
        
        # Perform detection when user clicks the button
        if st.button("Detect fractures"):
            with st.spinner("Detecting fractures..."):
                # Perform detection
                results = detect_objects(img_array, model)
                
                # Display results
                display_results(results, image)

if __name__ == "__main__":
    main()