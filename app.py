import streamlit as st
import torch
import torch.nn as nn
from torchvision import transforms, models
from ultralytics import YOLO
from sahi import AutoDetectionModel
from sahi.predict import get_sliced_prediction
from PIL import Image, ImageDraw
import numpy as np
import io
import os
import matplotlib.pyplot as plt

# --- CONFIGURATION (MUST MATCH TRAINING) ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
RESNET_WEIGHTS_PATH = 'models/resnet50_mitotic_plain_BEST.pth'
YOLO_MODEL_PATH = 'models/yolo_best.pt' 

# Processing params
PATCH_SIZE = 100 
CONFIDENCE_THRESHOLD = 0.25 
SLICE_HEIGHT = 640 
SLICE_WIDTH = 640
IMG_SIZE = 224

# Classification Maps
CLASS_MAP = {0: 'mitotic', 1: 'not_mitotic'}
COLOR_MAP = {'mitotic': 'red', 'not_mitotic': 'blue'}
mitotic_class_index = 0 
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


# --- 1. MODEL LOADING & CACHING (CRITICAL FOR STREAMLIT) ---

@st.cache_resource
def load_models():
    """Loads and caches both the ResNet classifier and the YOLO detector for speed."""
    
    # 1. Load ResNet50 Classifier
    model = models.resnet50(weights=None) 
    num_ftrs = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Dropout(p=0.2), 
        nn.Linear(num_ftrs, 2)
    )
    
    try:
        state_dict = torch.load(RESNET_WEIGHTS_PATH, map_location=DEVICE)
        model.load_state_dict(state_dict)
        model.to(DEVICE).eval()
    except Exception as e:
        st.error(f"Failed to load ResNet weights. Check path: {RESNET_WEIGHTS_PATH}. Error: {e}")
        st.stop()

    # 2. Load YOLO Model for SAHI
    # We load the weights file path directly into SAHI's AutoDetectionModel
    if not os.path.exists(YOLO_MODEL_PATH):
        st.error(f"YOLO weights file not found at: {YOLO_MODEL_PATH}")
        st.stop()
        
    detection_model = AutoDetectionModel.from_pretrained(
        model_type='yolov8', model_path=YOLO_MODEL_PATH, confidence_threshold=CONFIDENCE_THRESHOLD, device=DEVICE
    )
    
    return model, detection_model

# --- 2. PREPROCESSING ---

def get_transforms():
    """Returns the transformation pipeline for the ResNet classifier."""
    return transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(), 
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
    ])


# --- 3. CORE PROCESSING FUNCTION ---

def process_image_pipeline(uploaded_file, resnet_model, detection_model):
    """Executes the detection, cropping, and classification steps."""
    
    # 3.1. Convert uploaded file to PIL Image
    img_bytes = uploaded_file.read()
    original_image_pil = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    img_width, img_height = original_image_pil.size
    
    st.info(f"Image loaded. Dimensions: {img_width}x{img_height}")

    # Save the file to a temporary location for SAHI to access
    temp_image_path = "temp_wsi_tile.jpg"
    original_image_pil.save(temp_image_path)

    # 3.2. YOLO + SAHI Detection
    st.subheader("1. Detecting Cell Candidates.")
    sahi_result = get_sliced_prediction(
        temp_image_path, detection_model, slice_height=SLICE_HEIGHT, slice_width=SLICE_WIDTH,
        overlap_height_ratio=0.2, overlap_width_ratio=0.2, verbose=0
    )
    detected_bboxes = [[int(c) for c in pred.bbox.to_xyxy()] for pred in sahi_result.object_prediction_list]
    
    if not detected_bboxes:
        st.warning("No cell candidates found with the current YOLO confidence threshold.")
        return original_image_pil, 0, 0, 0
    
    st.success(f"Found {len(detected_bboxes)} cell candidates.")

    # 3.3. Crop, Preprocess, and Batch
    patches_to_batch = []
    bboxes_to_process = []
    transform = get_transforms()
    
    for bbox in detected_bboxes:
        x1, y1, x2, y2 = bbox 
        center_x = (x1 + x2) / 2; center_y = (y1 + y2) / 2
        crop_x1 = int(center_x - PATCH_SIZE / 2); crop_y1 = int(center_y - PATCH_SIZE / 2)
        crop_x2 = crop_x1 + PATCH_SIZE; crop_y2 = crop_y1 + PATCH_SIZE
        
        if crop_x1 < 0 or crop_y1 < 0 or crop_x2 > img_width or crop_y2 > img_height: continue 
        
        patch_pil = original_image_pil.crop((crop_x1, crop_y1, crop_x2, crop_y2))
        patch_tensor = transform(patch_pil)
        
        patches_to_batch.append(patch_tensor)
        bboxes_to_process.append(bbox)
    
    if not patches_to_batch:
        st.warning("No valid 100x100 patches could be generated from the detections (too close to image edges).")
        return original_image_pil, 0, 0, len(detected_bboxes)

    # 3.4. Batch Classification
    st.subheader("2. Classifying Patches...")
    batch_tensor = torch.stack(patches_to_batch).to(DEVICE)

    with torch.no_grad():
        logits = resnet_model(batch_tensor)
        predicted_indices = torch.argmax(logits, dim=1).cpu().numpy().tolist()

    # 3.5. Final Visualization and Counting
    final_annotated_image = original_image_pil.copy()
    draw = ImageDraw.Draw(final_annotated_image)
    mitotic_count = 0
    non_mitotic_count = 0
    
    for i, pred_index in enumerate(predicted_indices):
        bbox = bboxes_to_process[i]
        class_name = CLASS_MAP.get(pred_index, "Unknown")
        color = COLOR_MAP.get(class_name, 'yellow') 
        
        if class_name == 'mitotic':
            mitotic_count += 1
        else:
            non_mitotic_count += 1
        
        # Draw Bounding Box and Label
        draw.rectangle(bbox, outline=color, width=3)
        draw.text((bbox[0] + 5, bbox[1] - 15), class_name.upper(), fill=color)

    os.remove(temp_image_path) # Clean up temporary file
    return final_annotated_image, mitotic_count, non_mitotic_count, len(detected_bboxes)


# --- 4. STREAMLIT APP LAYOUT ---

def main():
    st.set_page_config(page_title="Mitosis Detection & Classification", layout="wide")
    st.title("🔬 Mitotic Cell Detector")
    
    # Load models once and cache them
    resnet_model, detection_model = load_models()

    st.markdown("""
    **Accelerate your histopathological workflow by leveraging this custom deep learning pipeline designed to precisely differentiate and count mitotic and non-mitotic cells within complex Whole Slide Image (WSI) tiles**
    """)
    
    st.sidebar.header("Upload Image")
    uploaded_file = st.sidebar.file_uploader("Choose a WSI Tile (.png, .jpg, .tif)", type=["png", "jpg", "jpeg", "tif", "tiff"])

    if uploaded_file is not None:
        
        # Display raw input image
        st.subheader("Uploaded Image")
        st.image(uploaded_file, caption=f"Processing: {uploaded_file.name}", use_container_width=True)
        
        if st.button("Start Analysis", type="primary"):
            with st.spinner("Running detection and classification pipeline... This may take a moment."):
                
                # Execute pipeline
                annotated_img, m_count, nm_count, total_detected = process_image_pipeline(
                    uploaded_file, resnet_model, detection_model
                )
            
            # --- Results Display ---
            st.markdown("---")
            st.header("Analysis Results")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Detection Summary")
                st.metric("Total Cells Detected (YOLO)", total_detected)
                st.metric("Total Cells Classified", m_count + nm_count)
            
            with col2:
                st.subheader("Classification Counts")
                st.markdown(f"**Mitotic Cells (🔴):** **{m_count}**")
                st.markdown(f"**Non-Mitotic Cells (🔵):** **{nm_count}**")
                if m_count + nm_count > 0:
                    mi = (m_count / (m_count + nm_count)) * 100
                    st.metric("Approximate Mitotic Index (MI)", f"{mi:.2f}%")
                
            st.subheader("Visualized Detections")
            st.image(annotated_img, caption="Color-Coded Classification Overlay", use_container_width=True)

if __name__ == '__main__':
    main()