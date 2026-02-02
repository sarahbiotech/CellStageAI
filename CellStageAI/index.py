import streamlit as st
import cv2
import numpy as np
from PIL import Image
import tempfile
import os
import pandas as pd
from ultralytics import YOLO
import matplotlib.pyplot as plt

# ============================================
# PAGE SETUP
# ============================================
st.set_page_config(
    page_title="Blood Cell Maturation Stage AI Detection System",
    page_icon="🩸",
    layout="wide"
)

# ============================================
# MODEL CONFIGURATION
# ============================================
MODEL_PATH = "best.pt"

# Cell types with descriptions
CELL_TYPES = {
    0: {"name": "BENIGN", "color": "#27ae60", "description": "Normal, mature blood cells"},
    1: {"name": "EARLY", "color": "#3498db", "description": "Early developmental stage"},
    2: {"name": "PRE", "color": "#f39c12", "description": "Intermediate stage"},
    3: {"name": "PRO", "color": "#e74c3c", "description": "Advanced stage"}
}

# ============================================
# SIMPLE CSS STYLING
# ============================================
st.markdown("""
<style>
    .main-title {
        text-align: center;
        color: #2c3e50;
        padding: 10px 0;
    }
    .subtitle {
        text-align: center;
        color: #7f8c8d;
        margin-bottom: 20px;
    }
    .cell-info {
        background: #f8f9fa;
        padding: 15px;
        border-radius: 10px;
        margin: 10px 0;
        border-left: 4px solid;
    }
    .result-box {
        background: white;
        padding: 15px;
        border-radius: 10px;
        margin: 10px 0;
        box-shadow: 0 2px 5px rgba(0,0,0,0.1);
    }
</style>
""", unsafe_allow_html=True)

# ============================================
# HEADER SECTION
# ============================================
st.markdown('<h1 class="main-title">🩸 Leukemia AI Detection System</h1>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">AI-powered system for detecting cells in bengin/early/pre/pro stages</p>' ,
unsafe_allow_html=True)
             
# ============================================
# SIMPLE EXPLANATION SECTION
# ============================================


# ============================================
# CELL TYPES VISUALIZATION
# ============================================
st.markdown(" Detected Cell Stages:")

# Create columns for cell type display
cols = st.columns(4)
for idx, (cell_id, cell_info) in enumerate(CELL_TYPES.items()):
    with cols[idx]:
        st.markdown(
            f"<div style='text-align:center; padding:10px; border-radius:8px; background:{cell_info['color']}20; "
            f"border:2px solid {cell_info['color']}'>"
            f"<span style='color:{cell_info['color']}; font-weight:bold; font-size:20px;'>●</span><br>"
            f"<b style='color:{cell_info['color']};'>{cell_info['name']}</b><br>"
            f"<small>{cell_info['description']}</small>"
            f"</div>",
            unsafe_allow_html=True
        )

st.write("---")  # Separator line

# ============================================
# SIDEBAR
# ============================================
with st.sidebar:
    st.title("⚙️ Settings")
    
    # Confidence threshold
    confidence = st.slider("Confidence Level", 0.1, 0.9, 0.25)
    
    # About button
   
    
   
    
    # Quick cell info in sidebar
    st.write("**🔬 Cell Stages Detected:**")
    
    # Display each cell type with color and info
    cell_info_html = """
    <style>
    .cell-sidebar {
        padding: 8px;
        margin: 5px 0;
        border-radius: 6px;
        border-left: 4px solid;
    }
    </style>
    """
    
    st.markdown(cell_info_html, unsafe_allow_html=True)
    
    # BENIGN
    st.markdown(
        f'<div class="cell-sidebar" style="border-color:#27ae60; background:#27ae6010;">'
        f'<b style="color:#27ae60">BENIGN</b><br>'
        f'<small>Normal cells • Accuracy: 83.5%</small>'
        f'</div>',
        unsafe_allow_html=True
    )
    
    # EARLY
    st.markdown(
        f'<div class="cell-sidebar" style="border-color:#3498db; background:#3498db10;">'
        f'<b style="color:#3498db">EARLY</b><br>'
        f'<small>Initial stage • Accuracy: 92.2%</small>'
        f'</div>',
        unsafe_allow_html=True
    )
    
    # PRE
    st.markdown(
        f'<div class="cell-sidebar" style="border-color:#f39c12; background:#f39c1210;">'
        f'<b style="color:#f39c12">PRE</b><br>'
        f'<small>Intermediate • Accuracy: 90.2%</small>'
        f'</div>',
        unsafe_allow_html=True
    )
    
    # PRO
    st.markdown(
        f'<div class="cell-sidebar" style="border-color:#e74c3c; background:#e74c3c10;">'
        f'<b style="color:#e74c3c">PRO</b><br>'
        f'<small>Advanced stage • Accuracy: 96.2%</small>'
        f'</div>',
        unsafe_allow_html=True
    )
    
    st.markdown("---")
    
    # Quick tips
    st.write("**💡 Tips:**")
    st.write("• Start with confidence: 0.25")
    st.write("• Use clear, focused images")
    st.write("• Best for PRO/Early detection")


# ============================================
# MAIN CONTENT
# ============================================
col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("📤 Upload Image")
    
    uploaded_file = st.file_uploader(
        "Choose blood smear image",
        type=['jpg', 'jpeg', 'png', 'bmp'],
        help="Supported formats: JPG, PNG, BMP"
    )
    
    if uploaded_file:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Blood Smear", use_column_width=True)

with col2:
    st.subheader("🔍 Analysis Results")
    
    if uploaded_file and st.button("🔬 Analyze Image", type="primary", use_container_width=True):
        with st.spinner("Analyzing blood cells..."):
            try:
                # Load model
                model = YOLO(MODEL_PATH)
                
                # Save temporary image
                with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as tmp_file:
                    image.save(tmp_file.name)
                    image_path = tmp_file.name
                
                # Run prediction
                results = model.predict(
                    source=image_path,
                    conf=confidence,
                    save=False
                )
                
                result = results[0]
                
                if result.boxes is not None:
                    boxes = result.boxes.xyxy.cpu().numpy()
                    confidences = result.boxes.conf.cpu().numpy()
                    class_ids = result.boxes.cls.cpu().numpy().astype(int)
                    
                    # Count detections
                    counts = {cell_id: 0 for cell_id in CELL_TYPES.keys()}
                    for class_id in class_ids:
                        if class_id in counts:
                            counts[class_id] += 1
                    
                    # Display summary
                    st.markdown("### 📊 Detection Summary")
                    
                    total_cells = len(boxes)
                    col_a, col_b, col_c = st.columns(3)
                    with col_a:
                        st.metric("Total Cells", total_cells)
                    with col_b:
                        st.metric("Avg Confidence", f"{np.mean(confidences):.2f}")
                    with col_c:
                        detected_types = sum(1 for count in counts.values() if count > 0)
                        st.metric("Cell Types Found", detected_types)
                    
                    # Display counts per type
                    st.markdown("### Cell Counts by Stage:")
                    
                    for cell_id, count in counts.items():
                        if count > 0:
                            cell_info = CELL_TYPES[cell_id]
                            percentage = (count / total_cells * 100) if total_cells > 0 else 0
                            
                            st.markdown(f"""
                            <div class="result-box">
                                <b style="color:{cell_info['color']}; font-size:18px;">
                                {cell_info['name']}: {count} cells ({percentage:.1f}%)
                                </b><br>
                                <small>{cell_info['description']}</small>
                            </div>
                            """, unsafe_allow_html=True)
                    
                    # Show detection image
                    st.markdown("### 👁️ Detection Visualization")
                    annotated_img = result.plot()
                    st.image(annotated_img, caption="Detected Cells", use_column_width=True)
                    
                    # Optional: Download button
                  
            
                
                # Clean up
                os.unlink(image_path)
                
            except Exception as e:
                st.error(f"Error processing image: {str(e)}")
                st.info("Please ensure the model file exists at: " + MODEL_PATH)
    
    elif not uploaded_file:
        st.info("👈 **Upload an image to begin analysis**")

# ============================================
# FOOTER
# ============================================
st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: #7f8c8d; padding: 20px;'>"
    "Blood Cell Maturation Stage AI Detection System | Developed by Sarah Ali | 2026"
    "</div>",
    unsafe_allow_html=True
)

# ============================================

