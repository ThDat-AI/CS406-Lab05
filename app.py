import streamlit as st
from ultralytics import YOLO
import cv2
import numpy as np
from PIL import Image
import config

# --- Cấu hình trang ---
st.set_page_config(
    page_title="Face Mask Detection",
    page_icon="😷",
    layout="centered"
)

# --- CSS tùy chỉnh ---
st.markdown("""
    <style>
    .stApp { background-color: #f0f2f6; }
    h1 { color: #333; text-align: center; }
    </style>
    """, unsafe_allow_html=True)

# --- Load Model ---
@st.cache_resource
def load_model(model_path):
    try:
        return YOLO(model_path)
    except Exception as e:
        st.error(f"Lỗi tải model: {e}")
        return None

# --- Hàm vẽ Bounding Box ---
def plot_boxes(image_source, results, conf_threshold):
    # Nếu image_source là PIL Image thì chuyển sang numpy array
    if isinstance(image_source, Image.Image):
        img_array = np.array(image_source)
    else:
        img_array = image_source.copy() # Nếu là numpy array (từ cv2)

    for result in results:
        boxes = result.boxes
        for box in boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = float(box.conf[0])
            cls_id = int(box.cls[0])

            if conf >= conf_threshold:
                color = config.COLORS.get(cls_id, (255, 255, 255))
                label = config.CLASS_NAMES.get(cls_id, "Unknown")
                text = f"{label}: {conf:.2f}"

                cv2.rectangle(img_array, (x1, y1), (x2, y2), color, 2)
                (w, h), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
                cv2.rectangle(img_array, (x1, y1 - 20), (x1 + w, y1), color, -1)
                cv2.putText(img_array, text, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)

    return img_array

# --- Main App ---
def main():
    st.title("😷 Face Mask Detection Demo")
    
    # Sidebar
    st.sidebar.header("Cấu hình")
    mode = st.sidebar.radio("Chọn chế độ:", ["Upload Ảnh", "Chụp Ảnh (Snapshot)", "Real-time Webcam"])
    conf_threshold = st.sidebar.slider("Độ tin cậy (Threshold)", 0.0, 1.0, config.CONFIDENCE_THRESHOLD, 0.05)

    model = load_model(config.MODEL_PATH)

    if not model:
        st.warning(f"Chưa tìm thấy model tại {config.MODEL_PATH}")
        return

    # --- CHẾ ĐỘ 1: UPLOAD ẢNH ---
    if mode == "Upload Ảnh":
        uploaded_file = st.file_uploader("Tải lên ảnh...", type=['jpg', 'jpeg', 'png'])
        if uploaded_file:
            image = Image.open(uploaded_file)
            st.image(image, caption="Ảnh gốc", use_container_width=True)
            if st.button("Phát hiện"):
                results = model(image, conf=0.15, agnostic_nms=True)
                res_img = plot_boxes(image, results, conf_threshold)
                st.image(res_img, caption="Kết quả", use_container_width=True)

    # --- CHẾ ĐỘ 2: CHỤP ẢNH (SNAPSHOT) ---
    elif mode == "Chụp Ảnh (Snapshot)":
        camera_image = st.camera_input("Chụp ảnh từ webcam")
        if camera_image:
            image = Image.open(camera_image)
            results = model(image, conf=0.15, agnostic_nms=True)
            res_img = plot_boxes(image, results, conf_threshold)
            st.image(res_img, caption="Kết quả", use_container_width=True)

    # --- CHẾ ĐỘ 3: REAL-TIME WEBCAM (MỚI) ---
    elif mode == "Real-time Webcam":
        st.write("Nhấn **Start** để bật camera. Nhấn **Stop** để dừng.")
        run = st.checkbox('Bật Camera')
        
        # Tạo một khung hình trống để cập nhật liên tục
        FRAME_WINDOW = st.image([])
        
        # Khởi tạo camera (ID 0 thường là webcam mặc định)
        camera = cv2.VideoCapture(0)

        while run:
            ret, frame = camera.read()
            if not ret:
                st.error("Không thể truy cập webcam.")
                break
            
            # OpenCV dùng hệ màu BGR, cần chuyển sang RGB để hiển thị đúng
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # --- Inference (Dự đoán) ---
            # Stream=True giúp model xử lý nhanh hơn cho video
            results = model(frame, stream=True, verbose=False, conf=0.15, agnostic_nms=True)
            
            # --- Vẽ Box ---
            # Lưu ý: frame lúc này là numpy array
            processed_frame = plot_boxes(frame, results, conf_threshold)

            # --- Hiển thị lên UI ---
            FRAME_WINDOW.image(processed_frame)

        # Giải phóng camera khi tắt checkbox
        camera.release()

if __name__ == "__main__":
    main()