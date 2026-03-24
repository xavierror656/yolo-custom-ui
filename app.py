import streamlit as st
import cv2
import numpy as np
import tempfile
import time
from pathlib import Path
from PIL import Image
from ultralytics import YOLO

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="YOLO Inference UI",
    page_icon="🎯",
    layout="wide",
)

# ── Constants ─────────────────────────────────────────────────────────────────
MODELS = {
    "YOLO26n (nano, fastest)": "yolo26n.pt",
    "YOLO26s (small)": "yolo26s.pt",
    "YOLO26m (medium)": "yolo26m.pt",
    "YOLO26l (large)": "yolo26l.pt",
    "YOLO11n (nano)": "yolo11n.pt",
    "YOLO11s (small)": "yolo11s.pt",
    "YOLO11m (medium)": "yolo11m.pt",
    "YOLOE-s (open-vocab)": "yoloe-s.pt",
    "YOLOE-m (open-vocab)": "yoloe-m.pt",
    "Custom model": "custom",
}

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("⚙️ Configuration")

    # Model selector
    model_label = st.selectbox("Model", list(MODELS.keys()))
    model_key = MODELS[model_label]

    if model_key == "custom":
        custom_path = st.text_input("Path to .pt file", placeholder="/path/to/model.pt")
        model_key = custom_path

    # Inference params
    st.subheader("Inference Parameters")
    conf = st.slider("Confidence threshold", 0.01, 1.0, 0.25, 0.01)
    iou = st.slider("IoU threshold (NMS)", 0.01, 1.0, 0.45, 0.01)
    imgsz = st.select_slider("Image size", options=[320, 416, 512, 640, 800, 1024], value=640)
    device = st.selectbox("Device", ["cpu", "0", "1", "2", "3"], index=0)

    # Display options
    st.subheader("Display")
    show_labels = st.toggle("Show labels", value=True)
    show_conf_val = st.toggle("Show confidence values", value=True)
    show_boxes = st.toggle("Show bounding boxes", value=True)
    line_width = st.slider("Box line width", 1, 6, 2)

    st.divider()
    st.caption("Powered by [Ultralytics](https://ultralytics.com)")


# ── Model loader (cached) ────────────────────────────────────────────────────
@st.cache_resource(show_spinner="Loading model…")
def load_model(path: str) -> YOLO:
    return YOLO(path)


# ── Inference helper ─────────────────────────────────────────────────────────
def run_inference(model: YOLO, source) -> list:
    return model.predict(
        source=source,
        conf=conf,
        iou=iou,
        imgsz=imgsz,
        device=device,
        show_labels=show_labels,
        show_conf=show_conf_val,
        show_boxes=show_boxes,
        line_width=line_width,
        verbose=False,
    )


def result_to_image(result) -> np.ndarray:
    """Return annotated BGR image from a YOLO result."""
    return result.plot(
        labels=show_labels,
        conf=show_conf_val,
        boxes=show_boxes,
        line_width=line_width,
    )


def detection_table(results: list):
    """Build a summary dict from results list."""
    rows = []
    for r in results:
        if r.boxes is None:
            continue
        for box in r.boxes:
            cls_id = int(box.cls[0])
            rows.append(
                {
                    "Class": r.names[cls_id],
                    "Confidence": f"{float(box.conf[0]):.2%}",
                    "x1": int(box.xyxy[0][0]),
                    "y1": int(box.xyxy[0][1]),
                    "x2": int(box.xyxy[0][2]),
                    "y2": int(box.xyxy[0][3]),
                }
            )
    return rows


# ── Main UI ───────────────────────────────────────────────────────────────────
st.title("🎯 YOLO Inference UI")
st.caption(f"Model: `{model_key}` · conf={conf} · iou={iou} · imgsz={imgsz} · device={device}")

# Load model
if not model_key or model_key == "custom":
    st.warning("Select or enter a model path in the sidebar.")
    st.stop()

model = load_model(model_key)

# ── Source tabs ───────────────────────────────────────────────────────────────
tab_image, tab_video, tab_webcam, tab_url = st.tabs(
    ["🖼️ Image", "🎬 Video", "📷 Webcam", "🌐 URL / Path"]
)

# ── IMAGE ─────────────────────────────────────────────────────────────────────
with tab_image:
    uploaded = st.file_uploader(
        "Upload an image", type=["jpg", "jpeg", "png", "bmp", "webp", "tiff"]
    )
    if uploaded:
        pil_img = Image.open(uploaded).convert("RGB")
        col_orig, col_result = st.columns(2)

        with col_orig:
            st.subheader("Original")
            st.image(pil_img, use_container_width=True)

        with st.spinner("Running inference…"):
            results = run_inference(model, np.array(pil_img))

        ann = result_to_image(results[0])
        ann_rgb = cv2.cvtColor(ann, cv2.COLOR_BGR2RGB)

        with col_result:
            st.subheader("Annotated")
            st.image(ann_rgb, use_container_width=True)

        rows = detection_table(results)
        if rows:
            st.subheader(f"Detections ({len(rows)})")
            st.dataframe(rows, use_container_width=True)
        else:
            st.info("No detections above threshold.")


# ── VIDEO ─────────────────────────────────────────────────────────────────────
with tab_video:
    vid_file = st.file_uploader("Upload a video", type=["mp4", "avi", "mov", "mkv", "webm"])
    if vid_file:
        # Save to temp file
        tfile = tempfile.NamedTemporaryFile(delete=False, suffix=Path(vid_file.name).suffix)
        tfile.write(vid_file.read())
        tfile.flush()

        st.video(vid_file)

        if st.button("▶️ Run inference on video"):
            cap = cv2.VideoCapture(tfile.name)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = cap.get(cv2.CAP_PROP_FPS) or 25

            # Output temp file
            out_tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            writer = cv2.VideoWriter(out_tmp.name, fourcc, fps, (w, h))

            frame_ph = st.empty()
            prog = st.progress(0, text="Processing frames…")
            t0 = time.time()

            frame_idx = 0
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = run_inference(model, rgb)
                ann = result_to_image(results[0])
                writer.write(ann)

                # Preview every 10 frames
                if frame_idx % 10 == 0:
                    frame_ph.image(cv2.cvtColor(ann, cv2.COLOR_BGR2RGB), use_container_width=True)

                frame_idx += 1
                if total_frames > 0:
                    prog.progress(frame_idx / total_frames, text=f"Frame {frame_idx}/{total_frames}")

            cap.release()
            writer.release()
            elapsed = time.time() - t0
            prog.empty()
            frame_ph.empty()

            st.success(f"Done! {frame_idx} frames in {elapsed:.1f}s ({frame_idx/elapsed:.1f} FPS)")
            with open(out_tmp.name, "rb") as f:
                st.download_button("⬇️ Download annotated video", f, file_name="annotated.mp4")


# ── WEBCAM ────────────────────────────────────────────────────────────────────
with tab_webcam:
    st.info(
        "Live webcam inference runs in real-time in your browser. "
        "Click **Start** to begin and **Stop** to end the stream."
    )
    run = st.toggle("📷 Start webcam")
    frame_ph = st.empty()
    stats_ph = st.empty()

    if run:
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            st.error("Could not open webcam. Check device access.")
        else:
            while run:
                ret, frame = cap.read()
                if not ret:
                    st.warning("Failed to read frame.")
                    break
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                t0 = time.perf_counter()
                results = run_inference(model, rgb)
                latency = (time.perf_counter() - t0) * 1000
                ann = result_to_image(results[0])
                frame_ph.image(cv2.cvtColor(ann, cv2.COLOR_BGR2RGB), use_container_width=True)
                n_det = len(results[0].boxes) if results[0].boxes else 0
                stats_ph.caption(f"Latency: {latency:.1f} ms · Detections: {n_det}")
                run = st.session_state.get("📷 Start webcam", False)
            cap.release()


# ── URL / PATH ────────────────────────────────────────────────────────────────
with tab_url:
    source_str = st.text_input(
        "Image / video URL or local path",
        placeholder="https://… or /home/user/image.jpg",
    )
    if source_str and st.button("▶️ Run"):
        with st.spinner("Running inference…"):
            results = run_inference(model, source_str)

        for i, r in enumerate(results):
            ann = result_to_image(r)
            st.image(cv2.cvtColor(ann, cv2.COLOR_BGR2RGB), caption=f"Frame {i+1}", use_container_width=True)

        rows = detection_table(results)
        if rows:
            st.subheader(f"Detections ({len(rows)})")
            st.dataframe(rows, use_container_width=True)
