import streamlit as st
import cv2
import numpy as np
import tempfile
import time
from collections import Counter
from pathlib import Path
from PIL import Image
from ultralytics import YOLO, YOLOE
from ultralytics.models.yolo.yoloe import YOLOEVPSegPredictor

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="YOLO Inference UI",
    page_icon="🎯",
    layout="wide",
)

# ── Constants ─────────────────────────────────────────────────────────────────
YOLO_MODELS = {
    "YOLO26n (nano, fastest)": "yolo26n.pt",
    "YOLO26s": "yolo26s.pt",
    "YOLO26m": "yolo26m.pt",
    "YOLO26l": "yolo26l.pt",
    "YOLO11n": "yolo11n.pt",
    "YOLO11s": "yolo11s.pt",
    "YOLO11m": "yolo11m.pt",
}

YOLOE_MODELS = {
    "YOLOE-26s-seg": "yoloe-26s-seg.pt",
    "YOLOE-26m-seg": "yoloe-26m-seg.pt",
    "YOLOE-26l-seg": "yoloe-26l-seg.pt",
    "YOLOE-26l-seg (prompt-free)": "yoloe-26l-seg-pf.pt",
    "YOLOE-11s-seg": "yoloe-11s-seg.pt",
    "YOLOE-11l-seg": "yoloe-11l-seg.pt",
    "YOLOE-11l-seg (prompt-free)": "yoloe-11l-seg-pf.pt",
}

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("⚙️ Configuration")

    framework = st.radio("Framework", ["YOLO", "YOLOE"], horizontal=True)

    if framework == "YOLO":
        model_options = {**YOLO_MODELS, "Custom": "custom"}
        model_label = st.selectbox("Model", list(model_options.keys()))
        model_key = model_options[model_label]
        if model_key == "custom":
            model_key = st.text_input("Path to .pt file", placeholder="/path/to/model.pt")
        prompt_mode = None

    else:  # YOLOE
        model_options = {**YOLOE_MODELS, "Custom": "custom"}
        model_label = st.selectbox("Model", list(model_options.keys()))
        model_key = model_options[model_label]
        if model_key == "custom":
            model_key = st.text_input("Path to .pt file", placeholder="/path/to/yoloe.pt")

        is_pf = "pf" in model_key
        if is_pf:
            prompt_mode = "prompt-free"
            st.info("Prompt-free: detects from built-in vocabulary automatically.")
        else:
            prompt_mode = st.radio(
                "Prompt mode",
                ["Text prompts", "Visual prompts"],
                help="Text: specify class names. Visual: draw boxes on a reference image.",
            )

        # ── Text prompts ──────────────────────────────────────────────────────
        if prompt_mode == "Text prompts":
            classes_raw = st.text_input(
                "Classes to detect (comma-separated)",
                value="person, car",
                help='e.g. "person, dog, traffic light"',
            )
            text_classes = [c.strip() for c in classes_raw.split(",") if c.strip()]

        # ── Visual prompts ────────────────────────────────────────────────────
        elif prompt_mode == "Visual prompts":
            st.caption("Upload a reference image and draw bounding boxes below.")
            ref_img_file = st.file_uploader("Reference image", type=["jpg", "jpeg", "png"])
            vp_boxes_raw = st.text_area(
                "Bounding boxes (one per line: x1,y1,x2,y2,class_id)",
                placeholder="221,405,344,857,0\n120,425,160,445,1",
                height=100,
            )

    # ── Inference params ──────────────────────────────────────────────────────
    st.subheader("Inference Parameters")
    conf      = st.slider("Confidence threshold", 0.01, 1.0, 0.25, 0.01)
    iou       = st.slider("IoU threshold (NMS)",  0.01, 1.0, 0.45, 0.01)
    imgsz     = st.select_slider("Image size", options=[320, 416, 512, 640, 800, 1024], value=640)
    device    = st.selectbox("Device", ["cpu", "0", "1", "2", "3"], index=0)

    st.subheader("Display")
    show_labels   = st.toggle("Show labels",            value=True)
    show_conf_val = st.toggle("Show confidence values", value=True)
    show_boxes    = st.toggle("Show bounding boxes",    value=True)
    line_width    = st.slider("Box line width", 1, 6, 2)

    st.divider()
    st.caption("Powered by [Ultralytics](https://ultralytics.com)")


# ── Model loader (cached) ─────────────────────────────────────────────────────
@st.cache_resource(show_spinner="Loading model…")
def load_yolo(path: str) -> YOLO:
    return YOLO(path)

@st.cache_resource(show_spinner="Loading YOLOE model…")
def load_yoloe(path: str) -> YOLOE:
    return YOLOE(path)


# ── Inference helpers ─────────────────────────────────────────────────────────
def predict_kwargs() -> dict:
    return dict(conf=conf, iou=iou, imgsz=imgsz, device=device, verbose=False)

def run_yolo(model: YOLO, source) -> list:
    return model.predict(source=source, **predict_kwargs())

def run_yoloe_text(model: YOLOE, source) -> list:
    model.set_classes(text_classes)
    return model.predict(source=source, **predict_kwargs())

def run_yoloe_visual(model: YOLOE, source, vp_boxes, vp_cls, ref_img=None) -> list:
    vp = dict(bboxes=np.array(vp_boxes, dtype=float), cls=np.array(vp_cls, dtype=int))
    kw = predict_kwargs()
    if ref_img is not None:
        kw["refer_image"] = ref_img
    return model.predict(source=source, visual_prompts=vp, predictor=YOLOEVPSegPredictor, **kw)

def run_yoloe_pf(model: YOLOE, source) -> list:
    return model.predict(source=source, **predict_kwargs())

def run_inference(model, source) -> list:
    if framework == "YOLO":
        return run_yolo(model, source)
    if prompt_mode == "Text prompts":
        return run_yoloe_text(model, source)
    if prompt_mode == "Visual prompts":
        boxes, cls_ids = parse_visual_prompts()
        ref = load_ref_image()
        return run_yoloe_visual(model, source, boxes, cls_ids, ref)
    return run_yoloe_pf(model, source)  # prompt-free


def parse_visual_prompts():
    boxes, cls_ids = [], []
    for line in vp_boxes_raw.strip().splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) == 5:
            x1, y1, x2, y2, c = parts
            boxes.append([float(x1), float(y1), float(x2), float(y2)])
            cls_ids.append(int(c))
    return boxes, cls_ids

def load_ref_image():
    if ref_img_file:
        img = Image.open(ref_img_file).convert("RGB")
        return np.array(img)
    return None

def result_to_image(result) -> np.ndarray:
    return result.plot(labels=show_labels, conf=show_conf_val, boxes=show_boxes, line_width=line_width)

def extract_labels(result) -> list[dict]:
    """Return list of {class, confidence, x1,y1,x2,y2} from a single result."""
    rows = []
    if result.boxes is None:
        return rows
    for box in result.boxes:
        cls_id = int(box.cls[0])
        rows.append({
            "Class":      result.names[cls_id],
            "Confidence": f"{float(box.conf[0]):.2%}",
            "x1": int(box.xyxy[0][0]),
            "y1": int(box.xyxy[0][1]),
            "x2": int(box.xyxy[0][2]),
            "y2": int(box.xyxy[0][3]),
        })
    return rows

def render_live_labels(result, label_ph):
    """Update a st.empty() placeholder with a real-time label pill summary."""
    if result.boxes is None or len(result.boxes) == 0:
        label_ph.info("No detections")
        return
    counts = Counter(result.names[int(b.cls[0])] for b in result.boxes)
    pills = "  ".join(f"`{cls}` ×{n}" for cls, n in sorted(counts.items()))
    label_ph.markdown(f"**Detected →** {pills}")


# ── Guard: need a valid model path ────────────────────────────────────────────
st.title("🎯 YOLO / YOLOE Inference UI")

if not model_key or model_key == "custom":
    st.warning("Select or enter a model path in the sidebar.")
    st.stop()

model = load_yoloe(model_key) if framework == "YOLOE" else load_yolo(model_key)

# Active config summary
_mode_str = prompt_mode or "standard"
if framework == "YOLOE" and prompt_mode == "Text prompts":
    _mode_str += f" → {', '.join(text_classes)}"
st.caption(f"`{model_key}` · {framework} · {_mode_str} · conf={conf} · iou={iou} · imgsz={imgsz} · device={device}")


# ── Tabs ──────────────────────────────────────────────────────────────────────
tab_image, tab_video, tab_webcam, tab_url = st.tabs(
    ["🖼️ Image", "🎬 Video", "📷 Webcam (live labels)", "🌐 URL / Path"]
)


# ── IMAGE ─────────────────────────────────────────────────────────────────────
with tab_image:
    uploaded = st.file_uploader("Upload an image", type=["jpg","jpeg","png","bmp","webp","tiff"])
    if uploaded:
        pil_img = Image.open(uploaded).convert("RGB")
        col_orig, col_result = st.columns(2)
        with col_orig:
            st.subheader("Original")
            st.image(pil_img, use_container_width=True)

        with st.spinner("Running inference…"):
            results = run_inference(model, np.array(pil_img))

        ann_rgb = cv2.cvtColor(result_to_image(results[0]), cv2.COLOR_BGR2RGB)
        with col_result:
            st.subheader("Annotated")
            st.image(ann_rgb, use_container_width=True)

        rows = extract_labels(results[0])
        if rows:
            st.subheader(f"Detections ({len(rows)})")
            st.dataframe(rows, use_container_width=True)
        else:
            st.info("No detections above threshold.")


# ── VIDEO ─────────────────────────────────────────────────────────────────────
with tab_video:
    vid_file = st.file_uploader("Upload a video", type=["mp4","avi","mov","mkv","webm"])
    if vid_file:
        tfile = tempfile.NamedTemporaryFile(delete=False, suffix=Path(vid_file.name).suffix)
        tfile.write(vid_file.read())
        tfile.flush()
        st.video(vid_file)

        if st.button("▶️ Run inference on video"):
            cap = cv2.VideoCapture(tfile.name)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps  = cap.get(cv2.CAP_PROP_FPS) or 25
            w    = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            h    = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

            out_tmp  = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
            writer   = cv2.VideoWriter(out_tmp.name, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

            frame_ph = st.empty()
            label_ph = st.empty()          # ← live labels
            prog     = st.progress(0, text="Processing frames…")
            t0       = time.time()

            frame_idx = 0
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                rgb     = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = run_inference(model, rgb)
                ann     = result_to_image(results[0])
                writer.write(ann)

                if frame_idx % 5 == 0:
                    frame_ph.image(cv2.cvtColor(ann, cv2.COLOR_BGR2RGB), use_container_width=True)
                    render_live_labels(results[0], label_ph)   # ← update labels live

                frame_idx += 1
                if total_frames > 0:
                    prog.progress(frame_idx / total_frames, text=f"Frame {frame_idx}/{total_frames}")

            cap.release()
            writer.release()
            elapsed = time.time() - t0
            prog.empty(); label_ph.empty(); frame_ph.empty()

            st.success(f"Done! {frame_idx} frames · {elapsed:.1f}s · {frame_idx/elapsed:.1f} FPS")
            with open(out_tmp.name, "rb") as f:
                st.download_button("⬇️ Download annotated video", f, file_name="annotated.mp4")


# ── WEBCAM — live label streaming ─────────────────────────────────────────────
with tab_webcam:
    st.info("Labels update in real-time next to the video feed.")

    col_cam, col_labels = st.columns([3, 1])

    with col_cam:
        run = st.toggle("📷 Start webcam")
        frame_ph = st.empty()

    with col_labels:
        st.subheader("Live labels")
        label_ph  = st.empty()   # pill summary  (updates every frame)
        table_ph  = st.empty()   # detail table  (updates every frame)
        stats_ph  = st.empty()   # latency / fps

    if run:
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            st.error("Could not open webcam.")
        else:
            frame_count = 0
            t_start     = time.perf_counter()

            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                rgb  = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                t0   = time.perf_counter()
                results = run_inference(model, rgb)
                latency = (time.perf_counter() - t0) * 1000
                frame_count += 1

                ann = result_to_image(results[0])
                frame_ph.image(cv2.cvtColor(ann, cv2.COLOR_BGR2RGB), use_container_width=True)

                # ── Real-time labels ─────────────────────────────────────────
                render_live_labels(results[0], label_ph)

                rows = extract_labels(results[0])
                if rows:
                    table_ph.dataframe(rows, use_container_width=True, height=300)
                else:
                    table_ph.empty()

                elapsed = time.perf_counter() - t_start
                fps_live = frame_count / elapsed if elapsed > 0 else 0
                stats_ph.caption(f"Latency: {latency:.0f} ms · {fps_live:.1f} FPS · {len(rows)} obj")
                # ─────────────────────────────────────────────────────────────

                # Check if toggle was switched off
                if not st.session_state.get("📷 Start webcam", False):
                    break

            cap.release()
            frame_ph.empty(); label_ph.empty(); table_ph.empty(); stats_ph.empty()
            st.info("Webcam stopped.")


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
            rows = extract_labels(r)
            if rows:
                st.dataframe(rows, use_container_width=True)
