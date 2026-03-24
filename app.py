import streamlit as st
import cv2
import numpy as np
import tempfile
import time
import io
import csv
import json
from collections import Counter
from pathlib import Path
from PIL import Image
from ultralytics import YOLO, YOLOE
from ultralytics.models.yolo.yoloe import YOLOEVPSegPredictor
import av
import pandas as pd
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, RTCConfiguration
from streamlit_js_eval import streamlit_js_eval

st.set_page_config(page_title="IINIA - Detector YOLO", page_icon=None, layout="wide")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

/* Global font */
html, body, [class*="css"] {
    font-family: 'Inter', system-ui, sans-serif !important;
}

/* Main background */
[data-testid="stAppViewContainer"] { background-color: #0d1117; }
[data-testid="stMain"] { background-color: #0d1117; }

/* Sidebar */
[data-testid="stSidebar"] {
    background-color: #0a1520 !important;
    border-right: 1px solid #1e3a3a;
}
[data-testid="stSidebar"] * { color: #cce8e8 !important; }
[data-testid="stSidebar"] h1, [data-testid="stSidebar"] h2, [data-testid="stSidebar"] h3 {
    color: #7acfcf !important;
    font-weight: 600 !important;
}

/* Metrics */
[data-testid="metric-container"] {
    background: #0f2030;
    border-radius: 10px;
    padding: 14px;
    border: 1px solid #1e4040;
}
[data-testid="metric-container"] label { color: #7acfcf !important; font-size: 0.8rem; }
[data-testid="metric-container"] [data-testid="stMetricValue"] { color: #e0f4f4 !important; font-weight: 600; }

/* Tabs */
.stTabs [data-baseweb="tab-list"] {
    background: #0f2030;
    border-radius: 10px;
    padding: 4px;
    gap: 4px;
    border: 1px solid #1e4040;
}
.stTabs [data-baseweb="tab"] {
    color: #6aafaf;
    border-radius: 8px;
    font-weight: 500;
    font-size: 0.9rem;
    padding: 8px 20px;
}
.stTabs [aria-selected="true"] {
    background: linear-gradient(135deg, #1a4a4a, #0f3535) !important;
    color: #7acfcf !important;
    border: 1px solid #2a6060 !important;
}

/* Buttons */
.stButton > button {
    background: linear-gradient(135deg, #1a5555, #0f3a3a);
    color: #a0e0e0;
    border: 1px solid #2a7070;
    border-radius: 8px;
    font-family: 'Inter', sans-serif;
    font-weight: 500;
    transition: all 0.2s;
}
.stButton > button:hover {
    background: linear-gradient(135deg, #226060, #1a4545);
    border-color: #3a8080;
    color: #c0f0f0;
}

/* Download button */
.stDownloadButton > button {
    background: #0f2a2a;
    color: #7acfcf;
    border: 1px solid #1e5050;
    border-radius: 8px;
    font-family: 'Inter', sans-serif;
}

/* Inputs / sliders */
[data-testid="stSlider"] [data-baseweb="slider"] [role="slider"] { background: #7acfcf !important; }
[data-baseweb="select"] { background-color: #0f2030 !important; border-color: #1e4040 !important; }
[data-baseweb="input"] { background-color: #0f2030 !important; }

/* Text and headings */
h1, h2, h3 { color: #e0f4f4 !important; font-weight: 600 !important; }
p, span, label { color: #b0d8d8; }

/* DataFrames */
div[data-testid="stDataFrame"] { border-radius: 10px; overflow: hidden; border: 1px solid #1e4040; }

/* Expander */
[data-testid="stExpander"] {
    background: #0f2030;
    border: 1px solid #1e4040;
    border-radius: 10px;
}

/* Info / warning boxes */
[data-testid="stAlert"] { border-radius: 8px; }

/* Progress bar */
[data-testid="stProgressBar"] > div > div { background-color: #7acfcf !important; }

/* Divider */
hr { border-color: #1e4040 !important; }
</style>
""", unsafe_allow_html=True)

YOLO_MODELS = {
    "YOLO26n - mas rapido (nano)": "yolo26n.pt",
    "YOLO26s - rapido (small)":    "yolo26s.pt",
    "YOLO26m - equilibrado":       "yolo26m.pt",
    "YOLO26l - preciso (large)":   "yolo26l.pt",
    "YOLO11n - nano":              "yolo11n.pt",
    "YOLO11s - small":             "yolo11s.pt",
    "YOLO11m - medium":            "yolo11m.pt",
}

YOLOE_MODELS = {
    "YOLOE-26s-seg":                "yoloe-26s-seg.pt",
    "YOLOE-26m-seg":                "yoloe-26m-seg.pt",
    "YOLOE-26l-seg":                "yoloe-26l-seg.pt",
    "YOLOE-26l-seg - sin etiquetas":"yoloe-26l-seg-pf.pt",
    "YOLOE-11s-seg":                "yoloe-11s-seg.pt",
    "YOLOE-11l-seg":                "yoloe-11l-seg.pt",
    "YOLOE-11l-seg - sin etiquetas":"yoloe-11l-seg-pf.pt",
}

with st.sidebar:
    st.image("logo.webp", width=120)
    st.markdown("**IINIA** · Detector de objetos")
    st.divider()
    st.title("Configuracion")

    framework = st.radio(
        "Modo de deteccion",
        ["YOLO", "YOLOE"],
        horizontal=True,
        help="YOLO: detecta objetos con cajas. YOLOE: tambien segmenta (silueta).",
    )

    text_classes = []
    ref_img_file = None
    vp_boxes_raw = ""
    prompt_mode  = None

    if framework == "YOLO":
        model_options = {**YOLO_MODELS, "Personalizado": "custom"}
        model_label = st.selectbox("Modelo", list(model_options.keys()))
        model_key = model_options[model_label]
        if model_key == "custom":
            model_key = st.text_input("Ruta al archivo .pt", placeholder="/ruta/al/modelo.pt")

    else:
        model_options = {**YOLOE_MODELS, "Personalizado": "custom"}
        model_label = st.selectbox("Modelo", list(model_options.keys()))
        model_key = model_options[model_label]
        if model_key == "custom":
            model_key = st.text_input("Ruta al archivo .pt", placeholder="/ruta/al/modelo.pt")

        is_pf = "pf" in model_key
        if is_pf:
            prompt_mode = "prompt-free"
            st.info("Modo automatico: detecta todo sin necesidad de configurar nada.")
        else:
            prompt_mode = st.radio(
                "Como indicas que detectar",
                ["Texto", "Imagen de referencia"],
                help="Texto: escribe los nombres. Imagen: marca con cajas lo que quieres detectar.",
            )

        if prompt_mode == "Texto":
            classes_raw = st.text_input(
                "Objetos a detectar (separados por coma)",
                value="person, car",
                help='Ejemplo: "persona, perro, semaforo"',
            )
            text_classes = [c.strip() for c in classes_raw.split(",") if c.strip()]

        elif prompt_mode == "Imagen de referencia":
            st.caption("Sube una imagen de ejemplo y marca con cajas los objetos.")
            ref_img_file = st.file_uploader("Imagen de referencia", type=["jpg", "jpeg", "png"])
            vp_boxes_raw = st.text_area(
                "Cajas (una por linea: x1,y1,x2,y2,id_clase)",
                placeholder="221,405,344,857,0\n120,425,160,445,1",
                height=100,
            )

    st.divider()
    st.subheader("Ajustes de deteccion")
    conf  = st.slider("Confianza minima", 0.01, 1.0, 0.25, 0.01,
                      help="Solo muestra detecciones con este porcentaje de certeza o mas.")
    iou   = st.slider("Solapamiento maximo (IoU)", 0.01, 1.0, 0.45, 0.01,
                      help="Elimina cajas duplicadas si se solapan mas de este valor.")
    imgsz = st.select_slider("Tamano de analisis", options=[320, 416, 512, 640, 800, 1024], value=640,
                              help="Mayor tamano = mas preciso pero mas lento.")
    device_raw = st.selectbox("Procesador", ["0 (GPU)", "cpu", "1", "2", "3"], index=0)
    device = device_raw.split(" ")[0]

    st.divider()
    st.subheader("Visualizacion")
    show_labels   = st.toggle("Mostrar nombre del objeto",        value=True)
    show_conf_val = st.toggle("Mostrar porcentaje de confianza",  value=True)
    show_boxes    = st.toggle("Mostrar cajas",                    value=True)
    line_width    = st.slider("Grosor de las cajas", 1, 6, 2)

    st.divider()
    st.caption("Motor: [Ultralytics](https://ultralytics.com)")


@st.cache_resource(show_spinner="Cargando modelo...")
def load_yolo(path: str) -> YOLO:
    m = YOLO(path)
    _warmup(m)
    return m

@st.cache_resource(show_spinner="Cargando modelo YOLOE...")
def load_yoloe(path: str) -> YOLOE:
    m = YOLOE(path)
    _warmup(m)
    return m

def _warmup(m):
    dummy = np.zeros((64, 64, 3), dtype=np.uint8)
    m.predict(source=dummy, imgsz=64, verbose=False)


def predict_kwargs() -> dict:
    return dict(conf=conf, iou=iou, imgsz=imgsz, device=device, verbose=False)

def run_yolo(model, source):
    return model.predict(source=source, **predict_kwargs())

def run_yoloe_text(model, source):
    if isinstance(model.model.names, list):
        model.model.names = {i: n for i, n in enumerate(model.model.names)}
    model.set_classes(text_classes)
    return model.predict(source=source, **predict_kwargs())

def run_yoloe_visual(model, source, vp_boxes, vp_cls, ref_img=None):
    vp = dict(bboxes=np.array(vp_boxes, dtype=float), cls=np.array(vp_cls, dtype=int))
    kw = predict_kwargs()
    if ref_img is not None:
        kw["refer_image"] = ref_img
    return model.predict(source=source, visual_prompts=vp, predictor=YOLOEVPSegPredictor, **kw)

def run_inference(model, source):
    if framework == "YOLO":
        return run_yolo(model, source)
    if prompt_mode == "Texto":
        return run_yoloe_text(model, source)
    if prompt_mode == "Imagen de referencia":
        boxes, cls_ids = parse_visual_prompts()
        if not boxes:
            st.warning("Anade al menos una caja en el panel lateral antes de analizar.")
            st.stop()
        return run_yoloe_visual(model, source, boxes, cls_ids, load_ref_image())
    return model.predict(source=source, **predict_kwargs())

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
        return np.array(Image.open(ref_img_file).convert("RGB"))
    return None

def result_to_image(result) -> np.ndarray:
    return result.plot(labels=show_labels, conf=show_conf_val, boxes=show_boxes, line_width=line_width)

def extract_labels(result) -> list[dict]:
    rows = []
    if result.boxes is None:
        return rows
    for box in result.boxes:
        cls_id = int(box.cls[0])
        rows.append({
            "Objeto":    result.names[cls_id],
            "Confianza": round(float(box.conf[0]), 4),
            "x1": int(box.xyxy[0][0]), "y1": int(box.xyxy[0][1]),
            "x2": int(box.xyxy[0][2]), "y2": int(box.xyxy[0][3]),
        })
    return rows

def apply_class_thresholds(result, class_thresholds: dict):
    if not class_thresholds or result.boxes is None or len(result.boxes) == 0:
        return result
    import torch
    mask = torch.tensor([
        float(b.conf[0]) >= class_thresholds.get(result.names[int(b.cls[0])], 0.0)
        for b in result.boxes
    ])
    result.boxes = result.boxes[mask]
    return result

def render_live_labels(result, label_ph):
    if result.boxes is None or len(result.boxes) == 0:
        label_ph.info("Sin detecciones")
        return
    counts = Counter(result.names[int(b.cls[0])] for b in result.boxes)
    pills = "  ".join(f"`{cls}` x{n}" for cls, n in sorted(counts.items()))
    label_ph.markdown(f"**Detectado:** {pills}")

def rows_to_csv(rows: list[dict]) -> bytes:
    if not rows:
        return b""
    buf = io.StringIO()
    w = csv.DictWriter(buf, fieldnames=rows[0].keys())
    w.writeheader()
    w.writerows(rows)
    return buf.getvalue().encode()

def rows_to_json(rows: list[dict]) -> bytes:
    return json.dumps(rows, ensure_ascii=False, indent=2).encode()

def show_download_buttons(rows: list[dict], key_prefix: str):
    if not rows:
        return
    c1, c2 = st.columns(2)
    with c1:
        st.download_button("Descargar CSV", rows_to_csv(rows),
                           file_name="detecciones.csv", mime="text/csv", key=f"{key_prefix}_csv")
    with c2:
        st.download_button("Descargar JSON", rows_to_json(rows),
                           file_name="detecciones.json", mime="application/json", key=f"{key_prefix}_json")

def show_distribution_chart(rows: list[dict]):
    if not rows:
        return
    counts = Counter(r["Objeto"] for r in rows)
    df = pd.DataFrame({"Objeto": list(counts.keys()), "Cantidad": list(counts.values())})
    st.bar_chart(df.sort_values("Cantidad", ascending=False).set_index("Objeto"))


# Guardia: modelo valido
col_logo, col_title = st.columns([1, 8])
with col_logo:
    st.image("logo.webp", width=72)
with col_title:
    st.markdown("## Detector de objetos YOLO")
    st.markdown('<p style="color:#6aafaf;margin-top:-12px;font-size:0.85rem;">Industrial Intelligence & AI Solutions</p>', unsafe_allow_html=True)

if not model_key or model_key == "custom":
    st.warning("Selecciona o introduce la ruta a un modelo en el panel lateral.")
    st.stop()

model = load_yoloe(model_key) if framework == "YOLOE" else load_yolo(model_key)

with st.expander("Info del modelo cargado"):
    names_dict = model.names if isinstance(model.names, dict) else {i: n for i, n in enumerate(model.names)}
    n_classes = len(names_dict)
    sample = ", ".join(list(names_dict.values())[:12])
    if n_classes > 12:
        sample += f" (+{n_classes-12} mas)"
    pt_path = Path(model_key)
    size_str = f"{pt_path.stat().st_size / 1e6:.1f} MB" if pt_path.exists() else "descargado automaticamente"
    col1, col2, col3 = st.columns(3)
    col1.metric("Clases", n_classes)
    col2.metric("Tamano", size_str)
    col3.metric("Procesador", device)
    st.caption(f"Clases: {sample}")

_modo = prompt_mode or "estandar"
if framework == "YOLOE" and prompt_mode == "Texto":
    _modo += f" - {', '.join(text_classes)}"
st.caption(f"`{model_key}` - {framework} - {_modo} - confianza={conf} - iou={iou} - tamano={imgsz} - procesador={device}")


tab_image, tab_video, tab_webcam, tab_url = st.tabs(
    ["Imagen", "Video", "Camara en vivo", "URL / Ruta"]
)


# IMAGEN
with tab_image:
    uploaded_files = st.file_uploader(
        "Sube una o varias imagenes",
        type=["jpg", "jpeg", "png", "bmp", "webp", "tiff"],
        accept_multiple_files=True,
    )

    if uploaded_files:
        all_rows: list[dict] = []

        for idx, uploaded in enumerate(uploaded_files):
            pil_img = Image.open(uploaded).convert("RGB")
            bgr_img = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)

            with st.spinner(f"Analizando {uploaded.name}..."):
                results = run_inference(model, bgr_img)

            result = results[0]
            rows_raw = extract_labels(result)
            detected_classes = sorted({r["Objeto"] for r in rows_raw})
            class_thresholds = {}

            if detected_classes:
                with st.expander(f"Umbral por clase - {uploaded.name}", expanded=False):
                    cols = st.columns(min(len(detected_classes), 4))
                    for i, cls_name in enumerate(detected_classes):
                        class_thresholds[cls_name] = cols[i % 4].slider(
                            cls_name, 0.01, 1.0, conf, 0.01,
                            key=f"thr_{idx}_{cls_name}"
                        )
                result = apply_class_thresholds(result, class_thresholds)

            ann_rgb = cv2.cvtColor(result_to_image(result), cv2.COLOR_BGR2RGB)
            rows = extract_labels(result)
            all_rows.extend(rows)

            col_orig, col_result = st.columns(2)
            with col_orig:
                st.subheader(f"Original - {uploaded.name}")
                st.image(pil_img, width="stretch")
            with col_result:
                st.subheader(f"Resultado - {uploaded.name}")
                st.image(ann_rgb, width="stretch")

            if rows:
                st.subheader(f"Objetos detectados ({len(rows)})")
                st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)
                show_distribution_chart(rows)
                show_download_buttons(rows, key_prefix=f"img_{idx}")
            else:
                st.info("No se detecto ningun objeto con la confianza configurada.")

            if len(uploaded_files) > 1:
                st.divider()

        if len(uploaded_files) > 1 and all_rows:
            st.subheader(f"Resumen total ({len(uploaded_files)} imagenes, {len(all_rows)} detecciones)")
            show_distribution_chart(all_rows)
            show_download_buttons(all_rows, key_prefix="img_all")


# VIDEO
with tab_video:
    vid_file = st.file_uploader("Sube un video", type=["mp4", "avi", "mov", "mkv", "webm"])
    if vid_file:
        tfile = tempfile.NamedTemporaryFile(delete=False, suffix=Path(vid_file.name).suffix)
        tfile.write(vid_file.read())
        tfile.flush()
        st.video(vid_file)

        skip_n = st.slider(
            "Analizar 1 de cada N fotogramas",
            min_value=1, max_value=10, value=1,
            help="Para videos largos, aumenta este valor para terminar mas rapido."
        )

        if st.button("Analizar video"):
            cap = cv2.VideoCapture(tfile.name)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps  = cap.get(cv2.CAP_PROP_FPS) or 25
            w    = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            h    = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

            out_tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
            writer  = cv2.VideoWriter(out_tmp.name, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

            frame_ph = st.empty()
            label_ph = st.empty()
            prog     = st.progress(0, text="Procesando fotogramas...")
            t0       = time.time()
            frame_idx = 0
            last_ann  = None
            video_rows: list[dict] = []

            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                if frame_idx % skip_n == 0:
                    results  = run_inference(model, frame)
                    last_ann = result_to_image(results[0])
                    video_rows.extend(extract_labels(results[0]))
                    if frame_idx % (skip_n * 5) == 0:
                        frame_ph.image(cv2.cvtColor(last_ann, cv2.COLOR_BGR2RGB), width="stretch")
                        render_live_labels(results[0], label_ph)
                writer.write(last_ann if last_ann is not None else frame)
                frame_idx += 1
                if total_frames > 0:
                    prog.progress(frame_idx / total_frames, text=f"Fotograma {frame_idx}/{total_frames}")

            cap.release()
            writer.release()
            elapsed = time.time() - t0
            prog.empty(); label_ph.empty(); frame_ph.empty()

            analyzed = frame_idx // skip_n
            st.success(f"Listo. {frame_idx} fotogramas, {analyzed} analizados, {elapsed:.1f}s, {analyzed/elapsed:.1f} inf/s")

            with open(out_tmp.name, "rb") as f:
                st.download_button("Descargar video anotado", f, file_name="resultado.mp4")

            if video_rows:
                st.subheader("Distribucion de detecciones")
                show_distribution_chart(video_rows)
                show_download_buttons(video_rows, key_prefix="video")


# CAMARA EN VIVO
with tab_webcam:
    st.info("Pulsa START para activar la camara. La deteccion corre en tiempo real fotograma a fotograma.")

    # ── Enumeración de cámaras via JS ────────────────────────────────────────
    raw_devices = streamlit_js_eval(
        js_expressions=(
            "navigator.mediaDevices.enumerateDevices()"
            ".then(ds => JSON.stringify("
            "  ds.filter(d => d.kind === 'videoinput')"
            "    .map((d, i) => ({id: d.deviceId, label: d.label || ('Camara ' + (i + 1))}))"
            "))"
        ),
        key="enum_cameras",
    )

    selected_device_id = None
    if raw_devices:
        try:
            cam_devices = json.loads(raw_devices)
            if cam_devices:
                cam_labels = [d["label"] for d in cam_devices]
                cam_sel = st.selectbox(
                    "Seleccionar camara",
                    cam_labels,
                    key="cam_selector",
                )
                cam_idx = cam_labels.index(cam_sel)
                selected_device_id = cam_devices[cam_idx]["id"] or None
        except (json.JSONDecodeError, KeyError, ValueError):
            pass

    _framework     = framework
    _model         = model
    _prompt_mode   = prompt_mode
    _conf          = conf
    _iou           = iou
    _imgsz         = imgsz
    _device        = device
    _show_labels   = show_labels
    _show_conf_val = show_conf_val
    _show_boxes    = show_boxes
    _line_width    = line_width
    _text_classes  = text_classes if (framework == "YOLOE" and prompt_mode == "Texto") else []

    class YOLOVideoProcessor(VideoProcessorBase):
        def __init__(self):
            self._frame_times: list[float] = []
            self.detection_counts = Counter()
            self.total_frames = 0

        def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
            t0  = time.perf_counter()
            img = frame.to_ndarray(format="bgr24")
            kw  = dict(conf=_conf, iou=_iou, imgsz=_imgsz, device=_device, verbose=False)

            if _framework == "YOLO":
                results = _model.predict(source=img, **kw)
            elif _prompt_mode == "Texto":
                if isinstance(_model.model.names, list):
                    _model.model.names = {i: n for i, n in enumerate(_model.model.names)}
                _model.set_classes(_text_classes)
                results = _model.predict(source=img, **kw)
            elif _prompt_mode == "Imagen de referencia":
                boxes, cls_ids = parse_visual_prompts()
                if not boxes:
                    return frame
                vp  = dict(bboxes=np.array(boxes, dtype=float), cls=np.array(cls_ids, dtype=int))
                ref = load_ref_image()
                if ref is not None:
                    kw["refer_image"] = ref
                results = _model.predict(source=img, visual_prompts=vp, predictor=YOLOEVPSegPredictor, **kw)
            else:
                results = _model.predict(source=img, **kw)

            t1 = time.perf_counter()
            self._frame_times.append(t1 - t0)
            if len(self._frame_times) > 30:
                self._frame_times.pop(0)
            fps_val = len(self._frame_times) / sum(self._frame_times) if self._frame_times else 0

            if results[0].boxes is not None:
                for b in results[0].boxes:
                    self.detection_counts[results[0].names[int(b.cls[0])]] += 1
            self.total_frames += 1

            ann = results[0].plot(
                labels=_show_labels, conf=_show_conf_val,
                boxes=_show_boxes,   line_width=_line_width,
            )
            cv2.putText(ann, f"FPS: {fps_val:.1f}", (10, 36),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0, 230, 0), 2, cv2.LINE_AA)

            return av.VideoFrame.from_ndarray(cv2.cvtColor(ann, cv2.COLOR_BGR2RGB), format="rgb24")

    RTC_CONFIG = RTCConfiguration({"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]})

    video_constraints: dict = {"width": {"ideal": 1280}, "height": {"ideal": 720}, "frameRate": {"ideal": 30}}
    if selected_device_id:
        video_constraints["deviceId"] = {"exact": selected_device_id}

    ctx = webrtc_streamer(
        key=f"yolo-webcam-{selected_device_id or 'default'}",
        video_processor_factory=YOLOVideoProcessor,
        rtc_configuration=RTC_CONFIG,
        media_stream_constraints={"video": video_constraints, "audio": False},
        async_processing=True,
    )

    if ctx and ctx.video_processor:
        st.button("Refrescar estadisticas")
        proc = ctx.video_processor
        if proc.total_frames > 0:
            st.subheader("Historial de detecciones")
            m1, m2 = st.columns(2)
            m1.metric("Fotogramas procesados", proc.total_frames)
            m2.metric("Detecciones totales", sum(proc.detection_counts.values()))
            if proc.detection_counts:
                df_hist = pd.DataFrame(
                    {"Objeto": list(proc.detection_counts.keys()),
                     "Veces detectado": list(proc.detection_counts.values())}
                ).sort_values("Veces detectado", ascending=False)
                st.dataframe(df_hist, width="stretch", hide_index=True)
                st.bar_chart(df_hist.set_index("Objeto"))
            if st.button("Limpiar historial"):
                proc.detection_counts = Counter()
                proc.total_frames = 0
                st.rerun()


# URL / RUTA
with tab_url:
    source_str = st.text_input(
        "Introduce una URL o ruta local",
        placeholder="https://... o /home/usuario/imagen.jpg",
    )
    if source_str and st.button("Analizar"):
        with st.spinner("Analizando..."):
            results = run_inference(model, source_str)
        url_rows: list[dict] = []
        for i, r in enumerate(results):
            ann = result_to_image(r)
            st.image(cv2.cvtColor(ann, cv2.COLOR_BGR2RGB), caption=f"Fotograma {i+1}", width="stretch")
            rows = extract_labels(r)
            url_rows.extend(rows)
            if rows:
                st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)
        if url_rows:
            st.subheader("Distribucion")
            show_distribution_chart(url_rows)
            show_download_buttons(url_rows, key_prefix="url")
