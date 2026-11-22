import os
import time
import uuid
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed
import numpy as np
import requests
from requests.adapters import HTTPAdapter
from PIL import Image, ImageOps
import streamlit as st
from insightface.app import FaceAnalysis

st.set_page_config(
    page_title="Tulsi Vivah & Shree Shyam Akhand Jyot Paath - Photo Finder",
    page_icon="📸",
    layout="wide",
)

INDEX_FILE = "event_face_index.npz"
THUMB_CACHE = "/tmp/thumb_cache"
ORIG_CACHE = "/tmp/orig_cache"
MATCH_THRESHOLD = 0.42  # Lower to 0.38 if missing photos; raise to 0.48 if false matches appear

os.makedirs(THUMB_CACHE, exist_ok=True)
os.makedirs(ORIG_CACHE, exist_ok=True)


@st.cache_resource
def load_session():
    s = requests.Session()
    adapter = HTTPAdapter(pool_connections=24, pool_maxsize=24)
    s.mount("https://", adapter)
    s.mount("http://", adapter)
    return s


@st.cache_resource
def load_index_and_model():
    data = np.load(INDEX_FILE, allow_pickle=False)
    embeddings = data["embeddings"].astype(np.float32)
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    embeddings = embeddings / np.maximum(norms, 1e-12)

    photo_ids = data["photo_ids"].tolist()
    photo_paths = data["photo_paths"].tolist()
    face_photo_idx = data["face_photo_idx"]

    # Load only detection + recognition to stay fast and light on free Cloud RAM
    model = FaceAnalysis(
        name="buffalo_l",
        allowed_modules=["detection", "recognition"],
        providers=["CPUExecutionProvider"],
    )
    model.prepare(ctx_id=-1, det_size=(640, 640))
    return embeddings, photo_ids, photo_paths, face_photo_idx, model


session = load_session()
db_embeddings, db_photo_ids, db_photo_paths, db_face_photo_idx, app = load_index_and_model()


def download_preview(item):
    fid = item["id"]
    local_path = os.path.join(THUMB_CACHE, f"{fid}.jpg")
    if os.path.exists(local_path) and os.path.getsize(local_path) > 0:
        return local_path

    try:
        r = session.get(
            f"https://drive.google.com/thumbnail?id={fid}&sz=w1000",
            timeout=15,
            allow_redirects=True,
        )
        if r.status_code == 200 and r.headers.get("content-type", "").startswith("image/") and r.content:
            with open(local_path, "wb") as f:
                f.write(r.content)
            return local_path
    except Exception:
        pass

    orig_path, _ = download_original(item)
    return orig_path


def download_original(item):
    fid = item["id"]
    ext = os.path.splitext(item["rel_path"])[1] or ".jpg"
    local_path = os.path.join(ORIG_CACHE, f"{fid}{ext}")
    if os.path.exists(local_path) and os.path.getsize(local_path) > 0:
        return local_path, item["rel_path"]

    urls = [
        f"https://drive.usercontent.google.com/download?id={fid}&export=download&confirm=t",
        f"https://drive.google.com/uc?export=download&id={fid}",
    ]
    for url in urls:
        for attempt in range(2):
            try:
                r = session.get(url, timeout=60, allow_redirects=True)
                ctype = r.headers.get("content-type", "").lower()
                if r.status_code == 200 and r.content and not ctype.startswith("text/html"):
                    with open(local_path, "wb") as f:
                        f.write(r.content)
                    return local_path, item["rel_path"]
                if r.status_code in (429, 500, 502, 503, 504):
                    time.sleep(1.5 * (attempt + 1))
                    continue
                break
            except Exception:
                time.sleep(1)
    return None, item["rel_path"]


# --- UI ---
st.title("Find your photos of Tulsi Vivah & Shree Shyam Akhand Jyot Paath")
st.write("Upload a clear selfie or take one with your camera to pull up all your pictures from the event.")

if "matched_items" not in st.session_state:
    st.session_state.matched_items = []
if "preview_pairs" not in st.session_state:
    st.session_state.preview_pairs = []
if "zip_path" not in st.session_state:
    st.session_state.zip_path = None
if "search_msg" not in st.session_state:
    st.session_state.search_msg = ""

tab_upload, tab_camera = st.tabs(["Upload a Selfie", "Take a Selfie"])
with tab_upload:
    uploaded_file = st.file_uploader("Choose a clear photo of yourself", type=["jpg", "jpeg", "png", "webp"])
with tab_camera:
    camera_file = st.camera_input("Snap a clear selfie")

selfie_source = uploaded_file or camera_file

if st.button("Find My Photos", type="primary", width="stretch"):
    if selfie_source is None:
        st.warning("Please upload or take a clear selfie first.")
    else:
        t0 = time.time()
        with st.spinner("Scanning your selfie and fetching matching photos..."):
            img = ImageOps.exif_transpose(Image.open(selfie_source)).convert("RGB")
            img.thumbnail((1000, 1000))
            img_bgr = np.array(img)[:, :, ::-1]

            faces = app.get(img_bgr)
            if not faces:
                st.session_state.matched_items = []
                st.session_state.preview_pairs = []
                st.session_state.zip_path = None
                st.error("No face detected. Try better lighting and face the camera directly.")
            else:
                target_face = max(faces, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]))
                scores = db_embeddings @ target_face.normed_embedding
                matched_indices = np.where(scores >= MATCH_THRESHOLD)[0]

                if len(matched_indices) == 0:
                    st.session_state.matched_items = []
                    st.session_state.preview_pairs = []
                    st.session_state.zip_path = None
                    st.warning("No matching photos found. Try a clearer selfie without sunglasses or blur.")
                else:
                    sorted_indices = matched_indices[np.argsort(scores[matched_indices])[::-1]]
                    seen = set()
                    matched_items = []
                    for idx in sorted_indices:
                        p_idx = int(db_face_photo_idx[idx])
                        if p_idx not in seen:
                            seen.add(p_idx)
                            matched_items.append({
                                "id": db_photo_ids[p_idx],
                                "rel_path": db_photo_paths[p_idx],
                            })

                    with ThreadPoolExecutor(max_workers=16) as pool:
                        previews = list(pool.map(download_preview, matched_items))

                    pairs = [(p, item) for p, item in zip(previews, matched_items) if p is not None]
                    st.session_state.matched_items = matched_items
                    st.session_state.preview_pairs = pairs
                    st.session_state.zip_path = None

                    elapsed = time.time() - t0
                    if pairs:
                        st.session_state.search_msg = f"Found {len(pairs)} photos in {elapsed:.1f}s!"
                    else:
                        st.error("Photos matched, but Google Drive blocked public access. Make sure the photographer's folder is set to 'Anyone with the link'.")

if st.session_state.preview_pairs:
    st.success(st.session_state.search_msg)

    col_zip1, col_zip2 = st.columns([1, 2])
    with col_zip1:
        if st.button("Prepare Full-Quality ZIP Download", width="stretch"):
            items = st.session_state.matched_items
            total = len(items)
            bar = st.progress(0, text=f"Downloading 0/{total} full-resolution photos...")
            downloaded = []
            with ThreadPoolExecutor(max_workers=10) as pool:
                futures = [pool.submit(download_original, item) for item in items]
                for done, fut in enumerate(as_completed(futures), 1):
                    local_path, rel_path = fut.result()
                    if local_path:
                        downloaded.append((local_path, rel_path))
                    bar.progress(done / total, text=f"Downloading {done}/{total} full-resolution photos...")

            if downloaded:
                zip_path = f"/tmp/tulsi_vivah_photos_{uuid.uuid4().hex[:8]}.zip"
                used_names = set()
                with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_STORED) as zf:
                    for i, (local_path, rel_path) in enumerate(downloaded, 1):
                        arcname = rel_path
                        if arcname in used_names:
                            arcname = f"{i:03d}_{os.path.basename(rel_path)}"
                        used_names.add(arcname)
                        zf.write(local_path, arcname=arcname)
                st.session_state.zip_path = zip_path
                bar.empty()
            else:
                bar.empty()
                st.error("Could not download original files from Google Drive.")

    with col_zip2:
        if st.session_state.zip_path and os.path.exists(st.session_state.zip_path):
            with open(st.session_state.zip_path, "rb") as zf:
                st.download_button(
                    label=f"Download ZIP ({len(st.session_state.matched_items)} Full-Res Photos)",
                    data=zf,
                    file_name="tulsi_vivah_my_photos.zip",
                    mime="application/zip",
                    type="primary",
                )

    st.divider()
    cols = st.columns(4)
    for idx, (preview_path, item) in enumerate(st.session_state.preview_pairs):
        with cols[idx % 4]:
            st.image(preview_path, width="stretch")
            drive_url = f"https://drive.google.com/file/d/{item['id']}/view"
            st.markdown(f"[Open Full Quality]({drive_url})")

st.markdown(
    "---\n"
    "<p style='text-align: center; opacity: 0.75; font-size: 14px;'>"
    "Created by <strong>Gurudutt Mundra</strong>"
    "</p>",
    unsafe_allow_html=True,
)
