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
    page_title="Tulsi Vivah - Photo & Video Finder",
    page_icon="📸",
    layout="wide",
)

PHOTO_INDEX_FILE = "event_face_index.npz"
VIDEO_INDEX_FILE = "video_face_index.npz"
THUMB_CACHE = "/tmp/thumb_cache"
ORIG_CACHE = "/tmp/orig_cache"

PHOTO_MATCH_THRESHOLD = 0.42  # Lower to 0.38 if missing photos; raise to 0.48 if false matches appear
VIDEO_MATCH_THRESHOLD = 0.40  # Slightly more lenient for moving faces in video

os.makedirs(THUMB_CACHE, exist_ok=True)
os.makedirs(ORIG_CACHE, exist_ok=True)


@st.cache_resource
def load_session():
    s = requests.Session()
    adapter = HTTPAdapter(pool_connections=24, pool_maxsize=24)
    s.mount("https://", adapter)
    s.mount("http://", adapter)
    return s


def _normalize_embeddings(arr):
    embs = arr.astype(np.float32)
    if len(embs) == 0:
        return np.empty((0, 512), dtype=np.float32)
    norms = np.linalg.norm(embs, axis=1, keepdims=True)
    return embs / np.maximum(norms, 1e-12)


@st.cache_resource
def load_indexes_and_model():
    p_data = np.load(PHOTO_INDEX_FILE, allow_pickle=False)
    photo_index = {
        "embeddings": _normalize_embeddings(p_data["embeddings"]),
        "ids": p_data["photo_ids"].tolist(),
        "paths": p_data["photo_paths"].tolist(),
        "face_idx": p_data["face_photo_idx"],
    }

    video_index = None
    if os.path.exists(VIDEO_INDEX_FILE):
        v_data = np.load(VIDEO_INDEX_FILE, allow_pickle=False)
        video_index = {
            "embeddings": _normalize_embeddings(v_data["embeddings"]),
            "ids": v_data["video_ids"].tolist(),
            "paths": v_data["video_paths"].tolist(),
            "face_idx": v_data["face_video_idx"],
        }

    model = FaceAnalysis(
        name="buffalo_l",
        allowed_modules=["detection", "recognition"],
        providers=["CPUExecutionProvider"],
    )
    model.prepare(ctx_id=-1, det_size=(640, 640))
    return photo_index, video_index, model


session = load_session()
photo_index, video_index, app = load_indexes_and_model()


def download_photo_preview(item):
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


def download_video_thumb(item):
    fid = item["id"]
    local_path = os.path.join(THUMB_CACHE, f"vid_{fid}.jpg")
    if os.path.exists(local_path) and os.path.getsize(local_path) > 0:
        return local_path

    try:
        r = session.get(
            f"https://drive.google.com/thumbnail?id={fid}&sz=w800",
            timeout=12,
            allow_redirects=True,
        )
        if r.status_code == 200 and r.headers.get("content-type", "").startswith("image/") and r.content:
            with open(local_path, "wb") as f:
                f.write(r.content)
            return local_path
    except Exception:
        pass
    return None


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


def search_index(index_dict, query_emb, threshold):
    if index_dict is None or len(index_dict["embeddings"]) == 0:
        return []
    scores = index_dict["embeddings"] @ query_emb
    matched_indices = np.where(scores >= threshold)[0]
    if len(matched_indices) == 0:
        return []

    sorted_indices = matched_indices[np.argsort(scores[matched_indices])[::-1]]
    seen = set()
    results = []
    for idx in sorted_indices:
        item_idx = int(index_dict["face_idx"][idx])
        if item_idx not in seen:
            seen.add(item_idx)
            results.append({
                "id": index_dict["ids"][item_idx],
                "rel_path": index_dict["paths"][item_idx],
            })
    return results


# --- UI ---
st.title("Find your photos & videos of Tulsi Vivah")
st.write("Upload a clear selfie or take one with your camera to pull up all your pictures and videos.")

for key, default in [
    ("matched_photos", []),
    ("photo_pairs", []),
    ("video_pairs", []),
    ("zip_path", None),
    ("search_msg", ""),
    ("searched", False),
]:
    if key not in st.session_state:
        st.session_state[key] = default

tab_upload, tab_camera = st.tabs(["Upload a Selfie", "Take a Selfie"])
with tab_upload:
    uploaded_file = st.file_uploader("Choose a clear photo of yourself", type=["jpg", "jpeg", "png", "webp"])
with tab_camera:
    camera_file = st.camera_input("Snap a clear selfie")

selfie_source = uploaded_file or camera_file

if st.button("Find My Photos & Videos", type="primary", width="stretch"):
    if selfie_source is None:
        st.warning("Please upload or take a clear selfie first.")
    else:
        t0 = time.time()
        with st.spinner("Scanning your selfie across all photos and videos..."):
            img = ImageOps.exif_transpose(Image.open(selfie_source)).convert("RGB")
            img.thumbnail((1000, 1000))
            img_bgr = np.array(img)[:, :, ::-1]

            faces = app.get(img_bgr)
            if not faces:
                st.session_state.matched_photos = []
                st.session_state.photo_pairs = []
                st.session_state.video_pairs = []
                st.session_state.zip_path = None
                st.session_state.searched = False
                st.error("No face detected. Try better lighting and face the camera directly.")
            else:
                target_face = max(faces, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]))
                query_emb = target_face.normed_embedding

                matched_photos = search_index(photo_index, query_emb, PHOTO_MATCH_THRESHOLD)
                matched_videos = search_index(video_index, query_emb, VIDEO_MATCH_THRESHOLD)

                if not matched_photos and not matched_videos:
                    st.session_state.matched_photos = []
                    st.session_state.photo_pairs = []
                    st.session_state.video_pairs = []
                    st.session_state.zip_path = None
                    st.session_state.searched = False
                    st.warning("No matching photos or videos found. Try a clearer selfie without sunglasses or blur.")
                else:
                    with ThreadPoolExecutor(max_workers=16) as pool:
                        photo_thumbs = list(pool.map(download_photo_preview, matched_photos)) if matched_photos else []
                        video_thumbs = list(pool.map(download_video_thumb, matched_videos)) if matched_videos else []

                    photo_pairs = [(p, item) for p, item in zip(photo_thumbs, matched_photos) if p is not None]
                    video_pairs = list(zip(video_thumbs, matched_videos))

                    st.session_state.matched_photos = matched_photos
                    st.session_state.photo_pairs = photo_pairs
                    st.session_state.video_pairs = video_pairs
                    st.session_state.zip_path = None
                    st.session_state.searched = True

                    elapsed = time.time() - t0
                    st.session_state.search_msg = (
                        f"Found {len(photo_pairs)} photos and {len(video_pairs)} videos in {elapsed:.1f}s!"
                    )

if st.session_state.searched:
    st.success(st.session_state.search_msg)

    photos_tab, videos_tab = st.tabs([
        f"📸 Photos ({len(st.session_state.photo_pairs)})",
        f"🎬 Videos ({len(st.session_state.video_pairs)})",
    ])

    with photos_tab:
        if not st.session_state.photo_pairs:
            st.info("No matching photos found for this selfie.")
        else:
            col_zip1, col_zip2 = st.columns([1, 2])
            with col_zip1:
                if st.button("Prepare Full-Quality Photos ZIP", width="stretch"):
                    items = st.session_state.matched_photos
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
                            label=f"Download ZIP ({len(st.session_state.matched_photos)} Full-Res Photos)",
                            data=zf,
                            file_name="tulsi_vivah_my_photos.zip",
                            mime="application/zip",
                            type="primary",
                        )

            st.divider()
            cols = st.columns(4)
            for idx, (preview_path, item) in enumerate(st.session_state.photo_pairs):
                with cols[idx % 4]:
                    st.image(preview_path, width="stretch")
                    drive_url = f"https://drive.google.com/file/d/{item['id']}/view"
                    st.markdown(f"[Open Full Quality]({drive_url})")

    with videos_tab:
        if video_index is None:
            st.info("Video search index (`video_face_index.npz`) has not been uploaded yet.")
        elif not st.session_state.video_pairs:
            st.info("No matching videos found for this selfie.")
        else:
            st.caption("Tap **Watch** to stream directly in Google Drive, or **Download** to save the MP4 to your device.")
            vcols = st.columns(3)
            for idx, (thumb_path, item) in enumerate(st.session_state.video_pairs):
                with vcols[idx % 3]:
                    if thumb_path:
                        st.image(thumb_path, width="stretch")
                    st.caption(os.path.basename(item["rel_path"]))
                    b1, b2 = st.columns(2)
                    watch_url = f"https://drive.google.com/file/d/{item['id']}/view"
                    dl_url = f"https://drive.usercontent.google.com/download?id={item['id']}&export=download&confirm=t"
                    with b1:
                        st.link_button("Watch", watch_url, width="stretch")
                    with b2:
                        st.link_button("Download", dl_url, width="stretch")

st.markdown(
    "---\n"
    "<p style='text-align: center; opacity: 0.75; font-size: 14px;'>"
    "Created by <strong>Gurudutt Mundra</strong>"
    "</p>",
    unsafe_allow_html=True,
)
