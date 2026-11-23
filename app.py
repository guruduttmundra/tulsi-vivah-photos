import os
import time
import uuid
import shutil
import zipfile
import mimetypes
import subprocess
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
WEB_VID_CACHE = "/tmp/web_vid_cache"

PHOTO_MATCH_THRESHOLD = 0.42  # Lower to 0.38 if missing photos; raise to 0.48 if false matches appear
VIDEO_MATCH_THRESHOLD = 0.40  # Slightly more lenient for moving faces in video

for d in (THUMB_CACHE, ORIG_CACHE, WEB_VID_CACHE):
    os.makedirs(d, exist_ok=True)


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


def prune_cache(folder, max_mb=400):
    """Remove oldest cached files if folder exceeds max_mb to protect free Cloud disk."""
    try:
        entries = []
        total = 0
        for name in os.listdir(folder):
            p = os.path.join(folder, name)
            if os.path.isfile(p) and not name.endswith(".part"):
                st_info = os.stat(p)
                total += st_info.st_size
                entries.append((st_info.st_mtime, st_info.st_size, p))
        limit = max_mb * 1024 * 1024
        if total <= limit:
            return
        entries.sort()
        for _, sz, p in entries:
            try:
                os.remove(p)
                total -= sz
            except OSError:
                pass
            if total <= limit * 0.75:
                break
    except Exception:
        pass


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

    prune_cache(ORIG_CACHE, max_mb=400)
    urls = [
        f"https://drive.usercontent.google.com/download?id={fid}&export=download&confirm=t",
        f"https://drive.google.com/uc?export=download&id={fid}",
    ]
    tmp_path = f"{local_path}.{uuid.uuid4().hex[:6]}.part"

    for url in urls:
        for attempt in range(2):
            try:
                with session.get(url, timeout=90, allow_redirects=True, stream=True) as r:
                    ctype = r.headers.get("content-type", "").lower()
                    if r.status_code == 200 and not ctype.startswith("text/html"):
                        with open(tmp_path, "wb") as f:
                            for chunk in r.iter_content(chunk_size=512 * 1024):
                                if chunk:
                                    f.write(chunk)
                        if os.path.getsize(tmp_path) > 0:
                            os.replace(tmp_path, local_path)
                            return local_path, item["rel_path"]
                    if r.status_code in (429, 500, 502, 503, 504):
                        time.sleep(1.5 * (attempt + 1))
                        continue
                    break
            except Exception:
                time.sleep(1)
            finally:
                if os.path.exists(tmp_path):
                    try:
                        os.remove(tmp_path)
                    except OSError:
                        pass
    return None, item["rel_path"]


def get_web_video(item):
    """Creates a lightweight, fast-start MP4 preview for smooth zero-buffer streaming."""
    fid = item["id"]
    web_path = os.path.join(WEB_VID_CACHE, f"{fid}_web.mp4")
    if os.path.exists(web_path) and os.path.getsize(web_path) > 0:
        return web_path

    orig_path, _ = download_original(item)
    if not orig_path or not os.path.exists(orig_path):
        return None

    ffmpeg_bin = shutil.which("ffmpeg")
    if not ffmpeg_bin:
        return orig_path

    prune_cache(WEB_VID_CACHE, max_mb=250)
    tmp_out = f"{web_path}.{uuid.uuid4().hex[:6]}.mp4"
    size_mb = os.path.getsize(orig_path) / (1024 * 1024)

    try:
        if size_mb <= 5.0:
            # Already small: just move moov atom to front for instant playback (<0.3s)
            cmd = [
                ffmpeg_bin, "-y", "-i", orig_path,
                "-c", "copy", "-movflags", "+faststart",
                tmp_out,
            ]
        else:
            # Compress heavy camera clip into smooth 720p fast-start web MP4
            cmd = [
                ffmpeg_bin, "-y", "-i", orig_path,
                "-vf", "scale='min(720,iw)':-2",
                "-c:v", "libx264", "-preset", "ultrafast", "-crf", "28",
                "-maxrate", "1500k", "-bufsize", "3000k",
                "-pix_fmt", "yuv420p",
                "-c:a", "aac", "-b:a", "96k",
                "-movflags", "+faststart",
                tmp_out,
            ]

        res = subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=120)
        if res.returncode == 0 and os.path.exists(tmp_out) and os.path.getsize(tmp_out) > 0:
            os.replace(tmp_out, web_path)
            return web_path
    except Exception:
        pass
    finally:
        if os.path.exists(tmp_out):
            try:
                os.remove(tmp_out)
            except OSError:
                pass

    return orig_path


def make_direct_downloader(item, fallback_path=None):
    """Lazy callable that fetches full-res bytes only when clicked."""
    def _fetch_bytes():
        local_path, _ = download_original(item)
        target = local_path or fallback_path
        if target and os.path.exists(target):
            with open(target, "rb") as f:
                return f.read()
        return b""
    return _fetch_bytes


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


@st.dialog("Video Player", width="large")
def open_video_dialog(item, label, clean_name, mime_type):
    st.caption(f"{label} — Playing optimized web preview. Use the button below for full original HD.")
    with st.spinner("Preparing smooth video playback..."):
        web_path = get_web_video(item)
    if web_path and os.path.exists(web_path):
        st.video(web_path, autoplay=True)
        st.download_button(
            label="Download Original HD Video",
            data=make_direct_downloader(item, fallback_path=web_path),
            file_name=clean_name,
            mime=mime_type,
            key=f"modal_dl_{item['id']}",
            type="primary",
            on_click="ignore",
            width="stretch",
        )
    else:
        st.error("Could not load this video right now. Try the Download button on the card instead.")


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
                if st.button("Prepare All Photos (ZIP)", width="stretch"):
                    items = st.session_state.matched_photos
                    total = len(items)
                    bar = st.progress(0, text=f"Preparing 0/{total} full-resolution photos...")
                    downloaded = []
                    with ThreadPoolExecutor(max_workers=10) as pool:
                        futures = [pool.submit(download_original, item) for item in items]
                        for done, fut in enumerate(as_completed(futures), 1):
                            local_path, rel_path = fut.result()
                            if local_path:
                                downloaded.append((local_path, rel_path))
                            bar.progress(done / total, text=f"Preparing {done}/{total} full-resolution photos...")

                    if downloaded:
                        zip_path = f"/tmp/tulsi_vivah_photos_{uuid.uuid4().hex[:8]}.zip"
                        with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_STORED) as zf:
                            for i, (local_path, rel_path) in enumerate(downloaded, 1):
                                ext = os.path.splitext(rel_path)[1] or ".jpg"
                                zf.write(local_path, arcname=f"Tulsi_Vivah_Photo_{i:03d}{ext}")
                        st.session_state.zip_path = zip_path
                        bar.empty()
                    else:
                        bar.empty()
                        st.error("Could not prepare the ZIP right now. Please try again.")

            with col_zip2:
                if st.session_state.zip_path and os.path.exists(st.session_state.zip_path):
                    with open(st.session_state.zip_path, "rb") as zf:
                        st.download_button(
                            label=f"Download ZIP ({len(st.session_state.matched_photos)} Full-Res Photos)",
                            data=zf,
                            file_name="Tulsi_Vivah_My_Photos.zip",
                            mime="application/zip",
                            type="primary",
                            on_click="ignore",
                        )

            st.divider()
            cols = st.columns(4)
            for idx, (preview_path, item) in enumerate(st.session_state.photo_pairs):
                ext = os.path.splitext(item["rel_path"])[1] or ".jpg"
                clean_name = f"Tulsi_Vivah_Photo_{idx + 1:02d}{ext}"
                mime_type = mimetypes.guess_type(clean_name)[0] or "image/jpeg"
                with cols[idx % 4]:
                    st.image(preview_path, width="stretch")
                    st.download_button(
                        label="Download HD",
                        data=make_direct_downloader(item, fallback_path=preview_path),
                        file_name=clean_name,
                        mime=mime_type,
                        key=f"dl_photo_{item['id']}",
                        on_click="ignore",
                        width="stretch",
                    )

    with videos_tab:
        if video_index is None:
            st.info("Video search index (`video_face_index.npz`) has not been uploaded yet.")
        elif not st.session_state.video_pairs:
            st.info("No matching videos found for this selfie.")
        else:
            st.caption("Tap **Watch** to play in a popup right here, or **Download** to save the original HD video.")
            vcols = st.columns(3)
            for idx, (thumb_path, item) in enumerate(st.session_state.video_pairs):
                ext = os.path.splitext(item["rel_path"])[1] or ".mp4"
                clean_name = f"Tulsi_Vivah_Video_{idx + 1:02d}{ext}"
                mime_type = mimetypes.guess_type(clean_name)[0] or "video/mp4"
                with vcols[idx % 3]:
                    if thumb_path:
                        st.image(thumb_path, width="stretch")
                    st.caption(f"Video {idx + 1}")
                    b1, b2 = st.columns(2)
                    with b1:
                        if st.button("Watch", key=f"watch_{item['id']}", width="stretch"):
                            open_video_dialog(item, f"Video {idx + 1}", clean_name, mime_type)
                    with b2:
                        st.download_button(
                            label="Download",
                            data=make_direct_downloader(item),
                            file_name=clean_name,
                            mime=mime_type,
                            key=f"dl_vid_{item['id']}",
                            on_click="ignore",
                            width="stretch",
                        )

st.markdown(
    "---\n"
    "<p style='text-align: center; opacity: 0.75; font-size: 14px;'>"
    "Created by <strong>Gurudutt Mundra</strong>"
    "</p>",
    unsafe_allow_html=True,
)
