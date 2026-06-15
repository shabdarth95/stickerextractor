import streamlit as st
from PIL import Image
import numpy as np
import cv2
import io
import zipfile

# ── Page config ────────────────────────────────────────────────────
st.set_page_config(
    page_title="Sticker Element Extractor",
    page_icon="✂️",
    layout="wide",
)

# ── Styles ─────────────────────────────────────────────────────────
st.markdown("""
<style>
    .main { background-color: #f8f8f8; }
    .block-container { padding-top: 2rem; }
    h1 { color: #1a1a1a; }
    .stDownloadButton > button {
        width: 100%;
        background-color: #111;
        color: white;
        border-radius: 6px;
        font-size: 12px;
        padding: 4px 0;
    }
    .stDownloadButton > button:hover { background-color: #333; }
    .element-card {
        background: white;
        border-radius: 10px;
        padding: 8px;
        margin-bottom: 8px;
        box-shadow: 0 1px 4px rgba(0,0,0,0.08);
    }
</style>
""", unsafe_allow_html=True)

# ── Header ──────────────────────────────────────────────────────────
st.title("✂️ Sticker Element Extractor")
st.markdown(
    "Upload a sticker sheet or cutout image — each connected element is "
    "automatically detected and extracted as a separate transparent PNG."
)
st.divider()


# ── Helpers ─────────────────────────────────────────────────────────
def detect_background_threshold(brightness: np.ndarray) -> int:
    """
    Auto-detect the background brightness threshold by sampling image corners.
    Returns a brightness value below which pixels are considered 'element' content.
    """
    h, w = brightness.shape
    sample_size = min(30, h // 10, w // 10)

    corners = [
        brightness[:sample_size,  :sample_size ],   # top-left
        brightness[:sample_size,  -sample_size:],   # top-right
        brightness[-sample_size:, :sample_size ],   # bottom-left
        brightness[-sample_size:, -sample_size:],   # bottom-right
    ]
    corner_vals = np.concatenate([c.ravel() for c in corners])
    bg_brightness = np.median(corner_vals)

    # Threshold = background brightness minus tolerance
    threshold = int(bg_brightness * 0.92)
    return max(threshold, 400)   # never go below 400


def build_element_mask(arr: np.ndarray) -> np.ndarray:
    """
    Returns a uint8 binary mask where 255 = element pixel, 0 = background.

    Strategy:
      • RGBA image with transparent pixels → alpha == 0 marks element regions
      • RGB / fully-opaque RGBA             → brightness threshold approach
    """
    r, g, b = arr[:,:,0], arr[:,:,1], arr[:,:,2]
    brightness = r.astype(int) + g.astype(int) + b.astype(int)

    if arr.shape[2] == 4:
        a = arr[:,:,3]
        transparent_count = (a == 0).sum()
        total = a.size
        if transparent_count > total * 0.01:          # >1% transparent pixels
            # Alpha-based mask: transparent = element
            return (a == 0).astype(np.uint8) * 255

    # Brightness-based mask
    threshold = detect_background_threshold(brightness)
    return (brightness < threshold).astype(np.uint8) * 255


def extract_elements(source_rgb: np.ndarray, element_mask: np.ndarray,
                     min_pixels: int = 100):
    """
    Runs 8-connected component analysis on element_mask.
    For each component, extracts the corresponding pixels from source_rgb.
    Returns list of RGBA PIL Images, sorted top→bottom, left→right.
    """
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        element_mask, connectivity=8
    )

    components = []
    for lbl in range(1, num_labels):
        px = stats[lbl, cv2.CC_STAT_AREA]
        if px < min_pixels:
            continue
        x = stats[lbl, cv2.CC_STAT_LEFT]
        y = stats[lbl, cv2.CC_STAT_TOP]
        w = stats[lbl, cv2.CC_STAT_WIDTH]
        h = stats[lbl, cv2.CC_STAT_HEIGHT]
        components.append((lbl, px, x, y, w, h))

    components.sort(key=lambda c: (c[3], c[2]))   # top→bottom, left→right

    results = []
    for lbl, px, x, y, w, h in components:
        src_crop   = source_rgb[y:y+h, x:x+w]
        label_crop = labels[y:y+h, x:x+w]

        canvas          = np.zeros((h, w, 4), dtype=np.uint8)
        mask            = (label_crop == lbl)
        canvas[mask, 0] = src_crop[mask, 0]
        canvas[mask, 1] = src_crop[mask, 1]
        canvas[mask, 2] = src_crop[mask, 2]
        canvas[mask, 3] = 255

        results.append({
            "image":  Image.fromarray(canvas, "RGBA"),
            "pixels": px,
            "bbox":   (x, y, w, h),
        })

    return results


def pil_to_bytes(img: Image.Image) -> bytes:
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def make_checkerboard(w: int, h: int, sq: int = 10) -> Image.Image:
    cb = Image.new("RGB", (w, h), (255, 255, 255))
    from PIL import ImageDraw
    draw = ImageDraw.Draw(cb)
    for row in range(0, h, sq):
        for col in range(0, w, sq):
            if (row // sq + col // sq) % 2 == 0:
                draw.rectangle([col, row, col+sq-1, row+sq-1], fill=(210, 210, 210))
    return cb


def composite_on_checker(elem_img: Image.Image, size: int = 180) -> Image.Image:
    ew, eh = elem_img.size
    scale  = min(size / ew, size / eh, 1.0)
    nw, nh = max(1, int(ew * scale)), max(1, int(eh * scale))
    thumb  = elem_img.resize((nw, nh), Image.LANCZOS)
    cb     = make_checkerboard(size, size)
    ox     = (size - nw) // 2
    oy     = (size - nh) // 2
    cb.paste(thumb, (ox, oy), mask=thumb)
    return cb


# ── Main UI ─────────────────────────────────────────────────────────
uploaded_file = st.file_uploader(
    "Drop your image here",
    type=["png", "jpg", "jpeg", "webp"],
    help="Supports PNG (with or without transparency), JPG, and WebP",
)

if uploaded_file is not None:
    # Load image
    pil_img  = Image.open(uploaded_file)
    arr_rgba = np.array(pil_img.convert("RGBA"))
    arr_rgb  = np.array(pil_img.convert("RGB"))

    col_preview, col_info = st.columns([1, 2])
    with col_preview:
        st.markdown("**Original image**")
        display_img = pil_img.convert("RGBA")
        cb_bg       = make_checkerboard(*display_img.size, sq=14)
        cb_bg.paste(display_img, mask=display_img)
        st.image(cb_bg, use_container_width=True)

    with col_info:
        st.markdown("**Image info**")
        st.write(f"Size: `{pil_img.width} × {pil_img.height}` px")
        st.write(f"Mode: `{pil_img.mode}`")

    # Processing
    with st.spinner("Detecting connected elements…"):
        element_mask = build_element_mask(arr_rgba)
        elements     = extract_elements(arr_rgb, element_mask, min_pixels=100)

    if not elements:
        st.warning("No elements found. Try a different image.")
        st.stop()

    st.success(f"✅ Found **{len(elements)} elements**")
    st.divider()

    # ── Download All ZIP ──────────────────────────────────────────
    zip_buf = io.BytesIO()
    with zipfile.ZipFile(zip_buf, "w", zipfile.ZIP_DEFLATED) as zf:
        for i, elem in enumerate(elements, 1):
            zf.writestr(f"element_{i:02d}.png", pil_to_bytes(elem["image"]))
    zip_buf.seek(0)

    st.download_button(
        label=f"📦  Download all {len(elements)} elements as ZIP",
        data=zip_buf.getvalue(),
        file_name="extracted_elements.zip",
        mime="application/zip",
        use_container_width=True,
    )

    st.divider()
    st.markdown(f"### Individual elements — click to download each")

    # ── Grid of elements ──────────────────────────────────────────
    COLS = 4
    for row_start in range(0, len(elements), COLS):
        cols = st.columns(COLS)
        for col_i, elem in enumerate(elements[row_start : row_start + COLS]):
            idx = row_start + col_i + 1
            with cols[col_i]:
                preview = composite_on_checker(elem["image"], size=200)
                st.image(preview, use_container_width=True)
                st.caption(
                    f"Element {idx} · {elem['image'].width}×{elem['image'].height}px"
                )
                st.download_button(
                    label="⬇ Download PNG",
                    data=pil_to_bytes(elem["image"]),
                    file_name=f"element_{idx:02d}.png",
                    mime="image/png",
                    key=f"dl_{idx}",
                    use_container_width=True,
                )

else:
    # Empty state
    st.markdown(
        """
        <div style="text-align:center; padding: 60px 20px; color: #888;">
            <div style="font-size: 64px;">🖼️</div>
            <p style="font-size: 18px; margin-top: 12px;">
                Upload a sticker sheet to get started
            </p>
            <p style="font-size: 14px;">
                Each connected element will be extracted as an individual transparent PNG
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )
