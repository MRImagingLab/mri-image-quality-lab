import io
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
from PIL import Image


st.write("App started successfully")

# -----------------------------
# MRI Image Quality Slider Lab
# -----------------------------
# Modes:
#  - Resolution: k-space crop (low-pass) -> blur
#  - SNR: add complex Gaussian noise (in k-space) -> noisy
#  - Contrast: simple phantom-based contrast scaling OR display contrast on uploaded image
#
# This app is intentionally minimal for teaching:
#   (1) choose mode
#   (2) move one slider from low -> high
#   (3) see output vs reference
#
# NOTE: This is an educational toy model. Contrast here is display/phantom contrast, not full Bloch simulation.

st.set_page_config(page_title="MRI Image Quality Slider Lab", layout="wide")

# -----------------------------
# Utilities
# -----------------------------
def fft2c(x: np.ndarray) -> np.ndarray:
    """Centered 2D FFT."""
    return np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(x)))

def ifft2c(k: np.ndarray) -> np.ndarray:
    """Centered 2D IFFT."""
    return np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(k)))

def normalize01(x: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    x = x.astype(np.float32)
    mn, mx = float(x.min()), float(x.max())
    return (x - mn) / (mx - mn + eps)

def make_simple_phantom(n: int = 256) -> np.ndarray:
    """Simple cardiac-like phantom: background + ring + blood pool + small lesion."""
    yy, xx = np.mgrid[-1:1:complex(0, n), -1:1:complex(0, n)]
    rr = np.sqrt(xx**2 + yy**2)

    bg = 0.15 * np.ones((n, n), dtype=np.float32)
    myocardium = ((rr < 0.55) & (rr > 0.35)).astype(np.float32) * 0.55
    blood = (rr <= 0.33).astype(np.float32) * 0.85

    # small "lesion" in myocardium
    lesion = (((xx + 0.25)**2 + (yy - 0.10)**2) < 0.05**2).astype(np.float32) * 0.35

    img = bg + myocardium + blood + lesion
    img = normalize01(img)
    return img

def fig_image(im: np.ndarray, title: str = "", cmap: str = "gray"):
    fig, ax = plt.subplots(figsize=(5.2, 5.2))
    ax.imshow(im, cmap=cmap, vmin=0, vmax=1)
    ax.set_title(title, fontsize=12)
    ax.axis("off")
    return fig

def fig_kspace(im: np.ndarray, title: str = "k-space (log magnitude)"):
    k = fft2c(im)
    kmag = np.log1p(np.abs(k))
    kmag = normalize01(kmag)
    fig, ax = plt.subplots(figsize=(5.2, 5.2))
    ax.imshow(kmag, cmap="gray", vmin=0, vmax=1)
    ax.set_title(title, fontsize=12)
    ax.axis("off")
    return fig

def kspace_crop_fraction(k: np.ndarray, frac: float) -> np.ndarray:
    """Keep centered square of k-space with side length = frac * N."""
    assert k.ndim == 2
    n = k.shape[0]
    m = k.shape[1]
    assert n == m, "Only square images supported in this simple demo."

    frac = float(np.clip(frac, 0.05, 1.0))
    half = int((n * frac) / 2)
    cy = n // 2
    cx = n // 2

    out = np.zeros_like(k)
    out[cy - half: cy + half, cx - half: cx + half] = k[cy - half: cy + half, cx - half: cx + half]
    return out

def apply_resolution(im_ref: np.ndarray, q: float):
    """
    Resolution: low->high means less blur.
    Uses smooth Gaussian low-pass in k-space (no hard truncation, avoids Gibbs).
    """
    sigma_max = 6.0  # increase if you want stronger blur at low quality
    sigma = sigma_max * (1.0 - q) ** 1.5
    im_out = gaussian_lowpass_fft(im_ref, sigma_px=sigma)
    return im_out, {"blur_sigma_px": sigma}



def apply_snr(im_ref: np.ndarray, q: float):
    """SNR: low->high means noise decreases."""
    # Add complex Gaussian noise in k-space (teaches acquisition noise intuition).
    # sigma is relative to max |k|.
    k = fft2c(im_ref)
    kmax = np.max(np.abs(k)) + 1e-8
    sigma_max = 0.10  # feel free to adjust; chosen for visible effect
    sigma = sigma_max * (1.0 - q) ** 2
    noise = (np.random.randn(*k.shape) + 1j * np.random.randn(*k.shape)) * (sigma * kmax / np.sqrt(2.0))
    k_noisy = k + noise
    im_out = np.abs(ifft2c(k_noisy))
    im_out = normalize01(im_out)
    # A simple, stable proxy for "relative SNR" (not a rigorous magnitude-MRI SNR definition)
    rsnr = 1.0 / (sigma + 1e-6)
    return im_out, {"noise_sigma_rel": sigma, "relative_snr_proxy": rsnr}

def apply_contrast_phantom(im_ref: np.ndarray, q: float):
    """Contrast: low->high means tissue differences increase (phantom-style)."""
    # Contrast scaling around the median intensity (robust to outliers)
    center = float(np.median(im_ref))
    c = 0.35 + 1.65 * q  # 0.35x to 2.0x contrast
    im_out = center + c * (im_ref - center)
    im_out = np.clip(im_out, 0.0, 1.0)
    return im_out, {"contrast_scale": c}

def apply_contrast_display(im_ref: np.ndarray, q: float):
    """Contrast for uploaded images: window/level + gamma-like display mapping."""
    # Keep it simple: gamma mapping (low contrast -> gamma>1 flattening; high -> gamma<1)
    gamma = 2.2 - 1.8 * q  # 2.2 (low) to 0.4 (high)
    im_out = np.clip(im_ref, 0.0, 1.0) ** gamma
    im_out = normalize01(im_out)
    return im_out, {"gamma": gamma}

def load_uploaded_image(file, n_target: int = 256) -> np.ndarray:
    """Load image as grayscale float32 [0,1] and resize to n_target x n_target."""
    img = Image.open(file).convert("L")
    img = img.resize((n_target, n_target), resample=Image.BICUBIC)
    arr = np.asarray(img).astype(np.float32)
    arr = normalize01(arr)
    return arr

def gaussian_kernel1d(sigma: float, radius: int = None) -> np.ndarray:
    sigma = float(max(sigma, 1e-6))
    if radius is None:
        radius = int(np.ceil(3.0 * sigma))
    x = np.arange(-radius, radius + 1, dtype=np.float32)
    k = np.exp(-(x**2) / (2.0 * sigma**2))
    k /= (k.sum() + 1e-8)
    return k

def conv1d_reflect(im: np.ndarray, k: np.ndarray, axis: int) -> np.ndarray:
    # Reflect padding and 1D convolution along one axis
    pad = len(k) // 2
    pad_width = [(0, 0), (0, 0)]
    pad_width[axis] = (pad, pad)
    x = np.pad(im, pad_width, mode="reflect")
    # Move axis to last for easier convolution
    x = np.moveaxis(x, axis, -1)
    out = np.zeros_like(x, dtype=np.float32)
    # Convolve along last axis
    for i in range(out.shape[-1]):
        # indices in padded signal
        j0 = i
        j1 = i + len(k)
        out[..., i] = np.sum(x[..., j0:j1] * k[None, None, :], axis=-1)
    out = np.moveaxis(out, -1, axis)
    return out

def gaussian_blur2d(im: np.ndarray, sigma: float) -> np.ndarray:
    k = gaussian_kernel1d(sigma)
    out = conv1d_reflect(im, k, axis=0)
    out = conv1d_reflect(out, k, axis=1)
    return out

def gaussian_lowpass_fft(im: np.ndarray, sigma_px: float) -> np.ndarray:
    """
    Gaussian low-pass in k-space using a smooth Gaussian, avoiding Gibbs ringing.
    sigma_px is in image pixels (higher sigma -> more blur).
    """
    sigma_px = float(max(sigma_px, 0.0))
    if sigma_px < 1e-6:
        return im.copy()

    n, m = im.shape
    # frequency coordinates in cycles/pixel
    fy = np.fft.fftshift(np.fft.fftfreq(n))
    fx = np.fft.fftshift(np.fft.fftfreq(m))
    FY, FX = np.meshgrid(fy, fx, indexing="ij")

    # Convert sigma in pixels to sigma in frequency domain (roughly inverse relation)
    sigma_f = 1.0 / (2.0 * np.pi * sigma_px + 1e-8)

    H = np.exp(-(FX**2 + FY**2) / (2.0 * sigma_f**2)).astype(np.float32)

    k = fft2c(im)
    k2 = k * H
    out = np.abs(ifft2c(k2))
    return normalize01(out)
def apply_artifact_gibbs(im_ref: np.ndarray, q: float):
    """
    Gibbs ringing (artifact): hard k-space truncation.
    Higher q => stronger ringing (more truncation).
    """
    frac = 1.0 - 0.75 * float(np.clip(q, 0.0, 1.0))  # 1.00 -> 0.25
    k = fft2c(im_ref)
    k2 = kspace_crop_fraction(k, frac)
    im_out = np.abs(ifft2c(k2))
    im_out = normalize01(im_out)
    return im_out, {"kspace_fraction": frac}


def apply_artifact_motion(im_ref: np.ndarray, q: float):
    """
    Motion ghosting (teaching model): phase errors on a subset of ky lines.
    Higher q => more lines affected + stronger phase ramp.
    """
    k = fft2c(im_ref)
    n = k.shape[0]
    frac_lines = 0.05 + 0.35 * float(np.clip(q, 0.0, 1.0))
    num = max(1, int(frac_lines * n))

    rng = np.random.default_rng(0)  # fixed seed = reproducible for class
    idx = rng.choice(np.arange(n), size=num, replace=False)

    x = np.linspace(-np.pi, np.pi, n, dtype=np.float32)
    ramp = np.exp(1j * (0.5 + 3.0 * q) * x)[None, :]  # strength increases with q

    k2 = k.copy()
    k2[idx, :] *= ramp
    im_out = np.abs(ifft2c(k2))
    im_out = normalize01(im_out)
    return im_out, {"affected_ky_lines": num}


def apply_artifact_aliasing(im_ref: np.ndarray, q: float):
    """
    Aliasing (wrap/undersampling): keep every R-th ky line (zero-filled recon).
    Higher q => higher R.
    """
    k = fft2c(im_ref)
    n = k.shape[0]
    R = int(np.round(1 + 7 * float(np.clip(q, 0.0, 1.0))))  # 1..8
    R = max(R, 1)

    k2 = np.zeros_like(k)
    k2[::R, :] = k[::R, :]
    im_out = np.abs(ifft2c(k2))
    im_out = normalize01(im_out)
    return im_out, {"acceleration_R": R}

def apply_artifact_spike_stripes(im_ref: np.ndarray, q: float):
    """
    Spike artifact in k-space -> stripes/streaks in image.
    Higher q => stronger spikes.
    """
    k = fft2c(im_ref)
    n = k.shape[0]

    rng = np.random.default_rng(0)
    num_spikes = int(1 + 10 * q)
    amp = (0.1 + 1.5 * q) * np.max(np.abs(k))

    k2 = k.copy()
    for _ in range(num_spikes):
        iy = rng.integers(0, n)
        ix = rng.integers(0, n)
        k2[iy, ix] += amp * (rng.standard_normal() + 1j * rng.standard_normal())

    im_out = normalize01(np.abs(ifft2c(k2)))
    return im_out, {"num_spikes": num_spikes}

def apply_artifact_shading(im_ref: np.ndarray, q: float):
    """
    Stronger RF/coil shading (bias field).
    Higher q => stronger intensity non-uniformity.
    """
    n, m = im_ref.shape
    yy, xx = np.mgrid[0:n, 0:m].astype(np.float32)
    xx = (xx / (m - 1)) - 0.5
    yy = (yy / (n - 1)) - 0.5

    # Increase amplitude: at q=1, allow up to ~3x variation
    a = 0.0 + 2.5 * float(np.clip(q, 0.0, 1.0))

    # Smooth bias field with low-order terms + low-frequency sinusoid
    field = (
        1.0
        + a * (0.9*xx + 0.6*yy + 0.6*xx*yy)
        + (0.6 * q) * np.sin(2*np.pi*xx*1.0)
    )

    # Clip to keep it physical and avoid negative intensities
    field = np.clip(field, 0.15, 3.5)

    im_out = normalize01(im_ref * field)
    return im_out, {"shading_amp": a, "field_min": float(field.min()), "field_max": float(field.max())}


def apply_artifact_banding(im_ref: np.ndarray, q: float):
    """
    bSSFP banding (teaching model): apply periodic dark bands based on a synthetic off-resonance phase map.
    Higher q => stronger banding.
    """
    n, m = im_ref.shape
    yy, xx = np.mgrid[0:n, 0:m].astype(np.float32)

    # Create a smooth spatial field (toy B0 map): combination of gradients + low-frequency sinusoid
    field = (
        0.8 * (xx / m - 0.5) +
        0.6 * (yy / n - 0.5) +
        0.35 * np.sin(2 * np.pi * xx / (0.45 * m)) +
        0.25 * np.sin(2 * np.pi * yy / (0.35 * n))
    )

    # Map to phase range; q controls "off-resonance strength"
    strength = 0.5 + 4.0 * float(np.clip(q, 0.0, 1.0))
    phi = strength * field * np.pi  # phase-like

    # Toy bSSFP banding: signal drops near cos(phi)= -1 (phi ~ pi)
    band = np.abs(np.cos(phi))  # 0 at pi/2? (periodic modulation)
    # Make bands darker and more obvious with exponent
    band = band ** (0.6 + 2.0 * q)

    im_out = normalize01(im_ref * band)
    return im_out, {"band_strength": strength}

def apply_artifact_partial_volume(im_ref: np.ndarray, q: float):
    """
    Partial volume effect: voxel averaging due to finite resolution.
    Higher q => stronger partial volume (coarser effective voxel size).
    """
    # Effective downsampling factor
    R = int(1 + 6 * float(np.clip(q, 0.0, 1.0)))  # 1..7
    R = max(R, 1)

    if R == 1:
        return im_ref.copy(), {"downsample_factor": 1}

    n, m = im_ref.shape
    # Crop to be divisible
    n2 = (n // R) * R
    m2 = (m // R) * R
    im = im_ref[:n2, :m2]

    # Block averaging (true voxel averaging)
    im_ds = im.reshape(n2//R, R, m2//R, R).mean(axis=(1, 3))

    # Upsample back (nearest-neighbor to show blockiness)
    im_us = np.repeat(np.repeat(im_ds, R, axis=0), R, axis=1)

    im_out = normalize01(im_us)
    return im_out, {"downsample_factor": R}


# -----------------------------
# Sidebar controls
# -----------------------------
st.title("MRI Image Quality Slider Lab")
st.caption("A minimal teaching UI: choose a quality dimension, then move one slider from low → high to see the effect.")

with st.sidebar:
    st.header("Controls")

    source = st.radio("Input", ["Phantom (default)", "Upload image"], index=0)
    n = st.selectbox("Image size", [128, 192, 256, 320, 384, 512], index=2)

    mode = st.selectbox(
        "Quality dimension",
        ["Resolution", "SNR", "Contrast", "Artifacts"],
        index=0
    )

    artifact_type = None
    if mode == "Artifacts":
        artifact_type = st.selectbox(
            "Artifact type",
            [
                "Gibbs ringing",
                "Motion ghosting",
                "Aliasing (wrap/undersample)",
                "bSSFP banding",
                "Shading (B1 / coil)",
                "Spike/stripe (streak-like)",
                "Partial volume (voxel averaging)"
            ],
            index=0
        )


    if mode == "Artifacts":
        q = st.slider("Artifact strength (weak → strong)", 0.0, 1.0, 0.25, 0.01)
    else:
        q = st.slider("Quality (low → high)", 0.0, 1.0, 0.25, 0.01)

    show_k = st.checkbox("Show k-space (log magnitude)", value=False)


    st.markdown("---")
    st.subheader("Notes (teaching)")

    # General notes
    st.write("• **Resolution** uses low-pass filtering (loss of high spatial frequencies).")
    st.write("• **SNR** adds noise; higher noise lowers SNR but does not change resolution.")
    st.write("• **Contrast** changes intensity separation between tissues.")

    # Artifact-specific notes
    if mode == "Artifacts":
        artifact_notes = {
            "Gibbs ringing": (
                "• **Gibbs ringing**: finite k-space truncation causes oscillations near sharp edges.\n"
                "  **Correction**: acquire more k-space lines, apply apodization, or increase resolution."
            ),
            "Motion ghosting": (
                "• **Motion ghosting**: inconsistent phase encoding due to motion produces repeated ghosts.\n"
                "  **Correction**: motion correction, gating, faster acquisition."
            ),
            "Aliasing (wrap/undersample)": (
                "• **Aliasing (wrap-around)**: insufficient FOV or undersampling causes anatomy to fold into the image.\n"
                "  **Correction**: increase FOV, phase oversampling, parallel imaging."
            ),
            "bSSFP banding": (
                "• **bSSFP banding**: off-resonance in bSSFP creates periodic signal nulls (dark bands).\n"
                "  **Correction**: better shimming, shorter TR, phase-cycled acquisitions."
            ),
            "Shading (B1 / coil)": (
                "• **Shading (B1 / coil sensitivity)**: receive field inhomogeneity causes intensity non-uniformity.\n"
                "  **Correction**: bias-field correction, improved coil combination."
            ),
            "Spike/stripe (streak-like)": (
                "• **Spike (k-space) / stripe artifact**: isolated high-amplitude k-space samples produce global "
                "periodic stripes/ripples in the image.\n"
                "  **Correction**: identify/remove corrupted k-space lines/points, improve RF shielding, "
                "repeat acquisition if severe, and use robust reconstruction/outlier rejection."
            ),

            "Partial volume (voxel averaging)": (
                "• **Partial volume effect**: multiple tissues within a voxel produce averaged signal.\n"
                "  **Correction**: increase spatial resolution, thinner slices."
            ),
        }

        note = artifact_notes.get(artifact_type, None)
        if note:
            st.info(note)

# -----------------------------
# Load reference image
# -----------------------------
if source == "Upload image":
    file = st.sidebar.file_uploader("Upload a grayscale MRI image (PNG/JPG).", type=["png", "jpg", "jpeg"])
    if file is None:
        st.info("Upload an image to proceed, or switch back to Phantom.")
        st.stop()
    im_ref = load_uploaded_image(file, n_target=int(n))
    is_phantom = False
else:
    im_ref = make_simple_phantom(int(n))
    is_phantom = True

# -----------------------------
# Apply selected effect
# -----------------------------
if mode == "Resolution":
    im_out, info = apply_resolution(im_ref, q)
    info_line = f"Gaussian blur σ: {info['blur_sigma_px']:.2f} px"

elif mode == "SNR":
    im_out, info = apply_snr(im_ref, q)
    info_line = (
        f"relative noise sigma (k-space): {info['noise_sigma_rel']:.4f} | "
        f"relative SNR proxy: {info['relative_snr_proxy']:.1f}"
    )

elif mode == "Contrast":
    if is_phantom:
        im_out, info = apply_contrast_phantom(im_ref, q)
        info_line = f"contrast scale: {info['contrast_scale']:.2f}×"
    else:
        im_out, info = apply_contrast_display(im_ref, q)
        info_line = f"display gamma: {info['gamma']:.2f}"

elif mode == "Artifacts":
    if artifact_type == "Gibbs ringing":
        im_out, info = apply_artifact_gibbs(im_ref, q)
        info_line = f"Artifact: Gibbs ringing | k-space fraction kept: {info['kspace_fraction']*100:.1f}%"

    elif artifact_type == "Motion ghosting":
        im_out, info = apply_artifact_motion(im_ref, q)
        info_line = f"Artifact: Motion ghosting | affected ky lines: {info['affected_ky_lines']}"

    elif artifact_type == "Aliasing (wrap/undersample)":
        im_out, info = apply_artifact_aliasing(im_ref, q)
        info_line = f"Artifact: Aliasing (wrap) | undersampling factor R={info['acceleration_R']}"

    elif artifact_type == "bSSFP banding":
        im_out, info = apply_artifact_banding(im_ref, q)
        info_line = f"Artifact: bSSFP banding | strength={info['band_strength']:.2f}"

    elif artifact_type == "Shading (B1 / coil)":
        im_out, info = apply_artifact_shading(im_ref, q)
        info_line = f"Artifact: Shading | amp={info['shading_amp']:.2f} | field range={info['field_min']:.2f}–{info['field_max']:.2f}"
    
    elif artifact_type == "Partial volume (voxel averaging)":
        im_out, info = apply_artifact_partial_volume(im_ref, q)
        info_line = f"Partial volume | effective voxel factor={info['downsample_factor']}×"

    else:  # "Spike/stripe (streak-like)"
        im_out, info = apply_artifact_spike_stripes(im_ref, q)
        info_line = f"Artifact: Spike/stripe | spikes injected: {info['num_spikes']}"
        



# -----------------------------
# Layout: output vs reference
# -----------------------------
col1, col2 = st.columns(2, gap="large")

with col1:
    st.subheader("Output (after slider)")

    if mode == "Artifacts":
        title = f"{mode} | strength={q:.2f}"
    else:
        title = f"{mode} | quality={q:.2f}"

    st.pyplot(fig_image(im_out, title=title), clear_figure=True)
    st.caption(info_line)



with col2:
    st.subheader("Reference (highest quality)")
    st.pyplot(fig_image(im_ref, title="Reference"), clear_figure=True)

if show_k:
    kcol1, kcol2 = st.columns(2, gap="large")
    with kcol1:
        st.subheader("Output k-space")
        st.pyplot(fig_kspace(im_out, title="Output k-space (log |K|)"), clear_figure=True)
    with kcol2:
        st.subheader("Reference k-space")
        st.pyplot(fig_kspace(im_ref, title="Reference k-space (log |K|)"), clear_figure=True)

st.markdown("---")
st.write("Implementation note: This app intentionally uses simplified models so students can develop intuition quickly. "
         "For research-grade MR contrast modeling, one would use Bloch simulations and sequence-specific signal equations.")
