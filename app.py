"""Application Streamlit interactive pour le projet PROJET_IMAGE_4.ipynb
(restauration d'images par Plug-and-Play avec GS-DRUNet).

Lancer : streamlit run app.py
"""
import sys
import time
from pathlib import Path

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
import pnp_core as core  # noqa: E402

ASSETS = ROOT / "assets"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

st.set_page_config(page_title="Restauration d'images PnP", page_icon="🔬", layout="wide")


# ============================================================
# CACHE
# ============================================================
@st.cache_resource(show_spinner="Chargement du débruiteur GS-DRUNet…")
def get_denoiser():
    return core.load_denoiser(DEVICE)


def load_image(source, max_side: int) -> Image.Image:
    img = Image.open(source).convert("RGB")
    img.thumbnail((max_side, max_side), Image.LANCZOS)
    # dimensions paires : évite les soucis de FFT / U-Net
    w, h = img.size
    return img.crop((0, 0, w - w % 8, h - h % 8))


def pil_to_tensor(img: Image.Image) -> torch.Tensor:
    arr = np.asarray(img, dtype=np.float32) / 255.0
    return torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).to(DEVICE)


def convergence_chart(residuals, psnrs, res_label="Résidu ‖xₖ₊₁ − xₖ‖"):
    c1, c2 = st.columns(2)
    df_r = pd.DataFrame({"Itération": range(1, len(residuals) + 1), "Résidu": residuals})
    c1.altair_chart(
        alt.Chart(df_r, title=res_label).mark_line(point=True)
        .encode(x="Itération", y=alt.Y("Résidu", scale=alt.Scale(type="log"), title="échelle log")),
        width="stretch",
    )
    df_p = pd.DataFrame({"Itération": range(1, len(psnrs) + 1), "PSNR (dB)": psnrs})
    c2.altair_chart(
        alt.Chart(df_p, title="Évolution du PSNR").mark_line(point=True, color="#e4572e")
        .encode(x="Itération", y=alt.Y("PSNR (dB)", scale=alt.Scale(zero=False))),
        width="stretch",
    )


def show_triplet(orig, degraded, restored, p_deg, p_rest, labels=("Originale", "Dégradée", "Restaurée"), zoom=None):
    cols = st.columns(3)
    imgs = [orig, degraded, restored]
    caps = [labels[0], f"{labels[1]} — {p_deg:.2f} dB", f"{labels[2]} — {p_rest:.2f} dB"]
    for c, im, cap in zip(cols, imgs, caps):
        c.image(im, caption=cap, width="stretch", clamp=True)
    if zoom is not None:
        y0, y1, x0, x1 = zoom
        zcols = st.columns(3)
        for c, im, lab in zip(zcols, imgs, labels):
            crop = Image.fromarray((np.clip(im[y0:y1, x0:x1], 0, 1) * 255).astype(np.uint8))
            c.image(crop.resize((crop.width * 4, crop.height * 4), Image.NEAREST),
                    caption=f"Zoom ×4 — {lab}", width="stretch")
    m1, m2, m3 = st.columns(3)
    m1.metric("PSNR dégradée", f"{p_deg:.2f} dB")
    m2.metric("PSNR restaurée", f"{p_rest:.2f} dB", delta=f"{p_rest - p_deg:+.2f} dB")
    return m3


def zoom_controls(key, H, W):
    if not st.checkbox("Afficher un zoom", value=True, key=f"zoom_{key}"):
        return None
    s = min(48, H // 3, W // 3)
    c1, c2 = st.columns(2)
    cx = c1.slider("Zoom — position horizontale", s // 2, W - s // 2, W // 2, key=f"zx_{key}")
    cy = c2.slider("Zoom — position verticale", s // 2, H - s // 2, H // 2, key=f"zy_{key}")
    return cy - s // 2, cy + s // 2, cx - s // 2, cx + s // 2


def progress_cb(bar, t0):
    def cb(k, K):
        bar.progress(k / K, text=f"Itération {k}/{K} — {time.time() - t0:.0f} s")
    return cb


# ============================================================
# EN-TETE
# ============================================================
st.title("🔬 Restauration d'images par Plug-and-Play")
st.markdown(
    "Projet semestriel du M1 Data Science & Vision par ordinateur (Université de Bordeaux). "
    "On résout trois **problèmes inverses** (retrouver une image $x$ à partir d'une observation "
    "dégradée $y = Ax + \\text{bruit}$) en insérant un **débruiteur profond** dans un algorithme "
    "d'optimisation : c'est le principe du *Plug-and-Play*."
)
st.caption("Débruiteur : GS-DRUNet pré-entraîné (Hurault, Leclaire & Papadakis, ICLR 2022) · bibliothèque deepinv")

# ============================================================
# SIDEBAR : IMAGE
# ============================================================
with st.sidebar:
    st.header("Image d'entrée")
    examples = sorted((ASSETS / "exemples").glob("*.png"))
    choice = st.radio("Source", ["Exemple", "Importer"], horizontal=True)
    if choice == "Importer":
        up = st.file_uploader("PNG / JPG", type=["png", "jpg", "jpeg"])
        source = up if up is not None else examples[0]
    else:
        ex = st.selectbox("Image d'exemple", examples, format_func=lambda p: p.stem.capitalize())
        source = ex
    max_side = st.select_slider("Taille max (px)", [128, 192, 256, 384, 512], value=256,
                                help="Le calcul est fait sur CPU en ligne : plus l'image est grande, plus c'est long.")
    seed = st.number_input("Graine aléatoire (bruit, masque)", 0, 9999, 0)
    st.caption(f"Calcul sur **{DEVICE.upper()}**")

img = load_image(source, max_side)
x_true = pil_to_tensor(img)
_, C, H, W = x_true.shape
st.sidebar.image(img, caption=f"{W}×{H} px", width="stretch")

denoiser = get_denoiser()

tab_blur, tab_inp, tab_mri, tab_res, tab_meth = st.tabs(
    ["🌫️ Défloutage", "🧩 Inpainting", "🧠 IRM", "📊 Résultats du projet", "📐 Méthode"]
)

# ============================================================
# 1. DEFLOUTAGE
# ============================================================
with tab_blur:
    st.subheader("Défloutage : flou gaussien + bruit")
    with st.form("form_blur"):
        c1, c2, c3 = st.columns(3)
        blur_sigma = c1.slider("Écart-type du flou", 0.5, 4.0, 2.0, 0.1)
        nu = c1.slider("Niveau de bruit ν", 0.0, 0.15, 0.05, 0.01)
        lamda = c2.slider("λ (poids du prior)", 0.01, 3.0, 0.6, 0.01,
                          help="↑ λ : le débruiteur agit davantage → image plus lisse")
        sig_fac = c2.slider("σ du débruiteur (× ν)", 0.2, 5.0, 1.5, 0.1,
                            help="Le meilleur réglage trouvé dans le projet : σ = 1.5 ν")
        K = c3.slider("Itérations max K", 5, 150, 30, 5)
        tau0 = c3.number_input("τ₀ (pas initial)", 0.1, 2.0, 0.99, 0.01)
        go = st.form_submit_button("▶ Lancer la restauration", type="primary")

    if go:
        torch.manual_seed(seed)
        physics, kernel = core.make_blur(25, blur_sigma, C, DEVICE)
        y = (physics(x_true) + nu * torch.randn_like(x_true)).clamp(0, 1)
        sigma = max(sig_fac * nu, 1e-3)
        bar, t0 = st.progress(0.0, text="Démarrage…"), time.time()
        r = core.pnp_deblur(y, physics, kernel, denoiser, lamda=lamda, sigma=sigma, tau0=tau0, K=K,
                            x_true=x_true, callback=progress_cb(bar, t0))
        bar.empty()
        st.session_state["blur"] = dict(y=core.to_numpy_rgb(y), x=core.to_numpy_rgb(r.x_hat),
                                       p_deg=core.psnr(x_true, y), p_rest=core.psnr(x_true, r.x_hat),
                                       res=r.residuals, psnrs=r.psnrs, n=r.n_iter,
                                       t=time.time() - t0, key=(str(source), max_side))

    out = st.session_state.get("blur")
    if out and out["key"] == (str(source), max_side):
        zoom = zoom_controls("blur", H, W)
        m3 = show_triplet(core.to_numpy_rgb(x_true), out["y"], out["x"], out["p_deg"], out["p_rest"], zoom=zoom)
        m3.metric("Itérations / temps", f"{out['n']} · {out['t']:.0f} s")
        convergence_chart(out["res"], out["psnrs"])
    else:
        st.info("Règle les paramètres puis lance la restauration.")

# ============================================================
# 2. INPAINTING
# ============================================================
with tab_inp:
    st.subheader("Inpainting : reconstruire des pixels manquants")
    with st.form("form_inp"):
        c1, c2, c3 = st.columns(3)
        keep = c1.slider("Pixels conservés", 0.1, 0.9, 0.5, 0.05, format="%.2f")
        nu_i = c1.slider("Bruit ν", 0.0, 0.1, 0.0, 0.01)
        s_main = c2.slider("σ principal (/255)", 1, 50, 10)
        s_warm = c2.slider("σ warm-up (/255)", 10, 100, 50)
        K_i = c3.slider("Itérations K", 10, 200, 40, 10)
        n_warm = c3.slider("Itérations de warm-up", 0, 30, 10)
        go_i = st.form_submit_button("▶ Lancer la restauration", type="primary")

    if go_i:
        torch.manual_seed(seed)
        physics, mask = core.make_inpainting(x_true.shape, keep, DEVICE, seed=seed)
        y = physics(x_true)
        if nu_i > 0:
            y = y + nu_i * torch.randn_like(y) * mask
        bar, t0 = st.progress(0.0, text="Démarrage…"), time.time()
        r = core.pnp_inpaint(y, mask, denoiser, sigma_main=s_main / 255, sigma_warmup=s_warm / 255,
                             K=K_i, n_warmup=n_warm, x_true=x_true, callback=progress_cb(bar, t0))
        bar.empty()
        st.session_state["inp"] = dict(y=core.to_numpy_rgb(y), x=core.to_numpy_rgb(r.x_hat),
                                      p_deg=core.psnr(x_true, y), p_rest=core.psnr(x_true, r.x_hat),
                                      res=r.residuals, psnrs=r.psnrs, n=r.n_iter,
                                      t=time.time() - t0, key=(str(source), max_side))

    out = st.session_state.get("inp")
    if out and out["key"] == (str(source), max_side):
        zoom = zoom_controls("inp", H, W)
        m3 = show_triplet(core.to_numpy_rgb(x_true), out["y"], out["x"], out["p_deg"], out["p_rest"], zoom=zoom)
        m3.metric("Itérations / temps", f"{out['n']} · {out['t']:.0f} s")
        convergence_chart(out["res"], out["psnrs"], "Résidu relatif ‖xₖ₊₁ − xₖ‖ / ‖x₀‖")
    else:
        st.info("Règle les paramètres puis lance la restauration.")

# ============================================================
# 3. IRM
# ============================================================
with tab_mri:
    st.subheader("IRM : reconstruction depuis un k-space sous-échantillonné")
    st.caption("L'image est convertie en niveaux de gris et redimensionnée en carré. "
               "L'observation affichée est la reconstruction naïve $A^* y$ (zéros dans les fréquences manquantes).")
    with st.form("form_mri"):
        c1, c2, c3 = st.columns(3)
        size_m = c1.select_slider("Taille", [128, 192, 256], value=min(256, max_side))
        keep_m = c1.slider("Fréquences conservées", 0.1, 0.9, 0.7, 0.05)
        s_main_m = c2.slider("σ principal (/255)", 1, 30, 5)
        s_warm_m = c2.slider("σ warm-up (/255)", 10, 100, 50)
        K_m = c3.slider("Itérations K", 10, 200, 40, 10)
        n_warm_m = c3.slider("Itérations de warm-up", 0, 30, 10)
        go_m = st.form_submit_button("▶ Lancer la reconstruction", type="primary")

    if go_m:
        gray = np.asarray(Image.open(source).convert("L").resize((size_m, size_m)), dtype=np.float32) / 255.0
        xt = core.to_mri_tensor(gray, DEVICE)
        physics, mask = core.make_mri(size_m, size_m, keep_m, DEVICE, seed=seed)
        y = physics(xt)
        obs = physics.A_adjoint(y)
        bar, t0 = st.progress(0.0, text="Démarrage…"), time.time()
        r = core.pnp_mri(y, physics, denoiser, K=K_m, sigma_main=s_main_m / 255, sigma_warmup=s_warm_m / 255,
                         n_warmup=n_warm_m, x_true=xt, callback=progress_cb(bar, t0))
        bar.empty()
        st.session_state["mri"] = dict(orig=core.mri_to_numpy(xt), y=core.mri_to_numpy(obs),
                                      x=core.mri_to_numpy(r.x_hat), mask=mask.cpu().numpy(),
                                      p_deg=core.psnr_mri(xt, obs), p_rest=core.psnr_mri(xt, r.x_hat),
                                      res=r.residuals, psnrs=r.psnrs, n=r.n_iter, t=time.time() - t0,
                                      key=str(source))

    out = st.session_state.get("mri")
    if out and out["key"] == str(source):
        S = out["orig"].shape[0]
        zoom = zoom_controls("mri", S, S)
        m3 = show_triplet(out["orig"], out["y"], out["x"], out["p_deg"], out["p_rest"],
                          labels=("Originale", "Observée (A*y)", "Reconstruite"), zoom=zoom)
        m3.metric("Itérations / temps", f"{out['n']} · {out['t']:.0f} s")
        with st.expander("Voir le masque d'échantillonnage du k-space"):
            st.image(np.fft.fftshift(out["mask"]), caption="Blanc = fréquence mesurée", width=256, clamp=True)
        convergence_chart(out["res"], out["psnrs"])
    else:
        st.info("Règle les paramètres puis lance la reconstruction.")

# ============================================================
# 4. RESULTATS DU NOTEBOOK
# ============================================================
with tab_res:
    st.subheader("Résultats obtenus dans le projet")
    st.markdown(
        "- **Défloutage** (flou σ = 2, bruit ν = 0.05) : meilleur réglage trouvé par recherche en grille "
        "**λ = 0.6, σ = 1.5 ν**, soit **26.27 dB**.\n"
        "- **Inpainting** avec seulement 20 % des pixels et du bruit (ν = 0.03) : **26.45 dB**.\n"
        "- **IRM** (70 % du k-space, 200 itérations) : de **24.9 → 41.6 dB** (femme), "
        "**23.9 → 39.7 dB** (papillon), **26.2 → 36.2 dB** (paysage)."
    )
    figs = [
        ("deblur_zoom", "Défloutage avec zoom"), ("deblur_convergence", "Défloutage : convergence"),
        ("deblur_lambda", "Influence de λ"), ("deblur_sigma", "Influence de σ"),
        ("deblur_grille", "Recherche en grille λ × σ"), ("inpainting_50", "Inpainting, 50 % des pixels"),
        ("inpainting_bruit", "Inpainting bruité, 20 % des pixels"), ("irm_toutes", "IRM sur 3 images"),
        ("irm_convergence", "IRM : convergence"),
    ]
    for name, cap in figs:
        p = ASSETS / "resultats" / f"{name}.jpg"
        if p.exists():
            st.markdown(f"**{cap}**")
            st.image(str(p), width="stretch")

# ============================================================
# 5. METHODE
# ============================================================
with tab_meth:
    st.subheader("Principe")
    st.markdown(
        r"""
On cherche $\hat x \in \arg\min_x \; F(x) = f(x) + \lambda\, g_\sigma(x)$, avec
$f(x) = \tfrac12\|Ax - y\|^2$ (attache aux données) et $g_\sigma$ un **régulariseur appris**.

Le **Gradient-Step Denoiser** s'écrit $D_\sigma = \mathrm{Id} - \nabla g_\sigma$ : c'est exactement
une descente de gradient sur un potentiel explicite $g_\sigma$. On peut donc l'insérer dans un
algorithme proximal **avec garantie de convergence**.

**Défloutage — GS-PnP avec backtracking**
$$
z_k = \lambda\tau\, D_\sigma(x_k) + (1-\lambda\tau)\,x_k, \qquad x_{k+1} = \mathrm{prox}_{\tau f}(z_k)
$$
Le prox du flou se calcule exactement en Fourier :
$\mathrm{prox}_{\tau f}(z) = \mathcal F^{-1}\!\Big(\frac{\tau \bar H Y + Z}{1 + \tau |H|^2}\Big)$.
Si $F$ ne décroît pas assez ($F(x_k) - F(x_{k+1}) < \frac{\gamma}{\tau}\|x_k - x_{k+1}\|^2$),
le pas est réduit : $\tau \leftarrow \eta\,\tau$.

**Inpainting** : $x_{k+1} = M \odot y + (1-M) \odot D_{\sigma_k}(x_k)$, avec un $\sigma_k$ élevé
pendant les premières itérations (*warm-up*) pour combler grossièrement les trous.

**IRM** : $A = M \mathcal F$ (transformée de Fourier masquée). On alterne débruitage des parties
réelle et imaginaire et $\mathrm{prox}_{\tau f}$ fourni par deepinv.

**Référence** : S. Hurault, A. Leclaire, N. Papadakis, *Gradient Step Denoiser for convergent
Plug-and-Play*, ICLR 2022.
"""
    )
