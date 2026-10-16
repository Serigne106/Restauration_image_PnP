"""
Algorithmes Plug-and-Play (PnP) avec le débruiteur GS-DRUNet (deepinv).

Reprise du notebook PROJET_IMAGE_4.ipynb, réorganisée pour une app :
- plus de variables globales (physics, x_true, denoiser) : tout est passé en argument ;
- le noyau de flou est transformé en FFT une seule fois (et non à chaque itération) ;
- un `callback(k, K)` optionnel permet d'afficher une barre de progression.

Trois problèmes inverses :
1. Défloutage (flou gaussien + bruit)  -> PnP avec backtracking sur tau (GS-PnP)
2. Inpainting (pixels manquants)        -> PnP avec warm-up sur sigma
3. IRM (k-space sous-échantillonné)     -> PnP avec prox de deepinv
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Optional

import numpy as np
import torch
import deepinv as dinv

Callback = Optional[Callable[[int, int], None]]


# ============================================================
# MODELE & METRIQUES
# ============================================================
def load_denoiser(device: str = "cpu", pretrained: str | None = "download"):
    """GS-DRUNet pré-entraîné (Hurault et al., 2022). Les poids (~70 Mo) sont
    téléchargés depuis Hugging Face au premier appel puis mis en cache."""
    model = dinv.models.GSDRUNet(device=device, pretrained=pretrained)
    model.eval()
    return model


def psnr(x_true: torch.Tensor, x_hat: torch.Tensor) -> float:
    mse = torch.mean((x_true - x_hat) ** 2)
    if mse == 0:
        return float("inf")
    return (10 * torch.log10(1.0 / mse)).item()


def psnr_mri(x_true: torch.Tensor, x_hat: torch.Tensor) -> float:
    """PSNR calculé sur le module |re + i·im|."""
    return psnr(mri_magnitude(x_true), mri_magnitude(x_hat))


def mri_magnitude(x: torch.Tensor) -> torch.Tensor:
    return torch.sqrt(x[:, 0] ** 2 + x[:, 1] ** 2)


@dataclass
class Result:
    x_hat: torch.Tensor
    residuals: list = field(default_factory=list)
    psnrs: list = field(default_factory=list)
    n_iter: int = 0


# ============================================================
# 1. DEFLOUTAGE
# ============================================================
def gaussian_kernel(size: int, sigma: float, channels: int = 3) -> torch.Tensor:
    ax = torch.arange(size).float() - size // 2
    xx, yy = torch.meshgrid(ax, ax, indexing="ij")
    kernel = torch.exp(-(xx**2 + yy**2) / (2 * sigma**2))
    kernel = kernel / kernel.sum()
    return kernel.view(1, 1, size, size).repeat(1, channels, 1, 1)


def kernel_fft(kernel: torch.Tensor, H: int, W: int) -> torch.Tensor:
    """Fonction de transfert du flou (convolution circulaire), shape (1, C, H, W).

    NB : le décalage est -(kh // 2). Dans le notebook, `-kh//2` vaut -13 pour
    kh = 25 en Python (division entière vers -inf), ce qui décale le noyau d'un
    pixel par rapport à `dinv.physics.Blur`.
    """
    _, C, kh, kw = kernel.shape
    pad = torch.zeros((1, C, H, W), device=kernel.device)
    pad[..., :kh, :kw] = kernel
    pad = torch.roll(pad, shifts=(-(kh // 2), -(kw // 2)), dims=(2, 3))
    return torch.fft.fft2(pad)


def make_blur(kernel_size: int, blur_sigma: float, channels: int, device: str):
    kernel = gaussian_kernel(kernel_size, blur_sigma, channels).to(device)
    physics = dinv.physics.Blur(filter=kernel, padding="circular", device=device)
    return physics, kernel


def prox_blur(z, y, tau, H_fft):
    """prox_{tau f}(z) avec f(x) = 1/2 ||h * x - y||² (solution exacte en Fourier)."""
    H_conj = torch.conj(H_fft)
    X = (tau * H_conj * torch.fft.fft2(y) + torch.fft.fft2(z)) / (1 + tau * (H_conj * H_fft))
    return torch.fft.ifft2(X).real


def pnp_deblur(
    y, physics, kernel, denoiser, *,
    lamda=0.6, sigma=0.15, epsilon=1e-6, tau0=0.99, K=100, eta=0.95, gamma=0.01,
    x_true=None, callback: Callback = None,
) -> Result:
    """GS-PnP avec backtracking (version `PnP_with_residus` du notebook)."""
    H_fft = kernel_fft(kernel, y.shape[2], y.shape[3])

    def energy(x):
        # F(x) = 1/2 ||Ax - y||² + λ g_σ(x)
        return 0.5 * torch.norm(physics(x) - y) ** 2 + lamda * denoiser.potential(x, sigma).sum()

    def D(x):
        with torch.no_grad():
            return denoiser(x, sigma)

    tau = tau0 / eta
    xk = prox_blur(y.clone(), y, tau, H_fft)
    F0 = energy(xk).item()
    Fk = F0
    delta = float("inf")
    res = Result(x_hat=xk)
    k = 0

    while k < K and delta > epsilon:
        zk = lamda * tau * D(xk) + (1 - lamda * tau) * xk
        xk1 = prox_blur(zk, y, tau, H_fft)

        Fk1 = energy(xk1).item()
        diff = Fk - Fk1
        step = torch.norm(xk1 - xk).item()
        res.residuals.append(step)
        if x_true is not None:
            res.psnrs.append(psnr(x_true, xk1.clamp(0, 1)))

        if diff < (gamma / tau) * step**2:
            tau = eta * tau  # backtracking : on réduit le pas
        else:
            delta = diff / (F0 + 1e-8)

        xk, Fk = xk1, Fk1
        k += 1
        if callback:
            callback(k, K)

    zk = lamda * tau * D(xk) + (1 - lamda * tau) * xk
    res.x_hat = prox_blur(zk, y, tau, H_fft).clamp(0, 1)
    res.n_iter = k
    if x_true is not None:
        res.psnrs.append(psnr(x_true, res.x_hat))
    return res


# ============================================================
# 2. INPAINTING
# ============================================================
def make_inpainting(shape, keep_ratio: float, device: str, seed: int = 0):
    """Masque aléatoire : chaque pixel est conservé avec probabilité keep_ratio."""
    g = torch.Generator().manual_seed(seed)
    _, C, H, W = shape
    mask = torch.bernoulli(keep_ratio * torch.ones(1, 1, H, W), generator=g)
    mask = mask.expand(1, C, H, W).contiguous().to(device)
    physics = dinv.physics.Inpainting(mask=mask[0], img_size=(C, H, W), device=device)
    return physics, mask


def pnp_inpaint(
    y, mask, denoiser, *,
    sigma_main=10 / 255, sigma_warmup=50 / 255, K=100, n_warmup=10,
    x_true=None, callback: Callback = None,
) -> Result:
    """PnP inpainting : x_{k+1} = Prox(D_σ(x_k)), avec σ élevé pendant le warm-up."""
    xk = torch.where(mask.bool(), y, torch.full_like(y, 0.5))
    x0_norm = torch.norm(xk).item() + 1e-8
    res = Result(x_hat=xk)

    for k in range(K):
        sigma_k = sigma_warmup if k < n_warmup else sigma_main
        with torch.no_grad():
            zk = denoiser(xk, sigma_k)
        xk1 = (zk + mask * (y - zk)).clamp(0, 1)  # on réimpose les pixels connus

        res.residuals.append(torch.norm(xk1 - xk).item() / x0_norm)
        if x_true is not None:
            res.psnrs.append(psnr(x_true, xk1))
        xk = xk1
        if callback:
            callback(k + 1, K)

    res.x_hat, res.n_iter = xk, K
    return res


# ============================================================
# 3. IRM
# ============================================================
def to_mri_tensor(gray: np.ndarray, device: str) -> torch.Tensor:
    """Image (H, W) en [0, 1] -> tensor complexe (1, 2, H, W) : [réel, imaginaire=0]."""
    re = torch.tensor(gray, dtype=torch.float32, device=device)
    return torch.stack([re, torch.zeros_like(re)], dim=0).unsqueeze(0)


def make_mri(H: int, W: int, keep_ratio: float, device: str, seed: int = 42):
    g = torch.Generator().manual_seed(seed)
    mask = torch.bernoulli(keep_ratio * torch.ones(H, W), generator=g).to(device)
    physics = dinv.physics.MRI(mask=mask, img_size=(H, W), device=device)
    return physics, mask


def pnp_mri(
    y, physics, denoiser, *,
    K=200, sigma_main=5 / 255, sigma_warmup=50 / 255, n_warmup=10, tau=1.0,
    x_true=None, callback: Callback = None,
) -> Result:
    def D(x, s):
        # Le débruiteur attend 3 canaux : on débruite séparément partie réelle et imaginaire.
        re = x[:, 0:1].repeat(1, 3, 1, 1)
        im = x[:, 1:2].repeat(1, 3, 1, 1)
        return torch.cat([denoiser(re, s)[:, 0:1], denoiser(im, s)[:, 0:1]], dim=1)

    xk = physics.A_adjoint(y).detach()
    res = Result(x_hat=xk)
    for k in range(K):
        s = sigma_warmup if k < n_warmup else sigma_main
        with torch.no_grad():
            xk1 = physics.prox_l2(D(xk, s), y, 1 / tau)
        res.residuals.append(torch.norm(xk1 - xk).item())
        if x_true is not None:
            res.psnrs.append(psnr_mri(x_true, xk1))
        xk = xk1.detach()
        if callback:
            callback(k + 1, K)

    res.x_hat, res.n_iter = xk, K
    return res


# ============================================================
# CONVERSIONS
# ============================================================
def to_numpy_rgb(x: torch.Tensor) -> np.ndarray:
    return x[0].detach().clamp(0, 1).permute(1, 2, 0).cpu().numpy()


def mri_to_numpy(x: torch.Tensor) -> np.ndarray:
    mag = mri_magnitude(x)[0].detach().cpu().numpy()
    return (mag - mag.min()) / (mag.max() - mag.min() + 1e-8)
