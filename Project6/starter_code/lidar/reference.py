"""
ME144/244
LIDAR Reference (ground truth) and Study Module  -  PROVIDED, DO NOT EDIT

A pixel of the focal plane array does not look along a single line: it sees a small patch of the
scene (its footprint). This module samples many sub-rays across each pixel's footprint, intersects
them with the TRUE surface, and reports the mean true surface height seen by each pixel. The
reconstructed height from lidarsim() can then be compared with it.

It also runs lidarsim() for several focal lengths at a fixed pixel pitch and plots the point clouds
and height maps side by side, to show the tradeoff between surface coverage and reconstruction
quality.
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from . import simulation as sim
from .surfaces import surfG, A_


def pixel_directions(sDist, nSub=1):
    """
    Unit viewing directions (from the lens center out into the scene) for sub-rays in every pixel

    Returns an array of shape (nPix * nPix, nSub * nSub, 3). With nSub = 1 these are the pixel
    center (chief) rays.
    """
    n, pitch = sim.nPix, sim.pixelPitch
    frac = (np.arange(nSub) + 0.5) / nSub - 0.5 if nSub > 1 else np.zeros(1)
    ox, oy = np.meshgrid(frac * pitch, frac * pitch)
    cols, rows = np.meshgrid(np.arange(n), np.arange(n))
    px = ((cols.ravel() + 0.5 - n / 2) * pitch)[:, None] + ox.ravel()[None, :]
    py = ((rows.ravel() + 0.5 - n / 2) * pitch)[:, None] + oy.ravel()[None, :]
    d = np.stack([-px, -py, -sDist * np.ones_like(px)], axis=-1)
    return d / np.linalg.norm(d, axis=-1)[..., None]


def true_mean_height(surface, sDist, nSub=5, nMarch=800, nBisect=40):
    """
    Mean true surface height over every pixel's footprint

    Parameters
    ----------
    surface : int
        Index into A_ array specifying which surface amplitude to use
    sDist : float
        Lens to focal plane array distance that was used (sets the pixel footprints)
    nSub : int
        Sub-rays per pixel side (nSub x nSub sub-rays per pixel)

    Returns
    -------
    zMean : ndarray, shape (nPix * nPix,)
        Mean surface height (m) of the points hit by the sub-rays of each pixel
    """
    A = A_[surface]
    d = pixel_directions(sDist, nSub).reshape(-1, 3)
    origin = sim.emitter

    def below(t):
        p = origin + t[:, None] * d
        return p[:, 2] <= surfG(p[:, 0], p[:, 1], A)

    # march along each sub-ray to the FIRST crossing, then bisect the bracket
    tMax = (origin[2] - (2 - A - 0.1)) / (-d[:, 2])
    lo = np.zeros(d.shape[0])
    hi = np.full(d.shape[0], np.nan)
    for k in range(1, nMarch + 1):
        t = tMax * k / nMarch
        found = np.isnan(hi) & below(t)
        hi[found] = t[found]
        lo[found] = (tMax * (k - 1) / nMarch)[found]
    hitMask = ~np.isnan(hi)
    hi[~hitMask] = 0.0
    for _ in range(nBisect):
        mid = (lo + hi) / 2
        b = below(mid)
        hi = np.where(b, mid, hi)
        lo = np.where(b, lo, mid)
    z = np.where(hitMask, origin[2] + hi * d[:, 2], np.nan)
    return np.nanmean(z.reshape(sim.nPix * sim.nPix, nSub * nSub), axis=1)


def reconstructed_height(surface):
    """Height (m) of the reconstructed point of every pixel of the last run (NaN = no return)"""
    dz = pixel_directions(sim.opticsUsed[surface]['sensorDist'])[:, 0, 2]
    return sim.emitter[2] + sim.rangeMap[surface] * dz


def _error_stats(recon, zMean):
    got = ~np.isnan(recon)
    err = recon[got] - zMean[got]
    return got.sum(), np.median(np.abs(err)), np.mean(np.abs(err)), np.mean(err)


def _heatmap(ax, field, ext, vmin, vmax):
    cmap = plt.get_cmap('viridis').copy()
    cmap.set_bad('lightgray')                       # pixels with no return
    n = sim.nPix
    # scene orientation: the lens inverts the image, so flip the pixel indices
    im = ax.imshow(field.reshape(n, n)[::-1, ::-1], origin='lower',
                   extent=[-ext, ext, -ext, ext], cmap=cmap, vmin=vmin, vmax=vmax)
    return im


def plot_height_maps(surface, nSub=5):
    """
    Heatmaps of the reconstructed height and the true mean height over each pixel's footprint for
    one surface, and the reconstruction error statistics. Run lidarsim(surface) first.
    """
    optics = sim.opticsUsed[surface]
    recon = reconstructed_height(surface)
    zMean = true_mean_height(surface, optics['sensorDist'], nSub)
    ext = optics['tanHalfFOV'] * sim.focusDist      # scene half-width at the focus plane (m)

    fig, axs = plt.subplots(1, 2, figsize=(11, 4.8), sharex=True, sharey=True)
    vmin, vmax = np.nanmin(zMean), np.nanmax(zMean)
    for ax, title, field in zip(axs, ["Reconstructed height", "True height: footprint mean"],
                                [recon, zMean]):
        im = _heatmap(ax, field, ext, vmin, vmax)
        ax.set_title(title)
        ax.set_xlabel('X at focus plane (m)')
    axs[0].set_ylabel('Y at focus plane (m)')
    fig.colorbar(im, ax=axs, label='height z (m)', shrink=0.9)
    fig.suptitle(f'Surface amplitude A = {A_[surface]}, focal length f = {optics["f"]} m')
    plt.show()

    nGot, med, mean, bias = _error_stats(recon, zMean)
    print(f"A = {A_[surface]}: pixels with a return = {nGot} / {sim.nPix ** 2}")
    print(f"  median |reconstruction - true mean height| = {med:.4f} m")
    print(f"  mean   |reconstruction - true mean height| = {mean:.4f} m")
    print(f"  mean signed error (bias)                   = {bias:+.4f} m")


def _draw_receiver(ax, sDist):
    """Draw the lens aperture (circle in the plane z = z0) and the sensor (square, to scale)"""
    z0 = sim.emitter[2]
    th = np.linspace(0, 2 * np.pi, 100)
    ax.plot(sim.apertureRadius * np.cos(th), sim.apertureRadius * np.sin(th),
            z0 * np.ones_like(th), color='tab:red', lw=2)
    w = sim.nPix * sim.pixelPitch / 2
    sq = [(-w, -w, z0 + sDist), (w, -w, z0 + sDist), (w, w, z0 + sDist), (-w, w, z0 + sDist)]
    ax.add_collection3d(Poly3DCollection([sq], facecolor='tab:blue', edgecolor='tab:blue',
                                         alpha=0.35))


def focal_length_study(surface, focal_lengths, nSub=5):
    """
    Run lidarsim() for each focal length (same pixel pitch and pixel count) and plot, for each one,
    the 3D point cloud (with the lens aperture and the sensor) next to the reconstructed height map.

    """
    runs = {}
    # run in reverse so the stored results of the first focal length are the ones left behind
    for fl in list(focal_lengths)[::-1]:
        sim.lidarsim(surface, plot=False, focalLength=fl)
        optics = dict(sim.opticsUsed[surface])
        runs[fl] = dict(cloud=sim.pointCloud[surface].copy(),
                        recon=reconstructed_height(surface),
                        zMean=true_mean_height(surface, optics['sensorDist'], nSub),
                        optics=optics)

    # guard: every focal length must really have produced different optics
    dists = [runs[fl]['optics']['sensorDist'] for fl in focal_lengths]
    if len(set(np.round(dists, 9))) != len(dists):
        raise RuntimeError(f"The runs used the same sensor distance {dists}; the focal length was "
                           "not applied. Restart the kernel so the updated modules are loaded.")

    A = A_[surface]
    vmin, vmax = 2 - A, 2 + A
    cmap = plt.get_cmap('viridis')
    n = len(focal_lengths)

    # one row per focal length: 3D point cloud | reconstructed height | true mean height, one colorbar
    fig = plt.figure(figsize=(16.5, 5.6 * n))
    gs = fig.add_gridspec(n, 4, width_ratios=[1.3, 1, 1, 0.04], wspace=0.35, hspace=0.3)

    # true surface over the 1 m x 1 m patch
    x = np.linspace(-0.5, 0.5, 41)
    X, Y = np.meshgrid(x, x)
    Z = surfG(X, Y, A)

    for k, fl in enumerate(focal_lengths):
        run = runs[fl]
        cloud, optics = run['cloud'], run['optics']
        ext = optics['tanHalfFOV'] * sim.focusDist      # scene half-width at the focus plane (m)
        nGot, med, mean, bias = _error_stats(run['recon'], run['zMean'])
        print(f"  f = {fl} m: sensor distance {optics['sensorDist']:.4f} m, "
              f"FOV +/-{ext:.3f} m at the focus plane, pixel footprint "
              f"{100 * sim.pixelPitch / optics['sensorDist']:.1f} cm, "
              f"{nGot} / {sim.nPix ** 2} pixels with a return")

        ax3 = fig.add_subplot(gs[k, 0], projection='3d')
        ax3.plot_surface(X, Y, Z, color='gray', alpha=0.2, linewidth=0, rcount=30, ccount=30)
        ax3.scatter3D(cloud[:, 0], cloud[:, 1], cloud[:, 2], c=cloud[:, 2], cmap=cmap,
                      vmin=vmin, vmax=vmax, s=10, depthshade=False)
        _draw_receiver(ax3, optics['sensorDist'])
        ax3.set_xlim(-0.75, 0.75)
        ax3.set_ylim(-0.75, 0.75)
        ax3.set_zlim(1.2, 3.4)
        ax3.set_xlabel('X (m)')
        ax3.set_ylabel('Y (m)')
        ax3.set_zlabel('Z (m)', labelpad=2)
        ax3.set_box_aspect((1, 1, 0.9))
        ax3.view_init(elev=22, azim=-60)
        ax3.set_title(f'f = {fl} m: point cloud')
        if k == 0:
            ax3.legend(handles=[Line2D([0], [0], color='tab:red', lw=2, label='Lens aperture'),
                                Patch(facecolor='tab:blue', alpha=0.35, label='Sensor (to scale)'),
                                Patch(facecolor='gray', alpha=0.3, label='True surface')],
                       loc='upper left', fontsize=8)

        # the axes follow this focal length's own field of view
        axh = fig.add_subplot(gs[k, 1])
        axt = fig.add_subplot(gs[k, 2], sharex=axh, sharey=axh)
        for ax, field in ((axh, run['recon']), (axt, run['zMean'])):
            im = _heatmap(ax, field, ext, vmin, vmax)
            ax.set_xlim(-ext, ext)
            ax.set_ylim(-ext, ext)
            ax.plot([-0.5, 0.5, 0.5, -0.5, -0.5], [-0.5, -0.5, 0.5, 0.5, -0.5], 'k--', lw=1)
            ax.set_xlabel('X at focus plane (m)')
        axh.set_ylabel('Y at focus plane (m)')
        axh.set_title(f'f = {fl} m: reconstruction, FOV $\\pm${ext:.2f} m, '
                      f'{100 * nGot / sim.nPix ** 2:.0f}% pixels\n'
                      f'median |err| {1000 * med:.1f} mm, mean |err| {1000 * mean:.1f} mm',
                      fontsize=10)
        axt.set_title(f'f = {fl} m: ground truth\n(mean surface height of each pixel footprint)',
                      fontsize=10)
        if k == 0:
            axh.plot([], [], 'k--', lw=1, label='1 m x 1 m patch')
            axh.legend(loc='lower right', fontsize=8)
    fig.colorbar(im, cax=fig.add_subplot(gs[:, 3]), label='height z (m)')
    fig.suptitle(f'Surface amplitude A = {A}')
    plt.show()
