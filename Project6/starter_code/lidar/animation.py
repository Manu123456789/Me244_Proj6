"""
ME144/244
Interactive Flash LiDAR Ray Propagation & Sensor Illumination Animation
"""

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.animation import FuncAnimation
from matplotlib.patches import Circle, Rectangle
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from IPython.display import HTML, display

from . import simulation as sim
from .surfaces import surfG, gradG, A_

mpl.rcParams['animation.embed_limit'] = 50.0  # Allow full HTML5/JS animation embedding (MB)


def create_lidar_ray_animation(
    surface_idx=2,
    focal_length=0.1,
    diffuse_fraction=1.0,
    n_rays=50000,
    n_show_3d=280,
    stride=12,
    num_frames=72,
    seed=42,
):
    """
    Simulate and animate a flash LiDAR pulse in 3 synchronized panels:
      1. 3D ray trajectories (emission -> surface reflection/multi-bounce -> aperture -> thin-lens focusing)
      2. Returning ray crossings at the lens plane (z = z0) vs. the aperture circle
      3. Live 32x32 focal plane array pixel illumination & out-of-focus spill (~onArray)
    """
    rng = np.random.default_rng(seed)
    A = A_[surface_idx]
    c, z0, dt = sim.c, sim.z0, sim.dt
    f = focal_length
    focusDist = sim.focusDist
    sensorDist = f * focusDist / (focusDist - f)
    pixelPitch, nPix = sim.pixelPitch, sim.nPix
    apertureRadius = sim.apertureRadius
    maxBounces, domainMax = sim.maxBounces, sim.domainMax
    emitter = sim.emitter.copy()
    tanHalfFOV = nPix * pixelPitch / (2 * sensorDist)

    # 1. Emit rays over the camera FOV
    tx = (2 * rng.random(n_rays) - 1) * tanHalfFOV
    ty = (2 * rng.random(n_rays) - 1) * tanHalfFOV
    d0 = np.column_stack([tx, ty, -np.ones(n_rays)])
    d0 /= np.linalg.norm(d0, axis=1, keepdims=True)

    r = np.tile(emitter, (n_rays, 1))
    v = c * d0

    active = np.ones(n_rays, dtype=bool)
    rtrn = np.zeros(n_rays, dtype=bool)
    crossed_z0 = np.zeros(n_rays, dtype=bool)
    rfln = np.zeros(n_rays, dtype=int)
    tj = np.full(n_rays, np.inf)       # time ray crossed z = z0
    rj = np.full((n_rays, 3), np.nan)  # position where ray crossed z = z0
    uj = np.zeros((n_rays, 3))         # unit velocity when crossing z = z0

    # Record 3D trajectory snapshots for a candidate pool of rays
    n_pool = min(n_rays, 10000)
    pos_hist = np.full((num_frames, n_pool, 3), np.nan, dtype=np.float32)
    rfln_hist = np.zeros((num_frames, n_pool), dtype=np.int8)
    active_hist = np.zeros((num_frames, n_pool), dtype=bool)
    frame_times = np.zeros(num_frames)

    pos_hist[0] = r[:n_pool]
    active_hist[0] = True

    time = 0.0
    max_steps = (num_frames - 1) * stride
    for step in range(1, max_steps + 1):
        if np.any(active):
            idx = np.where((r[:, 2] <= surfG(r[:, 0], r[:, 1], A)) & active)[0]
            if idx.size != 0:
                gx, gy = gradG(r[idx, 0], r[idx, 1], A)
                n = np.column_stack([gx, gy, -np.ones(len(idx))])
                n /= np.linalg.norm(n, axis=1, keepdims=True)

                vdotn = np.sum(v[idx] * n, axis=1)
                entering = vdotn > 0
                idx, n, vdotn = idx[entering], n[entering], vdotn[entering]

                spent = rfln[idx] >= maxBounces
                active[idx[spent]] = False
                idx, n, vdotn = idx[~spent], n[~spent], vdotn[~spent]

                if idx.size != 0:
                    rfln[idx] += 1
                    vSpec = v[idx] - 2 * vdotn[:, np.newaxis] * n

                    m = -n
                    helper = np.where(
                        np.abs(m[:, :1]) < 0.9,
                        np.array([[1.0, 0.0, 0.0]]),
                        np.array([[0.0, 1.0, 0.0]]),
                    )
                    t1 = np.cross(m, helper)
                    t1 /= np.linalg.norm(t1, axis=1, keepdims=True)
                    t2 = np.cross(m, t1)
                    u1 = rng.random((len(idx), 1))
                    phi = 2 * np.pi * rng.random((len(idx), 1))
                    vDiff = c * (
                        np.sqrt(u1) * np.cos(phi) * t1
                        + np.sqrt(u1) * np.sin(phi) * t2
                        + np.sqrt(1 - u1) * m
                    )
                    isDiff = rng.random((len(idx), 1)) < diffuse_fraction
                    v[idx] = np.where(isDiff, vDiff, vSpec)

            time += dt
            r[active] += v[active] * dt

            # Check crossing of lens plane z = z0
            idx_up = np.where(active & (r[:, 2] > z0))[0]
            if idx_up.size != 0:
                crossed_z0[idx_up] = True
                dt_over = (r[idx_up, 2] - z0) / v[idx_up, 2]
                tj[idx_up] = time - dt_over
                rj[idx_up] = r[idx_up] - dt_over[:, np.newaxis] * v[idx_up]
                uj[idx_up] = v[idx_up] / c

                det = np.sum(rj[idx_up, :2] ** 2, axis=1) < apertureRadius ** 2
                rtrn[idx_up[det]] = True
                active[idx_up] = False

            outside = (np.abs(r[:, 0]) > domainMax) | (np.abs(r[:, 1]) > domainMax)
            active[outside] = False
        else:
            time += dt

        if step % stride == 0:
            f_idx = step // stride
            frame_times[f_idx] = time
            pos_hist[f_idx] = r[:n_pool]
            rfln_hist[f_idx] = rfln[:n_pool]
            active_hist[f_idx] = active[:n_pool]

    # 2. Thin-lens refraction & sensor array mapping for rays that entered the aperture
    rtrn_idx = np.where(rtrn)[0]
    h = rj[rtrn_idx, :2]
    u = uj[rtrn_idx, :]
    slope = u[:, :2] / u[:, 2:3] - h / f
    pImg = h + sensorDist * slope

    col = np.floor(pImg[:, 0] / pixelPitch + nPix / 2).astype(int)
    row = np.floor(pImg[:, 1] / pixelPitch + nPix / 2).astype(int)
    onArray = (col >= 0) & (col < nPix) & (row >= 0) & (row < nPix)
    pix = row[onArray] * nPix + col[onArray]

    path = np.sqrt(np.sum((pImg - h) ** 2, axis=1) + sensorDist ** 2)
    tof = tj[rtrn_idx] + (path - sensorDist - np.sum(h ** 2, axis=1) / (2 * f)) / c

    # Map back to full n_rays indexing for stratified sampling
    on_array_full = np.zeros(n_rays, dtype=bool)
    on_array_full[rtrn_idx[onArray]] = True
    pImg_full = np.full((n_rays, 2), np.nan)
    pImg_full[rtrn_idx] = pImg

    # Chief rays for reconstructing height per pixel
    cols_g, rows_g = np.meshgrid(np.arange(nPix), np.arange(nPix))
    xc = (cols_g.ravel() + 0.5 - nPix / 2) * pixelPitch
    yc = (rows_g.ravel() + 0.5 - nPix / 2) * pixelPitch
    dPix = np.column_stack([-xc, -yc, -sensorDist * np.ones(nPix * nPix)])
    dPix /= np.linalg.norm(dPix, axis=1, keepdims=True)
    dz_pix = dPix[:, 2]

    # Print diagnostic breakdown of ray fates and pixel coverage
    tPix_final = np.full(nPix * nPix, np.inf)
    np.minimum.at(tPix_final, pix, tof[onArray])
    n_hit_pix = int(np.sum(np.isfinite(tPix_final)))
    n_crossed = int(np.sum(crossed_z0))
    n_ap = int(np.sum(rtrn))
    n_on_sensor = int(np.sum(onArray))
    n_multi = int(np.sum(rfln >= 2))

    print("=" * 78)
    print(f"RAY & PIXEL ILLUMINATION DIAGNOSTICS (A = {A}, f = {f} m, diffuse = {diffuse_fraction})")
    print("=" * 78)
    print(f"  1. Total rays emitted            : {n_rays:,d}")
    print(f"  2. Rays with multiple bounces    : {n_multi:,d} ({100*n_multi/n_rays:.1f}%)")
    print(f"  3. Reached lens plane (z = z0)   : {n_crossed:,d} ({100*n_crossed/n_rays:.1f}%)")
    print(f"     -> MISSED aperture (r > {apertureRadius}m) : {n_crossed - n_ap:,d} ({100*(n_crossed - n_ap)/max(n_crossed,1):.1f}% of returning rays lost!)")
    print(f"     -> ENTERED aperture (r <= {apertureRadius}m): {n_ap:,d} ({100*n_ap/n_rays:.2f}% of emitted rays)")
    print(f"  4. After thin-lens refraction    :")
    print(f"     -> Missed sensor array (~onArray): {n_ap - n_on_sensor:,d} ({100*(n_ap - n_on_sensor)/max(n_ap,1):.1f}% of aperture rays lost to defocus/angle)")
    print(f"     -> Landed on 32x32 sensor array  : {n_on_sensor:,d} rays across {nPix*nPix} pixels (avg {n_on_sensor/(nPix*nPix):.2f} rays/pixel)")
    print(f"  5. Final illuminated pixels      : {n_hit_pix} / {nPix*nPix} ({100*n_hit_pix/(nPix*nPix):.1f}% illuminated, {100*(1 - n_hit_pix/(nPix*nPix)):.1f}% dark/missing)")
    print("=" * 78)

    # 3. Select stratified sample of rays from n_pool for 3D visualization
    pool_mask = np.arange(n_pool)
    cat_hit_sensor = pool_mask[rtrn[:n_pool] & on_array_full[:n_pool]]
    cat_miss_sensor = pool_mask[rtrn[:n_pool] & ~on_array_full[:n_pool]]
    cat_multibounce = pool_mask[(~rtrn[:n_pool]) & (rfln[:n_pool] >= 2)]
    cat_miss_ap = pool_mask[(~rtrn[:n_pool]) & (rfln[:n_pool] == 1)]

    def sample_up_to(arr, k):
        return rng.choice(arr, size=min(len(arr), k), replace=False) if len(arr) > 0 else np.array([], dtype=int)

    s_hit = sample_up_to(cat_hit_sensor, int(0.40 * n_show_3d))
    s_off = sample_up_to(cat_miss_sensor, int(0.15 * n_show_3d))
    s_mul = sample_up_to(cat_multibounce, int(0.15 * n_show_3d))
    s_mis = sample_up_to(cat_miss_ap, n_show_3d - len(s_hit) - len(s_off) - len(s_mul))
    sample_ids = np.concatenate([s_hit, s_off, s_mul, s_mis])

    # Continue aperture-entering rays from lens (z0) to sensor (z0 + sensorDist) at speed c
    for i in np.concatenate([s_hit, s_off]):
        t_cross = tj[i]
        h_pt = np.array([rj[i, 0], rj[i, 1], z0], dtype=np.float32)
        s_pt = np.array([pImg_full[i, 0], pImg_full[i, 1], z0 + sensorDist], dtype=np.float32)
        ray_path_len = float(np.linalg.norm(s_pt - h_pt))
        ray_transit_time = max(ray_path_len / c, dt)  # physical transit time at speed c
        for f_idx in range(num_frames):
            if frame_times[f_idx] >= t_cross:
                alpha = np.clip((frame_times[f_idx] - t_cross) / ray_transit_time, 0.0, 1.0)
                pos_hist[f_idx, i] = (1.0 - alpha) * h_pt + alpha * s_pt
                active_hist[f_idx, i] = alpha < 1.0

    # Clamp missed-aperture rays at z = z0 when they cross
    for i in np.concatenate([s_mul, s_mis]):
        if np.isfinite(tj[i]):
            for f_idx in range(num_frames):
                if frame_times[f_idx] >= tj[i]:
                    pos_hist[f_idx, i] = rj[i].astype(np.float32)
                    active_hist[f_idx, i] = False

    # 4. Build the 3-panel figure
    fig = plt.figure(figsize=(17, 5.8))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.35, 1.0, 1.05], wspace=0.28)

    # --- Panel 1: 3D Ray Propagation ---
    ax3d = fig.add_subplot(gs[0, 0], projection='3d')
    xg = np.linspace(-0.55, 0.55, 45)
    Xg, Yg = np.meshgrid(xg, xg)
    Zg = surfG(Xg, Yg, A)
    ax3d.plot_surface(Xg, Yg, Zg, cmap=cm.plasma, alpha=0.28, linewidth=0, rcount=35, ccount=35)

    # Draw lens aperture (red circle at z0) and sensor array (blue square at z0 + sensorDist)
    th = np.linspace(0, 2 * np.pi, 100)
    ax3d.plot(apertureRadius * np.cos(th), apertureRadius * np.sin(th), z0 * np.ones_like(th),
              color='tab:red', lw=2.2, label='Lens aperture (z = z0)')
    w = nPix * pixelPitch / 2
    sq = [(-w, -w, z0 + sensorDist), (w, -w, z0 + sensorDist),
          (w, w, z0 + sensorDist), (-w, w, z0 + sensorDist)]
    ax3d.add_collection3d(Poly3DCollection([sq], facecolor='tab:blue', edgecolor='tab:blue', alpha=0.45))

    # 3D Line artists for ray heads and short trailing tails by category
    styles = {
        'out':  ('#0ea5e9', 'Outgoing pulse (to surface)'),
        'hit':  ('#16a34a', 'Hits aperture & illuminates pixel'),
        'off':  ('#f59e0b', 'Hits aperture, misses sensor (~onArray)'),
        'mul':  ('#d946ef', 'Multiple surface bounces (>=2)'),
        'miss': ('#ef4444', 'Reflected, misses lens aperture'),
    }
    heads_3d, tails_3d = {}, {}
    for key, (col_hex, lbl) in styles.items():
        tails_3d[key], = ax3d.plot([], [], [], color=col_hex, lw=1.1, alpha=0.65)
        heads_3d[key], = ax3d.plot([], [], [], linestyle='None', marker='o',
                                   markersize=3.5, color=col_hex, label=lbl)

    ax3d.set_xlim(-0.75, 0.75)
    ax3d.set_ylim(-0.75, 0.75)
    ax3d.set_zlim(1.25, z0 + sensorDist + 0.15)
    ax3d.set_xlabel('X (m)')
    ax3d.set_ylabel('Y (m)')
    ax3d.set_zlabel('Z (m)')
    ax3d.set_box_aspect((1, 1, 0.95))
    ax3d.view_init(elev=20, azim=-58)
    ax3d.legend(loc='upper left', fontsize=7.5, framealpha=0.9)
    title_3d = ax3d.set_title('1. 3D Ray Propagation (t = 0.0 ns)', fontsize=10.5, fontweight='bold')

    # --- Panel 2: Lens Plane (z = z0) Crossings vs. Aperture ---
    ax_lens = fig.add_subplot(gs[0, 1])
    ax_lens.set_facecolor('#f8fafc')
    crossed_all_idx = np.where(crossed_z0)[0]
    if len(crossed_all_idx) > 4000:
        miss_all = np.where(crossed_z0 & ~rtrn)[0]
        keep_miss = rng.choice(miss_all, size=min(len(miss_all), 3000), replace=False)
        lens_plot_miss = keep_miss
    else:
        lens_plot_miss = np.where(crossed_z0 & ~rtrn)[0]
    lens_plot_hit = rtrn_idx[onArray]
    lens_plot_off = rtrn_idx[~onArray]

    pts_miss_lens, = ax_lens.plot([], [], '.', color='#ef4444', markersize=2.5, alpha=0.35,
                                  label='Missed aperture (lost)')
    pts_off_lens, = ax_lens.plot([], [], '.', color='#f59e0b', markersize=4.0, alpha=0.8,
                                 label='In aperture -> missed sensor')
    pts_hit_lens, = ax_lens.plot([], [], '.', color='#16a34a', markersize=4.0, alpha=0.8,
                                 label='In aperture -> hit sensor pixel')
    ap_circle = Circle((0, 0), apertureRadius, edgecolor='tab:red', facecolor='none', lw=2.2,
                       label=f'Lens aperture (r = {apertureRadius} m)')
    ax_lens.add_patch(ap_circle)
    ax_lens.set_xlim(-1.0, 1.0)
    ax_lens.set_ylim(-1.0, 1.0)
    ax_lens.set_aspect('equal')
    ax_lens.set_xlabel('X at lens plane z = z0 (m)')
    ax_lens.set_ylabel('Y at lens plane z = z0 (m)')
    ax_lens.legend(loc='upper right', fontsize=7.5, framealpha=0.9)
    title_lens = ax_lens.set_title('2. Returning Rays at Lens Plane (z = z0)', fontsize=10.5, fontweight='bold')

    # --- Panel 3: Live 32x32 Sensor Array Illumination & Defocus Spill ---
    ax_sens = fig.add_subplot(gs[0, 2])
    ax_sens.set_facecolor('#1e293b')
    ext = tanHalfFOV * focusDist  # FOV half-width projected to focus plane (m)
    vmin, vmax = 2.0 - A, 2.0 + A
    cmap_sens = plt.get_cmap('viridis').copy()
    cmap_sens.set_bad('#334155')  # Dark slate for un-illuminated pixels

    empty_grid = np.full((nPix, nPix), np.nan)
    im_sens = ax_sens.imshow(empty_grid, origin='lower', extent=[-ext, ext, -ext, ext],
                             cmap=cmap_sens, vmin=vmin, vmax=vmax, interpolation='nearest')
    fig.colorbar(im_sens, ax=ax_sens, label='Reconstructed height z (m)', fraction=0.046, pad=0.04)

    sens_border = Rectangle((-ext, -ext), 2 * ext, 2 * ext, edgecolor='#38bdf8',
                            facecolor='none', lw=2.0, linestyle='--', label='32x32 Sensor array boundary')
    ax_sens.add_patch(sens_border)
    pImg_scene_off = -pImg[~onArray] * (focusDist / sensorDist)
    tof_off = tof[~onArray]
    pts_spill, = ax_sens.plot([], [], 'o', color='#f59e0b', markersize=3.0, alpha=0.75,
                              label='Refracted outside sensor (~onArray)')

    margin = 1.28 * ext
    ax_sens.set_xlim(-margin, margin)
    ax_sens.set_ylim(-margin, margin)
    ax_sens.set_aspect('equal')
    ax_sens.set_xlabel('Scene X at focus plane (m)')
    ax_sens.set_ylabel('Scene Y at focus plane (m)')
    ax_sens.legend(loc='lower right', fontsize=7.5, framealpha=0.9)
    title_sens = ax_sens.set_title('3. Sensor Pixels: 0 / 1024 (0.0%)', fontsize=10.5, fontweight='bold')

    fig.suptitle(
        f'Flash LiDAR Ray Tracing Animation  |  Surface A = {A} m, Focal Length f = {f} m, '
        f'diffuseFraction = {diffuse_fraction}',
        fontsize=12.5, fontweight='bold', y=0.98
    )

    def build_segments(f_idx, ray_indices):
        if len(ray_indices) == 0:
            return np.array([]), np.array([]), np.array([]), np.array([]), np.array([]), np.array([])
        f_prev = max(0, f_idx - 2)
        p_now = pos_hist[f_idx, ray_indices]
        p_old = pos_hist[f_prev, ray_indices]
        valid = ~np.isnan(p_now[:, 0]) & ~np.isnan(p_old[:, 0])
        p_now, p_old = p_now[valid], p_old[valid]
        if len(p_now) == 0:
            return np.array([]), np.array([]), np.array([]), np.array([]), np.array([]), np.array([])
        n_seg = len(p_now)
        xs = np.column_stack([p_old[:, 0], p_now[:, 0], np.full(n_seg, np.nan)]).ravel()
        ys = np.column_stack([p_old[:, 1], p_now[:, 1], np.full(n_seg, np.nan)]).ravel()
        zs = np.column_stack([p_old[:, 2], p_now[:, 2], np.full(n_seg, np.nan)]).ravel()
        return p_now[:, 0], p_now[:, 1], p_now[:, 2], xs, ys, zs

    def update(f_idx):
        t_now = frame_times[f_idx]

        # 1. Update 3D rays (split into outgoing vs post-reflection categories)
        bounced = rfln_hist[f_idx, sample_ids] >= 1
        out_ids = sample_ids[~bounced & active_hist[f_idx, sample_ids]]
        hit_ids = s_hit[rfln_hist[f_idx, s_hit] >= 1]
        off_ids = s_off[rfln_hist[f_idx, s_off] >= 1]
        mul_ids = s_mul[(rfln_hist[f_idx, s_mul] >= 1) & active_hist[f_idx, s_mul]]
        mis_ids = s_mis[(rfln_hist[f_idx, s_mis] >= 1) & active_hist[f_idx, s_mis]]

        for key, ids in [('out', out_ids), ('hit', hit_ids), ('off', off_ids),
                         ('mul', mul_ids), ('miss', mis_ids)]:
            hx, hy, hz, tx_s, ty_s, tz_s = build_segments(f_idx, ids)
            heads_3d[key].set_data(hx, hy)
            heads_3d[key].set_3d_properties(hz)
            tails_3d[key].set_data(tx_s, ty_s)
            tails_3d[key].set_3d_properties(tz_s)

        title_3d.set_text(f'1. 3D Ray Propagation (t = {t_now * 1e9:.2f} ns)')

        # 2. Update Lens Plane crossings up to t_now
        m_miss = tj[lens_plot_miss] <= t_now
        m_off = tj[lens_plot_off] <= t_now
        m_hit = tj[lens_plot_hit] <= t_now

        pts_miss_lens.set_data(rj[lens_plot_miss[m_miss], 0], rj[lens_plot_miss[m_miss], 1])
        pts_off_lens.set_data(rj[lens_plot_off[m_off], 0], rj[lens_plot_off[m_off], 1])
        pts_hit_lens.set_data(rj[lens_plot_hit[m_hit], 0], rj[lens_plot_hit[m_hit], 1])

        n_c_now = int(np.sum(tj[crossed_all_idx] <= t_now))
        n_ap_now = int(np.sum(tj[rtrn_idx] <= t_now))
        pct_ap = 100.0 * n_ap_now / n_c_now if n_c_now > 0 else 0.0
        title_lens.set_text(f'2. Lens Plane: {n_ap_now}/{n_c_now} in aperture ({pct_ap:.1f}%)')

        # 3. Update Sensor Array pixels & out-of-bounds spill up to t_now
        arrived_on = onArray & (tof <= t_now)
        tPix_now = np.full(nPix * nPix, np.inf)
        if np.any(arrived_on):
            np.minimum.at(tPix_now, row[arrived_on] * nPix + col[arrived_on], tof[arrived_on])
        hit_now = np.isfinite(tPix_now)
        recon_z = np.full(nPix * nPix, np.nan)
        recon_z[hit_now] = emitter[2] + (0.5 * c * tPix_now[hit_now]) * dz_pix[hit_now]

        im_sens.set_data(recon_z.reshape(nPix, nPix)[::-1, ::-1])

        arrived_off = tof_off <= t_now
        pts_spill.set_data(pImg_scene_off[arrived_off, 0], pImg_scene_off[arrived_off, 1])

        n_pix_lit = int(np.sum(hit_now))
        title_sens.set_text(
            f'3. Sensor Illuminated: {n_pix_lit}/{nPix*nPix} ({100*n_pix_lit/(nPix*nPix):.1f}%)'
        )
        return (
            list(heads_3d.values())
            + list(tails_3d.values())
            + [pts_miss_lens, pts_off_lens, pts_hit_lens, im_sens, pts_spill]
        )

    ani = FuncAnimation(fig, update, frames=num_frames, interval=80, blit=False)
    plt.close(fig)  # Prevent duplicate static figure below interactive widget
    return HTML(ani.to_jshtml())

