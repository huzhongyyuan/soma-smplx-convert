"""Tiny CPU mesh renderer (matplotlib painter's algorithm) for side-by-side SOMA / SMPL-X comparisons.
Grey torso, red anatomical left, blue anatomical right.  World is y-up."""
import numpy as np

AZ, EL, HALF = np.radians(-35), np.radians(12), 1.25
CA, SA, CE, SE = np.cos(AZ), np.sin(AZ), np.cos(EL), np.sin(EL)
LIGHT = np.array([0.4, 0.8, 0.45]); LIGHT /= np.linalg.norm(LIGHT)
PALETTE = np.array([[0.62, 0.62, 0.64], [0.84, 0.25, 0.24], [0.22, 0.42, 0.75]])
SMPLX_LEFT = {1, 4, 7, 10, 13, 16, 18, 20, 23, *range(25, 40)}
SMPLX_RIGHT = {2, 5, 8, 11, 14, 17, 19, 21, 24, *range(40, 55)}


def smplx_face_rgb(smplx_npz, faces):
    dom = np.load(smplx_npz, allow_pickle=True)['weights'].argmax(1)
    side = np.array([1 if j in SMPLX_LEFT else 2 if j in SMPLX_RIGHT else 0 for j in dom])
    return PALETTE[side[faces[:, 0]]]


def soma_face_rgb(skin, faces):
    names = skin.skeleton_skin.bone_order_names
    li, lw = skin.lbs_indices.numpy(), skin.lbs_weights.numpy()
    dom = li[np.arange(len(li)), lw.argmax(1)]
    side = np.array([1 if names[j].startswith('Left') else 2 if names[j].startswith('Right') else 0 for j in dom])
    return PALETTE[side[faces[:, 0]]]


def view(p, c):
    """world (y-up) -> screen x, y, depth (larger = closer). c: centre xz."""
    x = p[..., 0]-c[0]; z = p[..., 2]-c[1]; y = p[..., 1]
    x1 = CA*x - SA*z; z1 = SA*x + CA*z
    return np.stack([x1, CE*y - SE*z1, SE*y + CE*z1], -1)


def draw(ax, v, faces, rgb, c):
    from matplotlib.collections import PolyCollection
    s = view(v, c); tri = s[faces]
    vis = np.cross(tri[:, 1]-tri[:, 0], tri[:, 2]-tri[:, 0])[:, 2] > 0
    wn = np.cross(v[faces[:, 1]]-v[faces[:, 0]], v[faces[:, 2]]-v[faces[:, 0]])
    wn /= np.linalg.norm(wn, axis=1, keepdims=True).clip(1e-12)
    shade = 0.45 + 0.55*np.clip(wn @ LIGHT, 0, 1)
    order = np.argsort(tri[vis][:, :, 2].mean(1))
    ax.add_collection(PolyCollection(tri[vis][order][:, :, :2], facecolors=(rgb[vis]*shade[vis, None])[order],
                                     edgecolors='none', antialiased=False))


def draw_floor(ax, c):
    from matplotlib.collections import PolyCollection
    g = np.array([[c[0]-3, 0, c[1]-3], [c[0]+3, 0, c[1]-3], [c[0]+3, 0, c[1]+3], [c[0]-3, 0, c[1]+3]])
    ax.add_collection(PolyCollection([view(g, c)[:, :2]], facecolors='#e6e6e6', edgecolors='none', zorder=0))
    for k in np.arange(-3, 3.01, 0.5):
        for a, b in ((np.array([c[0]-3, 0, np.round(c[1])+k]), np.array([c[0]+3, 0, np.round(c[1])+k])),
                     (np.array([np.round(c[0])+k, 0, c[1]-3]), np.array([np.round(c[0])+k, 0, c[1]+3]))):
            s = view(np.stack([a, b]), c); ax.plot(s[:, 0], s[:, 1], color='#cfcfcf', lw=0.6, zorder=0)


def comparison_video(path, title, names, meshes, err_faces, err, centre, fps=30, heat_mm=20.0):
    """4 panels: meshes = [(verts [T,V,3], faces, face_rgb)] x3, 4th = per-vertex error heat map on meshes[2]
    (err [T,V] mm, white 0 -> dark red >= heat_mm) + per-frame error curve. Writes mp4; returns worst frame."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib import animation
    try:
        import imageio_ffmpeg
        matplotlib.rcParams['animation.ffmpeg_path'] = imageio_ffmpeg.get_ffmpeg_exe()
    except ImportError:
        pass
    n = len(err); t = np.arange(n) / fps
    fig = plt.figure(figsize=(18, 7.4), dpi=72); fig.suptitle(title, fontsize=11)
    axs = [fig.add_axes([0.005 + i * 0.248, 0.27, 0.245, 0.66]) for i in range(4)]
    kax = fig.add_axes([0.05, 0.06, 0.9, 0.15])
    kax.plot(t, err.mean(1), color='#555', lw=1.2, label='mean'); kax.plot(t, np.percentile(err, 95, 1), color='#ff7f0e', lw=1.2, label='p95')
    kax.plot(t, err.max(1), color='#d62728', lw=1.2, label='max'); kax.set_title('round-trip vertex error per frame (mm)', fontsize=9)
    kax.legend(fontsize=7, loc='upper right'); cur = kax.axvline(0, color='k', lw=0.8)

    def upd(f):
        c = centre[f]
        for i, ax in enumerate(axs):
            ax.cla(); ax.set_xlim(-HALF, HALF); ax.set_ylim(-0.35, 2 * HALF - 0.35); ax.set_aspect('equal'); ax.axis('off')
            ax.set_title(names[i], fontsize=10); draw_floor(ax, c)
            if i < 3: draw(ax, meshes[i][0][f], meshes[i][1], meshes[i][2], c)
            else:
                e = np.clip(err[f][err_faces].mean(1) / heat_mm, 0, 1)
                draw(ax, meshes[2][0][f], err_faces, np.stack([1 - 0.45 * e, 1 - e, 1 - e], 1) * 0.95, c)
        axs[3].text(-HALF + 0.05, -0.3, f'{t[f]:4.2f}s  mean {err[f].mean():.1f}  max {err[f].max():.1f} mm  (white 0, dark red >= {heat_mm:.0f})', fontsize=8)
        cur.set_xdata([t[f], t[f]]); return []
    animation.FuncAnimation(fig, upd, frames=n, interval=1000 / fps).save(path, writer=animation.FFMpegWriter(fps=fps, bitrate=2400))
    fw = int(err.max(1).argmax()); upd(fw); fig.savefig(str(path).rsplit('.', 1)[0] + '_worst.png'); plt.close(fig)
    return fw


def mp4_to_gif(mp4, gif, width=960, fps=15):
    import subprocess
    try:
        import imageio_ffmpeg; ff = imageio_ffmpeg.get_ffmpeg_exe()
    except ImportError:
        ff = 'ffmpeg'
    vf = f'fps={fps},scale={width}:-1:flags=lanczos'
    subprocess.run([ff, '-y', '-loglevel', 'error', '-i', str(mp4), '-vf',
                    f'{vf},split[a][b];[a]palettegen=max_colors=96[p];[b][p]paletteuse=dither=bayer', str(gif)], check=True)
