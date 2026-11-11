"""Ordinary Kriging of per-location soil moisture and suction means.

Usage: python3 -I kriging.py <data_dir> <out_dir>
Implements the covariance-form OK system used in the paper (Lagrange multiplier,
row of ones), fits a spherical variogram with nugget by weighted least squares,
cross-checks against PyKrige, runs leave-one-location-out CV, and draws maps.
"""
import sys, itertools, json
import numpy as np, pandas as pd
from scipy.optimize import least_squares
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

DATA, OUT = sys.argv[1], sys.argv[2]
LAT0 = -6.366
M_PER_DEG_LAT = 111320.0
M_PER_DEG_LON = 111320.0 * np.cos(np.radians(LAT0))

BLUE = LinearSegmentedColormap.from_list("blue", ["#cde2fb", "#86b6ef", "#3987e5", "#1c5cab", "#0d366b"])
ORANGE = LinearSegmentedColormap.from_list("orange", ["#fbe3d8", "#f3a986", "#eb6834", "#b8461b", "#7a2c0f"])
INK, MUTED = "#1f1f1e", "#6b6a66"

def to_xy(lat, lon, ref):
    return np.c_[(lon - ref[1]) * M_PER_DEG_LON, (lat - ref[0]) * M_PER_DEG_LAT]

def spherical(h, c0, c, a):
    h = np.asarray(h, float)
    g = np.where(h < a, c0 + c * (1.5 * h / a - 0.5 * (h / a) ** 3), c0 + c)
    return np.where(h == 0, 0.0, g)

def empirical(xy, z, edges):
    d, sq = [], []
    for i, j in itertools.combinations(range(len(z)), 2):
        d.append(np.hypot(*(xy[i] - xy[j]))); sq.append(0.5 * (z[i] - z[j]) ** 2)
    d, sq = np.array(d), np.array(sq)
    rows = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (d >= lo) & (d < hi)
        if m.any(): rows.append((d[m].mean(), sq[m].mean(), m.sum()))
    return d, sq, np.array(rows)

def fit(binned, sill, hmax):
    h, g, n = binned.T
    def res(p): return np.sqrt(n) * (spherical(h, *p) - g)
    p0 = [0.1 * sill, 0.9 * sill, hmax / 2]
    r = least_squares(res, p0, bounds=([0, 0, 20], [2 * sill, 2 * sill, 2 * hmax]))
    return r.x

def ok_weights(xy, x0, vg):
    """Covariance form: [C 1; 1' 0][w; gamma] = [c0; 1], C(h) = sill - gamma(h)."""
    c0n, c, a = vg; sill = c0n + c
    n = len(xy)
    D = np.hypot(*(xy[:, None, :] - xy[None, :, :]).transpose(2, 0, 1))
    C = sill - spherical(D, *vg)
    A = np.ones((n + 1, n + 1)); A[:n, :n] = C; A[n, n] = 0.0
    d0 = np.hypot(*(xy - x0).T)
    b = np.append(sill - spherical(d0, *vg), 1.0)
    sol = np.linalg.solve(A, b)
    w, lag = sol[:n], sol[n]
    var = sill - w @ b[:n] - lag     # OK variance in covariance form (lag = Lagrange multiplier)
    return w, max(var, 0.0)

def krige(xy, z, pts, vg):
    est, var = np.empty(len(pts)), np.empty(len(pts))
    for k, p in enumerate(pts):
        w, v = ok_weights(xy, p, vg); est[k] = w @ z; var[k] = v
    return est, var

def idw(xy, z, p, power=2):
    d = np.hypot(*(xy - p).T)
    if (d < 1e-9).any(): return z[d.argmin()]
    w = 1 / d ** power; return w @ z / w.sum()

results, figs = {}, {}
targets = [("moisture", "moisture", "Soil moisture", "%"), ("suction", "suction_pressure", "Soil suction", "bar")]
fig_v, axes_v = plt.subplots(1, 2, figsize=(10, 3.6))
fig_m, axes_m = plt.subplots(2, 2, figsize=(10, 8.6))

m_all = pd.read_csv(f"{DATA}/moisture.csv")
ref = (m_all.latitude.mean(), m_all.longitude.mean())

for col, (name, t, label, unit) in enumerate(targets):
    df = pd.read_csv(f"{DATA}/{name}.csv")
    df = df[df[t] != 0]
    L = df.groupby("combine_loc").agg(lat=("latitude", "first"), lon=("longitude", "first"),
                                     z=(t, "mean"), sd=(t, "std"), n=(t, "size")).reset_index()
    xy = to_xy(L.lat.values, L.lon.values, ref); z = L.z.values
    sample_var = z.var(ddof=1)
    edges = [0, 150, 400, 600, 800, 1200]
    d, sq, binned = empirical(xy, z, edges)
    vg = fit(binned, sample_var, d.max())

    # cross-check with PyKrige using the same model parameters (psill = c, range = a, nugget = c0)
    from pykrige.ok import OrdinaryKriging
    okp = OrdinaryKriging(xy[:, 0], xy[:, 1], z, variogram_model="spherical",
                          variogram_parameters={"psill": vg[1], "range": vg[2], "nugget": vg[0]})

    # leave-one-location-out CV
    loo = []
    for i in range(len(z)):
        keep = np.arange(len(z)) != i
        e_ok, v_ok = krige(xy[keep], z[keep], xy[i:i + 1], vg)
        nn = z[keep][np.hypot(*(xy[keep] - xy[i]).T).argmin()]
        loo.append(dict(location=L.combine_loc[i], observed=z[i], OK=e_ok[0], OK_sd=np.sqrt(v_ok[0]),
                        mean_of_others=z[keep].mean(), IDW=idw(xy[keep], z[keep], xy[i]), nearest=nn))
    loo = pd.DataFrame(loo)
    rmse = {k: float(np.sqrt(((loo[k] - loo.observed) ** 2).mean())) for k in ["OK", "mean_of_others", "IDW", "nearest"]}
    within = float(((loo.observed - loo.OK).abs() <= 1.96 * loo.OK_sd).mean())

    # grid
    pad = 120
    gx = np.linspace(xy[:, 0].min() - pad, xy[:, 0].max() + pad, 110)
    gy = np.linspace(xy[:, 1].min() - pad, xy[:, 1].max() + pad, 110)
    GX, GY = np.meshgrid(gx, gy)
    est, var = krige(xy, z, np.c_[GX.ravel(), GY.ravel()], vg)
    est, sd = est.reshape(GX.shape), np.sqrt(var).reshape(GX.shape)
    pk_est, pk_var = okp.execute("grid", gx, gy)
    max_diff = float(np.abs(np.asarray(pk_est) - est).max())

    # example query points from the paper's interface section
    q = {"(-6.366611, 106.827818)": (-6.366611, 106.827818), "(-6.366803, 106.829041)": (-6.366803, 106.829041)}
    qres = {}
    for k, (la, lo) in q.items():
        e, v = krige(xy, z, to_xy(np.array([la]), np.array([lo]), ref), vg)
        qres[k] = dict(estimate=round(float(e[0]), 3), sd=round(float(np.sqrt(v[0])), 3))

    results[name] = dict(n_locations=int(len(z)), location_means=L[["combine_loc", "z", "sd", "n"]].round(3).to_dict("records"),
                         sample_variance=round(float(sample_var), 3),
                         variogram=dict(model="spherical", nugget=round(float(vg[0]), 3), partial_sill=round(float(vg[1]), 3),
                                        range_m=round(float(vg[2]), 1), nugget_share=round(float(vg[0] / (vg[0] + vg[1])), 3)),
                         binned=[dict(h_m=round(float(a), 1), gamma=round(float(b), 3), pairs=int(c)) for a, b, c in binned],
                         loocv_rmse={k: round(v, 3) for k, v in rmse.items()}, loocv_within_95pct=round(within, 2),
                         pykrige_max_abs_diff=max_diff, grid_estimate_range=[round(float(est.min()), 2), round(float(est.max()), 2)],
                         paper_query_points=qres)
    loo.round(3).to_csv(f"{OUT}/kriging_loocv_{name}.csv", index=False)
    pd.DataFrame({"x_m": GX.ravel(), "y_m": GY.ravel(),
                  "lat": ref[0] + GY.ravel() / M_PER_DEG_LAT, "lon": ref[1] + GX.ravel() / M_PER_DEG_LON,
                  "estimate": est.ravel(), "kriging_sd": sd.ravel()}).round(6).to_csv(f"{OUT}/kriging_grid_{name}.csv", index=False)

    # ---- variogram figure
    ax = axes_v[col]
    ax.scatter(d, sq, s=14, color="#86b6ef", label="pairs of locations", zorder=2)
    ax.scatter(binned[:, 0], binned[:, 1], s=60, color="#1c5cab", label="binned mean", zorder=3)
    hh = np.linspace(0, d.max() * 1.05, 200)
    ax.plot(hh, spherical(hh, *vg), color=INK, lw=2, label="fitted spherical model")
    ax.axhline(sample_var, color=MUTED, lw=1, ls="--")
    ax.text(hh[-1], sample_var, " sample variance", color=MUTED, va="bottom", ha="right", fontsize=8)
    ax.set_title(f"{label}: semivariogram of location means (n = {len(z)})", fontsize=10, color=INK, loc="left")
    ax.set_xlabel("Distance between locations (m)"); ax.set_ylabel(f"Semivariance ({unit}²)")
    for s in ["top", "right"]: ax.spines[s].set_visible(False)
    ax.grid(axis="y", color="#e6e5e1", lw=0.8); ax.set_axisbelow(True)
    if col == 0: ax.legend(frameon=False, fontsize=8, loc="upper left")

    # ---- maps
    for row, (field, cmap, ttl, cb) in enumerate([(est, BLUE, f"{label} estimate", f"{label} ({unit})"),
                                                  (sd, ORANGE, f"{label} kriging standard deviation", f"Std. deviation ({unit})")]):
        ax = axes_m[row, col]
        lims = (z.min(), z.max()) if row == 0 else (0, float(np.sqrt(vg[0] + vg[1])))
        im = ax.pcolormesh(GX, GY, field, cmap=cmap, shading="auto", vmin=lims[0], vmax=lims[1])
        ax.scatter(xy[:, 0], xy[:, 1], s=46, c="white", edgecolors=INK, linewidths=1.2, zorder=3)
        if row == 0:
            for i, ((x, y), v) in enumerate(zip(xy, z)):
                close = any(np.hypot(*(xy[j] - xy[i])) < 40 for j in range(i))
                off, ha = ((6, -15), "left") if close else ((6, 5), "left")
                ax.annotate(f"{v:.1f}", (x, y), xytext=off, textcoords="offset points", fontsize=8, color=INK, ha=ha,
                            bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.85))
        ax.set_aspect("equal"); ax.set_title(ttl, fontsize=10, color=INK, loc="left")
        ax.set_xlabel("East (m)"); ax.set_ylabel("North (m)")
        c = fig_m.colorbar(im, ax=ax, shrink=0.85); c.set_label(cb, fontsize=8); c.outline.set_visible(False); c.formatter.set_useOffset(False); c.update_ticks()
        for s in ax.spines.values(): s.set_color("#d4d3cf")

fig_v.tight_layout(); fig_v.savefig(f"{OUT}/kriging_variograms.png", dpi=160)
fig_m.suptitle("Ordinary Kriging on per-location means (colour scales span the observed site values)",
               fontsize=10, color=MUTED)
fig_m.tight_layout(); fig_m.savefig(f"{OUT}/kriging_maps.png", dpi=160)
json.dump(results, open(f"{OUT}/kriging_results.json", "w"), indent=1)
print(json.dumps(results, indent=1))
