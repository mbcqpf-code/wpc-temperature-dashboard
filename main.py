import matplotlib
matplotlib.use("Agg")

import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from herbie import Herbie
from scipy.spatial import Delaunay
from scipy.interpolate import LinearNDInterpolator
from datetime import datetime, timedelta, timezone
from collections import defaultdict
import concurrent.futures
import threading
import requests
import tempfile
import os
import json
import time
import warnings
warnings.filterwarnings('ignore')

t_start = time.time()

# --- 1. CONFIGURATION & SESSION SETUP ---
os.makedirs("output", exist_ok=True)
_thread_local = threading.local()

MAX_DOWNLOAD_WORKERS = 20
MIN_GEFS_MEMBERS = 20
MIN_IFS_MEMBERS = 40
MIN_AIFS_MEMBERS = 40

nbm_triangulation = None
nbm_valid_mask = None

def get_http_session():
    if not hasattr(_thread_local, "session"):
        _thread_local.session = requests.Session()
    return _thread_local.session

# --- 2. AUTOMATIC SHIFT DETECTOR & CALENDAR SYNC ---
now = datetime.now(timezone.utc)
run_timestamp = now.strftime('%Y-%m-%d %H:%Mz')

if (now.hour == 1 and now.minute > 30) or (2 <= now.hour <= 12) or (now.hour == 13 and now.minute <= 30):
    print("⏰ Active Session: WPC Night Shift (18z NBM QMD / 12z Global Package)")
    base_date = now - timedelta(days=1)
    global_cycle = "12"
    nbm_cycle = "18"
    mode = "18z_package"
    date_offsets = [4, 5, 6, 7, 8]

    display_nbm_time = base_date + timedelta(days=1)
    display_nbm_str = display_nbm_time.strftime('%Y-%m-%d 01z')
else:
    print("⏰ Active Session: WPC Day Shift (06z NBM QMD / 00z Global Package)")
    base_date = now - timedelta(days=1) if now.hour <= 1 else now
    global_cycle = "00"
    nbm_cycle = "06"
    mode = "06z_package"
    date_offsets = [3, 4, 5, 6, 7]

    display_nbm_time = base_date
    display_nbm_str = display_nbm_time.strftime('%Y-%m-%d 13z')

global_init_time = base_date.strftime(f'%Y-%m-%d {global_cycle}:00')
nbm_date_str = base_date.strftime('%Y-%m-%d')
nbm_date_nomads = base_date.strftime('%Y%m%d')

print(f"📊 Global Model Baseline Cycle: {global_init_time}")
print(f"📊 NBM Baseline Cycle: {nbm_date_str} {nbm_cycle}:00 (Labeled as {display_nbm_str})")

lon_min, lon_max = -125.0, -65.0
lat_min, lat_max = 24.0, 50.0
grid_lats = np.arange(lat_min, lat_max + 0.25, 0.25)
grid_lons = np.arange(lon_min, lon_max + 0.25, 0.25)
target_lons, target_lats = np.meshgrid(grid_lons, grid_lats)

# --- 3. COLORMAP ENGINES & FAST CONUS GRID EXTRACTION ---
# A) Percentile Colormap
levels = np.arange(0, 105, 5)
base_cmap = plt.cm.RdYlBu_r
base_colors = base_cmap(np.linspace(0, 1, len(levels) - 1))
white = np.array([1.0, 1.0, 1.0, 1.0])
colors_b1 = base_colors.copy()

for i in range(len(levels) - 1):
    val = levels[i]
    if 25 <= val < 75:
        colors_b1[i] = colors_b1[i] * 0.30 + white * 0.70
        colors_b1[i][3] = 1.0

custom_percentile_cmap = mcolors.ListedColormap(colors_b1)
custom_percentile_cmap.set_bad(color='white', alpha=0)

# B) NWS Temperature Colormap (-60F to 120F) with smooth sub-range slicing
nws_hex = [
    "#91234c", "#cf529b", "#f0a9df", "#dce1ee", "#a09ed1",
    "#5a4299", "#13007a", "#3b67be", "#64aef2", "#9ae2ef",
    "#4f8f8b", "#3a793a", "#7bbb70", "#d9f1a7", "#ffdf74",
    "#f89b37", "#e43e26", "#8f121a", "#3b002b"
]
nws_full_cmap = mcolors.LinearSegmentedColormap.from_list("nws_temp_full", nws_hex)

def get_nws_subcmap(t_min, t_max, full_min=-60.0, full_max=120.0):
    lo = np.clip((t_min - full_min) / (full_max - full_min), 0.0, 1.0)
    hi = np.clip((t_max - full_min) / (full_max - full_min), 0.0, 1.0)
    sub = mcolors.LinearSegmentedColormap.from_list(
        f"nws_{int(t_min)}_{int(t_max)}",
        nws_full_cmap(np.linspace(lo, hi, 256))
    )
    sub.set_bad(color='white', alpha=0)
    return sub

nws_max_cmap = get_nws_subcmap(-20, 120)
nws_min_cmap = get_nws_subcmap(-40, 90)

def standardize_grid(ds):
    """Slices to CONUS before sorting/interpolating, uses exact .sel on 0.25-deg grids, and closes ds."""
    try:
        var_key = list(ds.data_vars)[0]
        lats = ds.latitude.values
        lons = ds.longitude.values

        lat_desc = lats[0] > lats[-1]
        lat_slice = slice(lat_max + 1.0, lat_min - 1.0) if lat_desc else slice(lat_min - 1.0, lat_max + 1.0)

        is_360 = lons.max() > 180.0
        lon_desc = lons[0] > lons[-1]
        if is_360:
            lo, hi = lon_min + 360.0 - 1.0, lon_max + 360.0 + 1.0
        else:
            lo, hi = lon_min - 1.0, lon_max + 1.0
        lon_slice = slice(hi, lo) if lon_desc else slice(lo, hi)

        sub = ds[var_key].sel(latitude=lat_slice, longitude=lon_slice)

        if is_360:
            sub = sub.assign_coords(longitude=(((sub.longitude + 180.0) % 360.0) - 180.0))

        sub = sub.sortby(['latitude', 'longitude'])

        try:
            exact = sub.sel(latitude=grid_lats, longitude=grid_lons)
            if exact.sizes['latitude'] == len(grid_lats) and exact.sizes['longitude'] == len(grid_lons):
                return exact.values.astype(np.float32)
        except KeyError:
            pass

        return sub.interp(latitude=grid_lats, longitude=grid_lons, method='linear').values.astype(np.float32)
    finally:
        ds.close()

# --- 4. RACE-FREE, DEDUPLICATED WORKER FETCHERS ---
def fetch_gefs_file(member, fxx, search_strings):
    """One thread per unique (member, fxx) file prevents .idx file collisions."""
    out = {}
    try:
        H = Herbie(
            global_init_time, model="gefs", product="atmos.5",
            member=member, fxx=fxx, priority=["aws", "nomads"], verbose=False
        )
        for s in search_strings:
            try:
                ds = H.xarray(s)
                if isinstance(ds, list):
                    ds = ds[0]
                out[s] = standardize_grid(ds)
            except Exception as e:
                print(f"⚠️ GEFS {member} f{fxx:03d} ({s}) failed: {e}")
    except Exception as e:
        print(f"⚠️ GEFS {member} f{fxx:03d} init failed: {e}")
    return out

def _unpack_ecmwf_ds_list(ds_list):
    if not isinstance(ds_list, list):
        ds_list = [ds_list]
    ctrl, pert = None, None
    for ds in ds_list:
        is_pert = ("number" in ds.dims and ds.sizes["number"] > 1)
        grid_data = standardize_grid(ds)
        if is_pert:
            pert = grid_data
        else:
            ctrl = np.squeeze(grid_data)
    return ctrl, pert

def fetch_ecmwf_file(model_name, fxx, search_strings):
    """One thread per unique (model_name, fxx) file prevents .index collisions."""
    out = {}
    try:
        H = Herbie(
            global_init_time, model=model_name, product="enfo",
            fxx=fxx, priority=["google", "ecmwf", "aws"], verbose=False
        )
        for s in search_strings:
            try:
                out[s] = _unpack_ecmwf_ds_list(H.xarray(s))
            except Exception as e:
                print(f"⚠️ {model_name.upper()} f{fxx:03d} ({s}) failed: {e}")
    except Exception as e:
        print(f"⚠️ {model_name.upper()} f{fxx:03d} init failed: {e}")
    return out

def fetch_nbm_bytes(nbm_fxx, stat):
    fxx_str = f"f{nbm_fxx:03d}"
    rel_path = f"blend.{nbm_date_nomads}/{nbm_cycle}/qmd/blend.t{nbm_cycle}z.qmd.{fxx_str}.co.grib2"
    mirrors = [
        f"https://noaa-nbm-grib2-pds.s3.amazonaws.com/{rel_path}",
        f"https://nomads.ncep.noaa.gov/pub/data/nccf/com/blend/prod/{rel_path}",
    ]
    stat_token = "max fcst" if stat == "max" else "min fcst"
    session = get_http_session()

    last_err = None
    for url in mirrors:
        try:
            r_idx = session.get(f"{url}.idx", timeout=15)
            r_idx.raise_for_status()
            idx_lines = r_idx.text.splitlines()

            target_idx = next(
                (i for i, line in enumerate(idx_lines)
                 if ":TMP:2 m above ground:" in line and stat_token in line and "50% level" in line),
                None,
            )
            if target_idx is None:
                raise ValueError(f"Could not locate 50th percentile {stat_token} grid in .idx")

            start_byte = int(idx_lines[target_idx].split(":")[1])
            end_byte = int(idx_lines[target_idx + 1].split(":")[1]) - 1 if target_idx + 1 < len(idx_lines) else ""
            r_grib = session.get(url, headers={"Range": f"bytes={start_byte}-{end_byte}"}, timeout=20)
            r_grib.raise_for_status()
            return r_grib.content
        except Exception as e:
            last_err = e

    print(f"❌ All NBM mirrors failed for f{nbm_fxx:03d} ({stat}): {last_err}")
    return None

def regrid_nbm_bytes(grib_bytes):
    global nbm_triangulation, nbm_valid_mask
    with tempfile.NamedTemporaryFile(delete=False, suffix=".grib2") as tmp:
        tmp.write(grib_bytes)
        tmp_path = tmp.name

    try:
        ds_nbm = xr.open_dataset(tmp_path, engine="cfgrib")
        var_key = list(ds_nbm.data_vars)[0]
        nbm_data = ds_nbm[var_key].values

        if nbm_triangulation is None:
            nbm_lat = ds_nbm.latitude.values
            nbm_lon = np.where(ds_nbm.longitude.values > 180, ds_nbm.longitude.values - 360, ds_nbm.longitude.values)
            nbm_valid_mask = (
                (nbm_lat >= lat_min - 3) & (nbm_lat <= lat_max + 3) &
                (nbm_lon >= lon_min - 3) & (nbm_lon <= lon_max + 3)
            )
            nbm_triangulation = Delaunay(np.column_stack((nbm_lon[nbm_valid_mask], nbm_lat[nbm_valid_mask])))
        ds_nbm.close()
    finally:
        os.remove(tmp_path)

    interp = LinearNDInterpolator(nbm_triangulation, nbm_data[nbm_valid_mask])
    return interp(target_lons, target_lats).astype(np.float32)

# --- 5. AGGREGATION & PLOTTING HELPERS ---
def aggregate_gefs(gefs_cache, members, fxx_hours, search_str, agg="max"):
    fn = np.max if agg == "max" else np.min
    results = {}
    for m in members:
        grids = [gefs_cache[(m, f)][search_str] for f in fxx_hours if search_str in gefs_cache.get((m, f), {})]
        if len(grids) == len(fxx_hours):
            results[m] = fn(np.stack(grids), axis=0)
    return results

def aggregate_ecmwf(ecmwf_cache, queries, agg="max"):
    """queries is a list of (fxx, search_str) tuples."""
    fn = np.max if agg == "max" else np.min
    ctrl_grids, pert_grids = [], []
    for fxx, search_str in queries:
        pair = ecmwf_cache.get(fxx, {}).get(search_str)
        if pair is not None:
            c, p = pair
            if c is not None:
                ctrl_grids.append(c)
            if p is not None:
                pert_grids.append(p)

    results = []
    if len(ctrl_grids) == len(queries):
        results.append(fn(np.stack(ctrl_grids), axis=0))
    if len(pert_grids) == len(queries):
        pert_agg = fn(np.stack(pert_grids), axis=0)
        for i in range(pert_agg.shape[0]):
            results.append(pert_agg[i])
    return results

def render_package_plots(
    offset, valid_date_str, stat_label, file_prefix,
    gefs_grids, ifs_grids, aifs_grids, nbm_regridded, t_levels, temp_cmap
):
    gefs_mean_f = (np.mean(np.stack(gefs_grids), axis=0) - 273.15) * 9 / 5 + 32
    ifs_mean_f = (np.mean(np.stack(ifs_grids), axis=0) - 273.15) * 9 / 5 + 32
    aifs_mean_f = (np.mean(np.stack(aifs_grids), axis=0) - 273.15) * 9 / 5 + 32
    nbm_f = (nbm_regridded - 273.15) * 9 / 5 + 32

    super_matrix = np.stack(gefs_grids + ifs_grids + aifs_grids)
    n_members = super_matrix.shape[0]

    # Tie-aware and NaN-safe empirical percentile rank
    valid_count = np.sum(~np.isnan(super_matrix), axis=0)
    below = np.sum(super_matrix < nbm_regridded, axis=0)
    equal = np.sum(super_matrix == nbm_regridded, axis=0)
    percentile_rank = np.where(valid_count > 0, ((below + 0.5 * equal) / valid_count) * 100.0, np.nan)

    # Clamp inside (0, 100) so exact 0% / 100% plateaus never create Cartopy reprojection holes
    percentile_clamped = np.clip(percentile_rank, 0.1, 99.9)
    percentile_rank_masked = np.ma.masked_where(np.isnan(nbm_regridded) | np.isnan(percentile_rank), percentile_clamped)

    title_valid = f"Valid Date: {valid_date_str}"
    title_inits = f"NBM Init: {display_nbm_str} | Ensemble Init: {global_init_time} (N={n_members})"

    f_pct = f"day_{offset}_{file_prefix}percentile.png"
    f_ver = f"day_{offset}_{file_prefix}verification.png"
    f_dif = f"day_{offset}_{file_prefix}difference.png"

    # --- PLOT 1: PERCENTILES ---
    fig1 = plt.figure(figsize=(14, 9))
    ax1 = plt.axes(projection=ccrs.LambertConformal(central_longitude=-96.0, central_latitude=39.2))
    ax1.set_extent([lon_min, lon_max, lat_min, lat_max], crs=ccrs.PlateCarree())
    ax1.add_feature(cfeature.COASTLINE, linewidth=1.2)
    ax1.add_feature(cfeature.BORDERS, linewidth=1.2)
    ax1.add_feature(cfeature.STATES, linewidth=0.4, linestyle="--")

    contour1 = ax1.contourf(
        target_lons, target_lats, percentile_rank_masked,
        levels=levels, cmap=custom_percentile_cmap, transform=ccrs.PlateCarree(), extend='neither'
    )
    cbar1 = plt.colorbar(contour1, shrink=0.7, pad=0.02, ticks=[0, 10, 25, 50, 75, 90, 100])
    cbar1.set_label('NBM Percentile Rank (%)', fontsize=12)
    cbar1.ax.hlines([10, 25, 75, 90], 0, 1, colors='black', linewidth=1.5, linestyles='--')

    ax1.set_title(
        f"{stat_label} Temperature Percentile Rank (NBM vs Superensemble)\n{title_valid} | {title_inits}",
        fontsize=13, loc='left', weight='bold'
    )
    fig1.subplots_adjust(left=0.05, right=0.95, top=0.88, bottom=0.10)
    fig1.text(0.98, 0.02, f"Generated: {run_timestamp}", ha='right', va='bottom', fontsize=10, color='gray')
    fig1.savefig(os.path.join("output", f_pct), dpi=150)

    # --- PLOT 2: VERIFICATION PANELS (NWS Color Curve) ---
    fig2, axs = plt.subplots(2, 2, figsize=(16, 11), subplot_kw={'projection': ccrs.LambertConformal(central_longitude=-96.0, central_latitude=39.2)})
    axs = axs.flatten()
    plot_data = [
        (gefs_mean_f, f"GEFS Ensemble Mean ({len(gefs_grids)} mem)"),
        (ifs_mean_f, f"IFS Ensemble Mean ({len(ifs_grids)} mem)"),
        (aifs_mean_f, f"AIFS Ensemble Mean ({len(aifs_grids)} mem, Corrected)"),
        (nbm_f, "NBM Native QMD Median"),
    ]
    t_min_val, t_max_val = float(t_levels[0]), float(t_levels[-1])

    for i, (dataset, title) in enumerate(plot_data):
        axs[i].set_extent([lon_min, lon_max, lat_min, lat_max], crs=ccrs.PlateCarree())
        axs[i].add_feature(cfeature.COASTLINE, linewidth=0.8)
        axs[i].add_feature(cfeature.STATES, linewidth=0.4, linestyle="--")
        cf = axs[i].contourf(
            target_lons, target_lats, dataset, levels=t_levels,
            cmap=temp_cmap, vmin=t_min_val, vmax=t_max_val,
            transform=ccrs.PlateCarree(), extend='both'
        )
        axs[i].set_title(title, fontsize=12, weight='bold')

    fig2.subplots_adjust(left=0.02, right=0.98, top=0.92, bottom=0.12, wspace=0.02, hspace=0.05)
    cbar_ax = fig2.add_axes([0.15, 0.06, 0.7, 0.02])
    cbar2 = fig2.colorbar(cf, cax=cbar_ax, orientation='horizontal', ticks=np.arange(t_min_val, t_max_val + 1, 10))
    cbar2.set_label(f'{stat_label} Temperature (°F)', fontsize=14)
    fig2.suptitle(
        f"Input Verification Checklist: {stat_label} Temperature Forecast Mapping\n{title_valid} | {title_inits}",
        fontsize=15, weight='bold', y=0.96
    )
    fig2.text(0.98, 0.02, f"Generated: {run_timestamp}", ha='right', va='bottom', fontsize=10, color='gray')
    fig2.savefig(os.path.join("output", f_ver), dpi=150)

    # --- PLOT 3: 3-PANEL DIFFERENCE MAP ---
    diff_levels = np.arange(-16, 18, 2)
    diff_data = [
        (nbm_f - gefs_mean_f, "NBM minus GEFS Mean"),
        (nbm_f - ifs_mean_f, "NBM minus IFS Mean"),
        (nbm_f - aifs_mean_f, "NBM minus AIFS Mean (Corrected)"),
    ]

    fig3, axs3 = plt.subplots(1, 3, figsize=(24, 7), subplot_kw={'projection': ccrs.LambertConformal(central_longitude=-96.0, central_latitude=39.2)})
    axs3 = axs3.flatten()

    for i, (dataset, title) in enumerate(diff_data):
        axs3[i].set_extent([lon_min, lon_max, lat_min, lat_max], crs=ccrs.PlateCarree())
        axs3[i].add_feature(cfeature.COASTLINE, linewidth=0.8)
        axs3[i].add_feature(cfeature.STATES, linewidth=0.4, linestyle="--")
        dataset_masked = np.ma.masked_where(np.isnan(nbm_regridded), dataset)
        cf3 = axs3[i].contourf(target_lons, target_lats, dataset_masked, levels=diff_levels, cmap='RdBu_r', transform=ccrs.PlateCarree(), extend='both')
        axs3[i].set_title(title, fontsize=12, weight='bold')

    fig3.subplots_adjust(left=0.02, right=0.98, top=0.88, bottom=0.15, wspace=0.02)
    cbar_ax3 = fig3.add_axes([0.15, 0.06, 0.7, 0.03])
    cbar3 = fig3.colorbar(cf3, cax=cbar_ax3, orientation='horizontal')
    cbar3.set_label('Temperature Difference (°F) [Positive/Red = NBM is Warmer]', fontsize=14)
    fig3.suptitle(
        f"Ensemble Bias Check ({stat_label}T): NBM QMD vs Global Means\n{title_valid} | {title_inits}",
        fontsize=15, weight='bold', y=0.98
    )
    fig3.text(0.98, 0.02, f"Generated: {run_timestamp}", ha='right', va='bottom', fontsize=10, color='gray')
    fig3.savefig(os.path.join("output", f_dif), dpi=150)

    plt.close('all')
    return [f_pct, f_ver, f_dif]

# --- 6. BUILD UNIQUE DOWNLOAD MANIFEST ACROSS ALL 5 DAYS ---
all_gefs_members = ["c00"] + [f"p{str(i).zfill(2)}" for i in range(1, 31)]
day_specs = []
gefs_tasks = defaultdict(set)
ifs_tasks = defaultdict(set)
aifs_tasks = defaultdict(set)
nbm_tasks = set()

for offset in date_offsets:
    valid_date = base_date + timedelta(days=offset)
    valid_date_str = valid_date.strftime('%A, %B %d, %Y')

    if mode == "18z_package":
        nbm_fxx_max = offset * 24 + 12  # 12z Day D -> 06z Day D+1
        nbm_fxx_min = offset * 24       # 00z Day D -> 18z Day D
        global_base_fxx = offset * 24   # 12z Day D
        fxx_synop_max = [global_base_fxx, global_base_fxx + 6, global_base_fxx + 12, global_base_fxx + 18]
        fxx_synop_min = [global_base_fxx - 12, global_base_fxx - 6, global_base_fxx, global_base_fxx + 6]
    else:
        nbm_fxx_max = offset * 24 + 24  # 12z Day D -> 06z Day D+1
        nbm_fxx_min = offset * 24 + 12  # 00z Day D -> 18z Day D
        global_base_fxx = offset * 24   # 00z Day D
        fxx_synop_max = [global_base_fxx + 12, global_base_fxx + 18, global_base_fxx + 24, global_base_fxx + 30]
        fxx_synop_min = [global_base_fxx, global_base_fxx + 6, global_base_fxx + 12, global_base_fxx + 18]

    fxx_period_max = fxx_synop_max[1:]
    fxx_period_min = fxx_synop_min[1:]

    ifs_max_queries = []
    for f in fxx_period_max:
        ifs_max_queries.extend([(f - 3, ":mx2t3:"), (f, ":mx2t3:")] if f <= 144 else [(f, ":mx2t6:")])

    ifs_min_queries = []
    for f in fxx_period_min:
        ifs_min_queries.extend([(f - 3, ":mn2t3:"), (f, ":mn2t3:")] if f <= 144 else [(f, ":mn2t6:")])

    day_specs.append({
        "offset": offset,
        "valid_date_str": valid_date_str,
        "nbm_fxx_max": nbm_fxx_max,
        "nbm_fxx_min": nbm_fxx_min,
        "fxx_synop_max": fxx_synop_max,
        "fxx_synop_min": fxx_synop_min,
        "fxx_period_max": fxx_period_max,
        "fxx_period_min": fxx_period_min,
        "ifs_max_queries": ifs_max_queries,
        "ifs_min_queries": ifs_min_queries,
    })

    for m in all_gefs_members:
        for f in fxx_period_max:
            gefs_tasks[(m, f)].add(":TMAX:2 m")
        for f in fxx_period_min:
            gefs_tasks[(m, f)].add(":TMIN:2 m")

    for f in set(fxx_synop_max + fxx_synop_min):
        gefs_tasks[("c00", f)].add(":TMP:2 m")
        ifs_tasks[f].add(":2t:")
        aifs_tasks[f].add(":2t:")

    for f, s in ifs_max_queries + ifs_min_queries:
        ifs_tasks[f].add(s)

    nbm_tasks.add((nbm_fxx_max, "max"))
    nbm_tasks.add((nbm_fxx_min, "min"))

print(
    f"\n⬇️ Downloading unique files concurrently "
    f"(GEFS={len(gefs_tasks)}, IFS={len(ifs_tasks)}, AIFS={len(aifs_tasks)}, NBM={len(nbm_tasks)})..."
)

# --- 7. UNIFIED CONCURRENT DOWNLOAD PHASE ---
gefs_cache = {}
ifs_cache = {}
aifs_cache = {}
nbm_bytes_cache = {}

with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_DOWNLOAD_WORKERS) as executor:
    future_map = {}

    for (m, f), searches in gefs_tasks.items():
        future_map[executor.submit(fetch_gefs_file, m, f, sorted(searches))] = ("gefs", (m, f))
    for f, searches in ifs_tasks.items():
        future_map[executor.submit(fetch_ecmwf_file, "ifs", f, sorted(searches))] = ("ifs", f)
    for f, searches in aifs_tasks.items():
        future_map[executor.submit(fetch_ecmwf_file, "aifs", f, sorted(searches))] = ("aifs", f)
    for (nbm_fxx, stat) in nbm_tasks:
        future_map[executor.submit(fetch_nbm_bytes, nbm_fxx, stat)] = ("nbm", (nbm_fxx, stat))

    for future in concurrent.futures.as_completed(future_map):
        kind, key = future_map[future]
        try:
            res = future.result()
            if kind == "gefs":
                gefs_cache[key] = res
            elif kind == "ifs":
                ifs_cache[key] = res
            elif kind == "aifs":
                aifs_cache[key] = res
            elif kind == "nbm" and res is not None:
                nbm_bytes_cache[key] = res
        except Exception as e:
            print(f"⚠️ Download task {kind} {key} raised: {e}")

print(f"✅ All downloads finished in {time.time() - t_start:.1f}s. Starting processing & plotting...")

# --- 8. PROCESS & PLOT EACH VALID DATE ---
completed_max_offsets = []
completed_min_offsets = []
written_files = set()

for spec in day_specs:
    offset = spec["offset"]
    valid_date_str = spec["valid_date_str"]
    print(f"\n🚀 Processing Forecast Package for Valid Date: {valid_date_str}")

    # --- A. MAXT ---
    gefs_max_dict = aggregate_gefs(gefs_cache, all_gefs_members, spec["fxx_period_max"], ":TMAX:2 m", "max")
    gefs_max_grids = [gefs_max_dict[m] for m in all_gefs_members if m in gefs_max_dict]
    gefs_c00_synop_max = aggregate_gefs(gefs_cache, ["c00"], spec["fxx_synop_max"], ":TMP:2 m", "max")

    ifs_max_grids = aggregate_ecmwf(ifs_cache, spec["ifs_max_queries"], "max")
    ifs_synop_max_grids = aggregate_ecmwf(ifs_cache, [(f, ":2t:") for f in spec["fxx_synop_max"]], "max")
    aifs_raw_max = aggregate_ecmwf(aifs_cache, [(f, ":2t:") for f in spec["fxx_synop_max"]], "max")

    gefs_max_delta = (
        np.clip(gefs_max_dict["c00"] - gefs_c00_synop_max["c00"], 0, None)
        if ("c00" in gefs_max_dict and "c00" in gefs_c00_synop_max) else 0.0
    )
    ifs_max_delta = (
        np.clip(np.mean(np.stack(ifs_max_grids), axis=0) - np.mean(np.stack(ifs_synop_max_grids), axis=0), 0, None)
        if (ifs_max_grids and ifs_synop_max_grids) else 0.0
    )
    diurnal_max_corr = (gefs_max_delta + ifs_max_delta) / 2.0
    aifs_max_grids = [g + diurnal_max_corr for g in aifs_raw_max] if aifs_raw_max else []

    nbm_max_bytes = nbm_bytes_cache.get((spec["nbm_fxx_max"], "max"))
    if (
        len(gefs_max_grids) >= MIN_GEFS_MEMBERS
        and len(ifs_max_grids) >= MIN_IFS_MEMBERS
        and len(aifs_max_grids) >= MIN_AIFS_MEMBERS
        and nbm_max_bytes is not None
    ):
        try:
            nbm_max_grid = regrid_nbm_bytes(nbm_max_bytes)
            print(
                f"📈 Plotting MaxT (GEFS={len(gefs_max_grids)}, IFS={len(ifs_max_grids)}, "
                f"AIFS={len(aifs_max_grids)}, N={len(gefs_max_grids)+len(ifs_max_grids)+len(aifs_max_grids)})..."
            )
            files = render_package_plots(
                offset, valid_date_str, "Max", "",
                gefs_max_grids, ifs_max_grids, aifs_max_grids, nbm_max_grid,
                t_levels=np.arange(-20, 122, 2), temp_cmap=nws_max_cmap
            )
            written_files.update(files)
            completed_max_offsets.append({"offset": offset, "valid_date": valid_date_str})
        except Exception as e:
            print(f"❌ Failed MaxT plotting for {valid_date_str}: {e}")
    else:
        print(
            f"⚠️ Skipping MaxT for {valid_date_str} "
            f"(GEFS={len(gefs_max_grids)}, IFS={len(ifs_max_grids)}, AIFS={len(aifs_max_grids)}, NBM={'ok' if nbm_max_bytes else 'missing'})."
        )

    # --- B. MINT ---
    gefs_min_dict = aggregate_gefs(gefs_cache, all_gefs_members, spec["fxx_period_min"], ":TMIN:2 m", "min")
    gefs_min_grids = [gefs_min_dict[m] for m in all_gefs_members if m in gefs_min_dict]
    gefs_c00_synop_min = aggregate_gefs(gefs_cache, ["c00"], spec["fxx_synop_min"], ":TMP:2 m", "min")

    ifs_min_grids = aggregate_ecmwf(ifs_cache, spec["ifs_min_queries"], "min")
    ifs_synop_min_grids = aggregate_ecmwf(ifs_cache, [(f, ":2t:") for f in spec["fxx_synop_min"]], "min")
    aifs_raw_min = aggregate_ecmwf(aifs_cache, [(f, ":2t:") for f in spec["fxx_synop_min"]], "min")

    gefs_min_delta = (
        np.clip(gefs_c00_synop_min["c00"] - gefs_min_dict["c00"], 0, None)
        if ("c00" in gefs_min_dict and "c00" in gefs_c00_synop_min) else 0.0
    )
    ifs_min_delta = (
        np.clip(np.mean(np.stack(ifs_synop_min_grids), axis=0) - np.mean(np.stack(ifs_min_grids), axis=0), 0, None)
        if (ifs_min_grids and ifs_synop_min_grids) else 0.0
    )
    nocturnal_min_corr = (gefs_min_delta + ifs_min_delta) / 2.0
    aifs_min_grids = [g - nocturnal_min_corr for g in aifs_raw_min] if aifs_raw_min else []

    nbm_min_bytes = nbm_bytes_cache.get((spec["nbm_fxx_min"], "min"))
    if (
        len(gefs_min_grids) >= MIN_GEFS_MEMBERS
        and len(ifs_min_grids) >= MIN_IFS_MEMBERS
        and len(aifs_min_grids) >= MIN_AIFS_MEMBERS
        and nbm_min_bytes is not None
    ):
        try:
            nbm_min_grid = regrid_nbm_bytes(nbm_min_bytes)
            print(
                f"📉 Plotting MinT (GEFS={len(gefs_min_grids)}, IFS={len(ifs_min_grids)}, "
                f"AIFS={len(aifs_min_grids)}, N={len(gefs_min_grids)+len(ifs_min_grids)+len(aifs_min_grids)})..."
            )
            files = render_package_plots(
                offset, valid_date_str, "Min", "min_",
                gefs_min_grids, ifs_min_grids, aifs_min_grids, nbm_min_grid,
                t_levels=np.arange(-40, 92, 2), temp_cmap=nws_min_cmap
            )
            written_files.update(files)
            completed_min_offsets.append({"offset": offset, "valid_date": valid_date_str})
        except Exception as e:
            print(f"❌ Failed MinT plotting for {valid_date_str}: {e}")
    else:
        print(
            f"⚠️ Skipping MinT for {valid_date_str} "
            f"(GEFS={len(gefs_min_grids)}, IFS={len(ifs_min_grids)}, AIFS={len(aifs_min_grids)}, NBM={'ok' if nbm_min_bytes else 'missing'})."
        )

# --- 9. SAFE MANIFEST UPDATE & STALE PNG CLEANUP ---
if completed_max_offsets or completed_min_offsets:
    for fname in os.listdir("output"):
        if fname.startswith("day_") and fname.endswith(".png") and fname not in written_files:
            try:
                os.remove(os.path.join("output", fname))
            except OSError:
                pass

    with open("output/manifest.json", "w") as f:
        json.dump({
            "offsets": completed_max_offsets,
            "min_offsets": completed_min_offsets,
            "run_timestamp": run_timestamp,
            "nbm_init": display_nbm_str,
            "global_init": global_init_time,
        }, f, indent=2)
    print(f"✅ Run Complete in {time.time() - t_start:.1f}s. Manifest updated and stale maps cleaned.")
else:
    print(f"⚠️ No valid packages generated in {time.time() - t_start:.1f}s. Preserving previous manifest and images.")
