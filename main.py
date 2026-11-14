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
import concurrent.futures
import requests
import tempfile
import os
import json
import warnings
warnings.filterwarnings('ignore')

# --- 1. DIRECTORY & SESSION SETUP ---
os.makedirs("output", exist_ok=True)
http_session = requests.Session()
nbm_triangulation = None
nbm_valid_mask = None

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

# --- 3. COLORMAP ENGINES & GRID STANDARDIZATION ---
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
custom_percentile_cmap.set_under(colors_b1[0])
custom_percentile_cmap.set_over(colors_b1[-1])

# B) NWS Temperature Colormap (-60F to 120F)
nws_hex = [
    "#91234c", "#cf529b", "#f0a9df", "#dce1ee", "#a09ed1",
    "#5a4299", "#13007a", "#3b67be", "#64aef2", "#9ae2ef",
    "#4f8f8b", "#3a793a", "#7bbb70", "#d9f1a7", "#ffdf74",
    "#f89b37", "#e43e26", "#8f121a", "#3b002b"
]
nws_temp_cmap = mcolors.LinearSegmentedColormap.from_list("nws_temp", nws_hex)
nws_temp_cmap.set_bad(color='white', alpha=0)

def standardize_grid(ds):
    var_key = list(ds.data_vars)[0]
    if ds.longitude.max() > 180:
        ds = ds.assign_coords(longitude=(((ds.longitude + 180) % 360) - 180))
    ds = ds.sortby(['latitude', 'longitude'])
    ds_interp = ds.interp(latitude=grid_lats, longitude=grid_lons, method='linear')
    return ds_interp[var_key].values

# --- 4. ASYNC DATA FETCHING FUNCTIONS ---
def _fetch_single_gefs(member, fxx, search_string):
    H = Herbie(
        global_init_time, model="gefs", product="atmos.5",
        member=member, fxx=fxx, priority=["aws", "nomads"], verbose=False
    )
    ds = H.xarray(search_string)
    return standardize_grid(ds)

def get_gefs_by_fxx(members, fxx_hours, search_string, max_workers=24):
    data = {m: {} for m in members}
    tasks = [(m, fxx) for m in members for fxx in fxx_hours]

    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_key = {
            executor.submit(_fetch_single_gefs, m, fxx, search_string): (m, fxx)
            for m, fxx in tasks
        }
        for future in concurrent.futures.as_completed(future_to_key):
            m, fxx = future_to_key[future]
            try:
                data[m][fxx] = future.result()
            except Exception as e:
                print(f"⚠️ GEFS {m} f{fxx:03d} ({search_string}) failed: {e}")
    return data

def aggregate_gefs_members(gefs_by_fxx, members, fxx_hours, agg="max"):
    fn = np.max if agg == "max" else np.min
    results = {}
    for m in members:
        grids = [gefs_by_fxx[m][f] for f in fxx_hours if f in gefs_by_fxx.get(m, {})]
        if len(grids) == len(fxx_hours):
            results[m] = fn(np.stack(grids), axis=0)
    return results

def _unpack_ecmwf_ds_list(ds_list):
    if not isinstance(ds_list, list):
        ds_list = [ds_list]
    pert, ctrl = None, None
    for ds in ds_list:
        grid_data = standardize_grid(ds)
        if "number" in ds.dims and ds.sizes["number"] > 1:
            pert = grid_data
        else:
            ctrl = np.squeeze(grid_data)
    return ctrl, pert

def get_ecmwf_synoptic_by_fxx(model_name, fxx_hours, search_string):
    by_fxx = {}
    for fxx in fxx_hours:
        try:
            H = Herbie(
                global_init_time, model=model_name, product="enfo",
                fxx=fxx, priority=["google", "ecmwf", "aws"], verbose=False
            )
            by_fxx[fxx] = _unpack_ecmwf_ds_list(H.xarray(search_string))
        except Exception as e:
            print(f"⚠️ {model_name.upper()} f{fxx:03d} ({search_string}) failed: {e}")
    return by_fxx

def aggregate_ecmwf_synoptic(by_fxx, fxx_hours, agg="max"):
    fn = np.max if agg == "max" else np.min
    ctrl_grids, pert_grids = [], []
    for f in fxx_hours:
        if f in by_fxx:
            c, p = by_fxx[f]
            if c is not None:
                ctrl_grids.append(c)
            if p is not None:
                pert_grids.append(p)

    results = []
    if len(ctrl_grids) == len(fxx_hours):
        results.append(fn(np.stack(ctrl_grids), axis=0))
    if len(pert_grids) == len(fxx_hours):
        pert_agg = fn(np.stack(pert_grids), axis=0)
        for i in range(pert_agg.shape[0]):
            results.append(pert_agg[i])
    return results

def get_ifs_extremes_grids(fxx_period_list, stat="max"):
    queries = []
    for f in fxx_period_list:
        if stat == "max":
            if f <= 144:
                queries.extend([(f - 3, ":mx2t3:"), (f, ":mx2t3:")])
            else:
                queries.append((f, ":mx2t6:"))
        else:
            if f <= 144:
                queries.extend([(f - 3, ":mn2t3:"), (f, ":mn2t3:")])
            else:
                queries.append((f, ":mn2t6:"))

    ctrl_grids, pert_grids = [], []
    for fxx, search_str in queries:
        try:
            H = Herbie(
                global_init_time, model="ifs", product="enfo",
                fxx=fxx, priority=["google", "ecmwf", "aws"], verbose=False
            )
            c, p = _unpack_ecmwf_ds_list(H.xarray(search_str))
            if c is not None:
                ctrl_grids.append(c)
            if p is not None:
                pert_grids.append(p)
        except Exception as e:
            print(f"⚠️ IFS f{fxx:03d} ({search_str}) failed: {e}")

    fn = np.max if stat == "max" else np.min
    results = []
    if len(ctrl_grids) == len(queries):
        results.append(fn(np.stack(ctrl_grids), axis=0))
    if len(pert_grids) == len(queries):
        pert_agg = fn(np.stack(pert_grids), axis=0)
        for i in range(pert_agg.shape[0]):
            results.append(pert_agg[i])
    return results

def fetch_and_regrid_nbm_50(nbm_date_nomads, nbm_cycle, nbm_fxx, stat="max"):
    global nbm_triangulation, nbm_valid_mask
    fxx_str = f"f{nbm_fxx:03d}"
    rel_path = f"blend.{nbm_date_nomads}/{nbm_cycle}/qmd/blend.t{nbm_cycle}z.qmd.{fxx_str}.co.grib2"
    mirrors = [
        f"https://noaa-nbm-grib2-pds.s3.amazonaws.com/{rel_path}",
        f"https://nomads.ncep.noaa.gov/pub/data/nccf/com/blend/prod/{rel_path}",
    ]
    stat_token = "max fcst" if stat == "max" else "min fcst"

    grib_bytes = None
    last_err = None
    for url in mirrors:
        try:
            r_idx = http_session.get(f"{url}.idx", timeout=15)
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
            r_grib = http_session.get(url, headers={"Range": f"bytes={start_byte}-{end_byte}"}, timeout=20)
            r_grib.raise_for_status()
            grib_bytes = r_grib.content
            break
        except Exception as e:
            last_err = e

    if grib_bytes is None:
        raise RuntimeError(f"All NBM mirrors failed ({stat}): {last_err}")

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
    return interp(target_lons, target_lats)

def render_package_plots(
    offset, valid_date_str, stat_label, file_prefix,
    gefs_grids, ifs_grids, aifs_grids, nbm_regridded, t_levels
):
    gefs_mean_f = (np.mean(np.stack(gefs_grids), axis=0) - 273.15) * 9 / 5 + 32
    ifs_mean_f = (np.mean(np.stack(ifs_grids), axis=0) - 273.15) * 9 / 5 + 32
    aifs_mean_f = (np.mean(np.stack(aifs_grids), axis=0) - 273.15) * 9 / 5 + 32
    nbm_f = (nbm_regridded - 273.15) * 9 / 5 + 32

    super_matrix = np.stack(gefs_grids + ifs_grids + aifs_grids)
    members_below_nbm = np.sum(super_matrix < nbm_regridded, axis=0)
    percentile_rank = (members_below_nbm / super_matrix.shape[0]) * 100
    percentile_rank_masked = np.ma.masked_where(np.isnan(nbm_regridded), percentile_rank)

    title_valid = f"Valid Date: {valid_date_str}"
    title_inits = f"NBM Init: {display_nbm_str} | Ensemble Init: {global_init_time}"

    # --- PLOT 1: PERCENTILES ---
    fig1 = plt.figure(figsize=(14, 9))
    ax1 = plt.axes(projection=ccrs.LambertConformal(central_longitude=-96.0, central_latitude=39.2))
    ax1.set_extent([lon_min, lon_max, lat_min, lat_max], crs=ccrs.PlateCarree())
    ax1.add_feature(cfeature.COASTLINE, linewidth=1.2)
    ax1.add_feature(cfeature.BORDERS, linewidth=1.2)
    ax1.add_feature(cfeature.STATES, linewidth=0.4, linestyle="--")

    contour1 = ax1.contourf(
        target_lons, target_lats, percentile_rank_masked,
        levels=levels, cmap=custom_percentile_cmap, transform=ccrs.PlateCarree(), extend='both'
    )
    cbar1 = plt.colorbar(contour1, shrink=0.7, pad=0.02, ticks=[0, 10, 25, 50, 75, 90, 100])
    cbar1.set_label('NBM Percentile Rank (%)', fontsize=12)
    cbar1.ax.hlines([10, 25, 75, 90], 0, 1, colors='black', linewidth=1.5, linestyles='--')

    ax1.set_title(
        f"{stat_label} Temperature Percentile Rank (NBM vs Superensemble)\n{title_valid} | {title_inits}",
        fontsize=13, loc='left', weight='bold'
    )
    # Replaced bbox_inches='tight' with fixed subplots adjustments 
    fig1.subplots_adjust(left=0.05, right=0.95, top=0.88, bottom=0.10)
    fig1.text(0.98, 0.02, f"Generated: {run_timestamp}", ha='right', va='bottom', fontsize=10, color='gray')
    fig1.savefig(f"output/day_{offset}_{file_prefix}percentile.png", dpi=150) 

    # --- PLOT 2: VERIFICATION PANELS (Using NWS Color Curve) ---
    fig2, axs = plt.subplots(2, 2, figsize=(16, 11), subplot_kw={'projection': ccrs.LambertConformal(central_longitude=-96.0, central_latitude=39.2)})
    axs = axs.flatten()
    plot_data = [
        (gefs_mean_f, "GEFS Ensemble Mean"),
        (ifs_mean_f, "IFS Ensemble Mean"),
        (aifs_mean_f, "AIFS Ensemble Mean (Corrected)"),
        (nbm_f, "NBM Native QMD Median"),
    ]

    for i, (dataset, title) in enumerate(plot_data):
        axs[i].set_extent([lon_min, lon_max, lat_min, lat_max], crs=ccrs.PlateCarree())
        axs[i].add_feature(cfeature.COASTLINE, linewidth=0.8)
        axs[i].add_feature(cfeature.STATES, linewidth=0.4, linestyle="--")
        # NWS Curve Anchored from -60 to 120
        cf = axs[i].contourf(
            target_lons, target_lats, dataset, levels=t_levels,
            cmap=nws_temp_cmap, vmin=-60, vmax=120,
            transform=ccrs.PlateCarree(), extend='both'
        )
        axs[i].set_title(title, fontsize=12, weight='bold')

    fig2.subplots_adjust(left=0.02, right=0.98, top=0.92, bottom=0.12, wspace=0.02, hspace=0.05)
    cbar_ax = fig2.add_axes([0.15, 0.06, 0.7, 0.02])
    cbar2 = fig2.colorbar(cf, cax=cbar_ax, orientation='horizontal')
    cbar2.set_label(f'{stat_label} Temperature (°F)', fontsize=14)
    fig2.suptitle(
        f"Input Verification Checklist: {stat_label} Temperature Forecast Mapping\n{title_valid} | {title_inits}",
        fontsize=15, weight='bold', y=0.96
    )
    fig2.text(0.98, 0.02, f"Generated: {run_timestamp}", ha='right', va='bottom', fontsize=10, color='gray')
    fig2.savefig(f"output/day_{offset}_{file_prefix}verification.png", dpi=150)

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
    fig3.savefig(f"output/day_{offset}_{file_prefix}difference.png", dpi=150)

    plt.close('all')

# --- 5. PRODUCTION MULTI-DAY LOOP ---
completed_max_offsets = []
completed_min_offsets = []
all_gefs_members = ["c00"] + [f"p{str(i).zfill(2)}" for i in range(1, 31)]

for offset in date_offsets:
    valid_date = base_date + timedelta(days=offset)
    valid_date_str = valid_date.strftime('%A, %B %d, %Y')
    print(f"\n🚀 Processing Forecast Package for Valid Date: {valid_date_str}")

    if mode == "18z_package":
        nbm_fxx_max = offset * 24 + 12  
        nbm_fxx_min = offset * 24       
        global_base_fxx = offset * 24   
        fxx_synop_max = [global_base_fxx, global_base_fxx + 6, global_base_fxx + 12, global_base_fxx + 18]
        fxx_synop_min = [global_base_fxx - 12, global_base_fxx - 6, global_base_fxx, global_base_fxx + 6]
    else:
        nbm_fxx_max = offset * 24 + 24  
        nbm_fxx_min = offset * 24 + 12  
        global_base_fxx = offset * 24   
        fxx_synop_max = [global_base_fxx + 12, global_base_fxx + 18, global_base_fxx + 24, global_base_fxx + 30]
        fxx_synop_min = [global_base_fxx, global_base_fxx + 6, global_base_fxx + 12, global_base_fxx + 18]

    fxx_period_max = fxx_synop_max[1:]  
    fxx_period_min = fxx_synop_min[1:]  
    all_synop_fxx = sorted(set(fxx_synop_max + fxx_synop_min))

    with concurrent.futures.ThreadPoolExecutor(max_workers=7) as model_exec:
        fut_gefs_tmax = model_exec.submit(get_gefs_by_fxx, all_gefs_members, fxx_period_max, ":TMAX:2 m", 24)
        fut_gefs_tmin = model_exec.submit(get_gefs_by_fxx, all_gefs_members, fxx_period_min, ":TMIN:2 m", 24)
        fut_gefs_synop = model_exec.submit(get_gefs_by_fxx, ["c00"], all_synop_fxx, ":TMP:2 m", 6)
        fut_ifs_max = model_exec.submit(get_ifs_extremes_grids, fxx_period_max, "max")
        fut_ifs_min = model_exec.submit(get_ifs_extremes_grids, fxx_period_min, "min")
        fut_ifs_synop = model_exec.submit(get_ecmwf_synoptic_by_fxx, "ifs", all_synop_fxx, ":2t:")
        fut_aifs_synop = model_exec.submit(get_ecmwf_synoptic_by_fxx, "aifs", all_synop_fxx, ":2t:")

        gefs_tmax_raw = fut_gefs_tmax.result()
        gefs_tmin_raw = fut_gefs_tmin.result()
        gefs_synop_raw = fut_gefs_synop.result()
        ifs_max_grids = fut_ifs_max.result()
        ifs_min_grids = fut_ifs_min.result()
        ifs_synop_raw = fut_ifs_synop.result()
        aifs_synop_raw = fut_aifs_synop.result()

    # --- A. PROCESS & PLOT MAXT ---
    gefs_max_dict = aggregate_gefs_members(gefs_tmax_raw, all_gefs_members, fxx_period_max, "max")
    gefs_max_grids = [gefs_max_dict[m] for m in all_gefs_members if m in gefs_max_dict]
    gefs_c00_synop_max = aggregate_gefs_members(gefs_synop_raw, ["c00"], fxx_synop_max, "max")
    ifs_synop_max_grids = aggregate_ecmwf_synoptic(ifs_synop_raw, fxx_synop_max, "max")
    aifs_raw_max = aggregate_ecmwf_synoptic(aifs_synop_raw, fxx_synop_max, "max")

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

    if gefs_max_grids and ifs_max_grids and aifs_max_grids:
        try:
            nbm_max_grid = fetch_and_regrid_nbm_50(nbm_date_nomads, nbm_cycle, nbm_fxx_max, "max")
            print("📈 Plotting MaxT maps...")
            render_package_plots(
                offset, valid_date_str, "Max", "",
                gefs_max_grids, ifs_max_grids, aifs_max_grids, nbm_max_grid,
                t_levels=np.arange(-20, 122, 2)
            )
            completed_max_offsets.append({"offset": offset, "valid_date": valid_date_str})
        except Exception as e:
            print(f"❌ Failed MaxT NBM/plotting for {valid_date_str}: {e}")
    else:
        print(f"⚠️ Incomplete MaxT arrays for {valid_date_str}. Skipping MaxT.")

    # --- B. PROCESS & PLOT MINT ---
    gefs_min_dict = aggregate_gefs_members(gefs_tmin_raw, all_gefs_members, fxx_period_min, "min")
    gefs_min_grids = [gefs_min_dict[m] for m in all_gefs_members if m in gefs_min_dict]
    gefs_c00_synop_min = aggregate_gefs_members(gefs_synop_raw, ["c00"], fxx_synop_min, "min")
    ifs_synop_min_grids = aggregate_ecmwf_synoptic(ifs_synop_raw, fxx_synop_min, "min")
    aifs_raw_min = aggregate_ecmwf_synoptic(aifs_synop_raw, fxx_synop_min, "min")

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

    if gefs_min_grids and ifs_min_grids and aifs_min_grids:
        try:
            nbm_min_grid = fetch_and_regrid_nbm_50(nbm_date_nomads, nbm_cycle, nbm_fxx_min, "min")
            print("📉 Plotting MinT maps...")
            render_package_plots(
                offset, valid_date_str, "Min", "min_",
                gefs_min_grids, ifs_min_grids, aifs_min_grids, nbm_min_grid,
                t_levels=np.arange(-40, 92, 2)
            )
            completed_min_offsets.append({"offset": offset, "valid_date": valid_date_str})
        except Exception as e:
            print(f"❌ Failed MinT NBM/plotting for {valid_date_str}: {e}")
    else:
        print(f"⚠️ Incomplete MinT arrays for {valid_date_str}. Skipping MinT.")

# --- 6. METADATA MANIFEST GENERATION ---
with open("output/manifest.json", "w") as f:
    json.dump({
        "offsets": completed_max_offsets,
        "min_offsets": completed_min_offsets,
        "run_timestamp": run_timestamp,
        "nbm_init": display_nbm_str,
        "global_init": global_init_time,
    }, f)
print("✅ Run Complete. Manifest written.")
