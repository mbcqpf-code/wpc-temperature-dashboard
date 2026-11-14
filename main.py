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

# --- 1. DIRECTORY SETUP FOR GITHUB ACTIONS ---
os.makedirs("output", exist_ok=True)

# Shared HTTP session for TCP connection pooling
http_session = requests.Session()
nbm_triangulation = None
nbm_valid_mask = None
completed_offsets = []
valid_dates_map = {}

# --- 2. AUTOMATIC SHIFT DETECTOR & CALENDAR SYNC ---
now = datetime.now(timezone.utc)
run_timestamp = now.strftime('%Y-%m-%d %H:%Mz')

# Determine operational window based on WPC arrival schedules
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

# CONUS Domain Setup
lon_min, lon_max = -125.0, -65.0
lat_min, lat_max = 24.0, 50.0

grid_lats = np.arange(lat_min, lat_max + 0.25, 0.25)
grid_lons = np.arange(lon_min, lon_max + 0.25, 0.25)
target_lons, target_lats = np.meshgrid(grid_lons, grid_lats)

# --- 3. CUSTOM COLORMAP ENGINE ---
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
custom_percentile_cmap.set_under(color='white', alpha=0)

def standardize_grid(ds):
    var_key = list(ds.data_vars)[0]
    if ds.longitude.max() > 180:
        ds = ds.assign_coords(longitude=(((ds.longitude + 180) % 360) - 180))
    ds = ds.sortby(['latitude', 'longitude'])
    ds_interp = ds.interp(latitude=grid_lats, longitude=grid_lons, method='linear')
    return ds_interp[var_key].values

# --- 4. DATA FETCHING FUNCTIONS ---
def get_gefs_grid(member, fxx_hours, search_string):
    daily_grids = []
    for fxx in fxx_hours:
        try:
            H = Herbie(global_init_time, model="gefs", product="atmos.5", member=member, fxx=fxx)
            ds = H.xarray(search_string)
            daily_grids.append(standardize_grid(ds))
        except Exception:
            pass 
    if len(daily_grids) == len(fxx_hours) and len(fxx_hours) > 0:
        return np.max(np.stack(daily_grids), axis=0)
    return None

def get_ecmwf_synoptic_grids(model_name, fxx_hours, search_string):
    pert_grids = []
    ctrl_grids = []
    for fxx in fxx_hours:
        try:
            H = Herbie(global_init_time, model=model_name, product="enfo", fxx=fxx)
            ds_list = H.xarray(search_string)
            if not isinstance(ds_list, list): ds_list = [ds_list]
            for ds in ds_list:
                grid_data = standardize_grid(ds)
                
                # Robust Control Member Check
                is_ctrl = False
                if ds.attrs.get("GRIB_dataType") == "cf" or (hasattr(ds, 'number') and getattr(ds, 'number') == 0):
                    is_ctrl = True
                elif 'number' not in ds.dims:
                    is_ctrl = True

                if is_ctrl: ctrl_grids.append(grid_data)
                else: pert_grids.append(grid_data)
        except Exception:
            pass 
            
    results = []
    if len(pert_grids) == len(fxx_hours) * 50: # Expecting 50 perturbed members
        stacked_perts = np.stack(pert_grids)
        # Reshape to (fxx, members, lat, lon) then take max over fxx axis
        grouped = stacked_perts.reshape(len(fxx_hours), 50, *stacked_perts.shape[1:])
        pert_max = np.max(grouped, axis=0)
        for i in range(pert_max.shape[0]): results.append(pert_max[i])
    if len(ctrl_grids) == len(fxx_hours):
        results.insert(0, np.max(np.stack(ctrl_grids), axis=0))
    return results

def get_ifs_max_grids(fxx_hours):
    queries = []
    for f in fxx_hours:
        if f <= 144:
            queries.append((f - 3, ":mx2t3:"))
            queries.append((f, ":mx2t3:"))
        else:
            queries.append((f, ":mx2t6:"))
            
    all_perts = []
    all_ctrls = []
    for fxx, search_str in queries:
        try:
            H = Herbie(global_init_time, model="ifs", product="enfo", fxx=fxx)
            ds_list = H.xarray(search_str)
            if not isinstance(ds_list, list): ds_list = [ds_list]
            for ds in ds_list:
                grid_data = standardize_grid(ds)
                is_ctrl = ds.attrs.get("GRIB_dataType") == "cf" or (hasattr(ds, 'number') and getattr(ds, 'number') == 0) or 'number' not in ds.dims
                
                if is_ctrl: all_ctrls.append(grid_data)
                else: all_perts.append(grid_data)
        except Exception:
            pass 
            
    results = []
    if len(all_perts) == len(queries) * 50:
        stacked_perts = np.stack(all_perts)
        grouped = stacked_perts.reshape(len(queries), 50, *stacked_perts.shape[1:])
        pert_max = np.max(grouped, axis=0)
        for i in range(pert_max.shape[0]): results.append(pert_max[i])
    if len(all_ctrls) == len(queries):
        results.insert(0, np.max(np.stack(all_ctrls), axis=0))
    return results

def fetch_nbm_maxt_50(nbm_date_nomads, nbm_cycle, nbm_fxx):
    fxx_str = f"f{nbm_fxx:03d}"
    rel_path = f"blend.{nbm_date_nomads}/{nbm_cycle}/qmd/blend.t{nbm_cycle}z.qmd.{fxx_str}.co.grib2"
    mirrors = [
        f"https://noaa-nbm-grib2-pds.s3.amazonaws.com/{rel_path}",
        f"https://nomads.ncep.noaa.gov/pub/data/nccf/com/blend/prod/{rel_path}",
    ]

    last_err = None
    for url in mirrors:
        try:
            r_idx = http_session.get(f"{url}.idx", timeout=15)
            r_idx.raise_for_status()
            idx_lines = r_idx.text.splitlines()

            target_idx = next((i for i, line in enumerate(idx_lines) if ":TMP:2 m above ground:" in line and "max fcst" in line and "50% level" in line), None)
            if target_idx is None: raise ValueError("Could not locate 50th percentile MaxT grid in .idx")

            start_byte = int(idx_lines[target_idx].split(":")[1])
            end_byte = int(idx_lines[target_idx + 1].split(":")[1]) - 1 if target_idx + 1 < len(idx_lines) else ""
            r_grib = http_session.get(url, headers={"Range": f"bytes={start_byte}-{end_byte}"}, timeout=20)
            r_grib.raise_for_status()
            return r_grib.content
        except Exception as e:
            last_err = e
    raise RuntimeError(f"All NBM mirrors failed: {last_err}")

# --- 5. PRODUCTION MULTI-DAY LOOP ---
for offset in date_offsets:
    valid_date = base_date + timedelta(days=offset)
    valid_date_str = valid_date.strftime('%A, %B %d, %Y')
    valid_dates_map[str(offset)] = valid_date_str
    print(f"\n🚀 Processing Forecast Package for Valid Date: {valid_date_str}")
    
    if mode == "18z_package":
        nbm_fxx = offset * 24 + 12
        global_base_fxx = offset * 24
        fxx_list_6hr = [global_base_fxx, global_base_fxx + 6, global_base_fxx + 12, global_base_fxx + 18] 
    else:  
        nbm_fxx = offset * 24 + 24
        global_base_fxx = offset * 24
        fxx_list_6hr = [global_base_fxx + 12, global_base_fxx + 18, global_base_fxx + 24, global_base_fxx + 30] 

    # Strictly slice off the 12z window for period-maximum fields to prevent morning temperature contamination
    fxx_max = fxx_list_6hr[1:]
    fxx_synop = fxx_list_6hr

    # Fetch Core Data & Calculate Spatial Diurnal Correction Grid
    gefs_input_grids = []
    gefs_ctrl_tmax = get_gefs_grid("c00", fxx_max, ":TMAX:2 m")
    gefs_ctrl_synop = get_gefs_grid("c00", fxx_synop, ":TMP:2 m")
    
    if gefs_ctrl_tmax is not None and gefs_ctrl_synop is not None:
        gefs_delta = np.clip(gefs_ctrl_tmax - gefs_ctrl_synop, 0, None) 
        gefs_input_grids.append(gefs_ctrl_tmax) 
    else:
        gefs_delta = 0.0

    ifs_input_grids = get_ifs_max_grids(fxx_max)
    ifs_synop_grids = get_ecmwf_synoptic_grids("ifs", fxx_synop, ":2t:")
    
    if ifs_input_grids and ifs_synop_grids:
        # Robustly calculate the difference using the ensemble mean to avoid single-member noise
        ifs_delta = np.clip(np.mean(np.stack(ifs_input_grids), axis=0) - np.mean(np.stack(ifs_synop_grids), axis=0), 0, None)
    else:
        ifs_delta = 0.0
        
    diurnal_correction_grid = (gefs_delta + ifs_delta) / 2.0

    # Parallel Fetch GEFS
    with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
        gefs_members = [f"p{str(i).zfill(2)}" for i in range(1, 31)]
        def fetch_gefs_worker(mem): return get_gefs_grid(mem, fxx_max, ":TMAX:2 m")
        results = executor.map(fetch_gefs_worker, gefs_members)
        for res in results:
            if res is not None: gefs_input_grids.append(res)

    # Fetch AIFS
    aifs_raw_results = get_ecmwf_synoptic_grids("aifs", fxx_synop, ":2t:")
    aifs_input_grids = [grid + diurnal_correction_grid for grid in aifs_raw_results] if aifs_raw_results else []

    if not gefs_input_grids or not ifs_input_grids or not aifs_input_grids:
        print(f"⚠️ Incomplete ensemble arrays. Skipping {valid_date_str}.")
        continue

    # Validation Means
    gefs_mean_f = (np.mean(np.stack(gefs_input_grids), axis=0) - 273.15) * 9/5 + 32
    ifs_mean_f = (np.mean(np.stack(ifs_input_grids), axis=0) - 273.15) * 9/5 + 32
    aifs_mean_f = (np.mean(np.stack(aifs_input_grids), axis=0) - 273.15) * 9/5 + 32

    superensemble_grids = gefs_input_grids + ifs_input_grids + aifs_input_grids
    super_matrix = np.stack(superensemble_grids)

    # Fetch NBM 
    try:
        grib_bytes = fetch_nbm_maxt_50(nbm_date_nomads, nbm_cycle, nbm_fxx)
        
        with tempfile.NamedTemporaryFile(delete=False, suffix=".grib2") as tmp:
            tmp.write(grib_bytes)
            tmp_path = tmp.name
            
        ds_nbm = xr.open_dataset(tmp_path, engine="cfgrib")
        var_key = list(ds_nbm.data_vars)[0]
        nbm_data = ds_nbm[var_key].values
        
        # Build the Delaunay triangulation ONCE on the first day to save massive CPU time
        if nbm_triangulation is None:
            nbm_lat = ds_nbm.latitude.values
            nbm_lon = np.where(ds_nbm.longitude.values > 180, ds_nbm.longitude.values - 360, ds_nbm.longitude.values)
            nbm_valid_mask = (nbm_lat >= lat_min - 3) & (nbm_lat <= lat_max + 3) & (nbm_lon >= lon_min - 3) & (nbm_lon <= lon_max + 3)
            nbm_triangulation = Delaunay(np.column_stack((nbm_lon[nbm_valid_mask], nbm_lat[nbm_valid_mask])))
            
        ds_nbm.close()
        os.remove(tmp_path)
        
        interp = LinearNDInterpolator(nbm_triangulation, nbm_data[nbm_valid_mask])
        nbm_regridded = interp(target_lons, target_lats)
        nbm_f = (nbm_regridded - 273.15) * 9/5 + 32
    except Exception as e:
        print(f"❌ Failed to parse NBM for {valid_date_str}: {e}")
        continue

    members_below_nbm = np.sum(super_matrix < nbm_regridded, axis=0)
    percentile_rank = (members_below_nbm / super_matrix.shape[0]) * 100
    percentile_rank_masked = np.ma.masked_where(np.isnan(nbm_regridded), percentile_rank)

    title_valid = f"Valid Date: {valid_date_str}"
    title_inits = f"NBM Init: {display_nbm_str} | Ensemble Init: {global_init_time}"

    # --- PLOT 1: PERCENTILES ---
    fig1 = plt.figure(figsize=(14, 9))
    ax1 = plt.axes(projection=ccrs.LambertConformal(central_longitude=-96.0, central_latitude=39.2))
    ax1.set_extent([lon_min, lon_max, lat_min, lat_max], crs=ccrs.PlateCarree())
    ax1.add_feature(cfeature.COASTLINE, linewidth=1.2); ax1.add_feature(cfeature.BORDERS, linewidth=1.2)
    ax1.add_feature(cfeature.STATES, linewidth=0.4, linestyle="--")
    
    contour1 = ax1.contourf(target_lons, target_lats, percentile_rank_masked, levels=levels, cmap=custom_percentile_cmap, transform=ccrs.PlateCarree(), extend='both')
    cbar1 = plt.colorbar(contour1, shrink=0.7, pad=0.02, ticks=[0, 10, 25, 50, 75, 90, 100])
    cbar1.set_label('NBM Percentile Rank (%)', fontsize=12)
    cbar1.ax.hlines([10, 25, 75, 90], 0, 1, colors='black', linewidth=1.5, linestyles='--')
    ax1.set_title(f"Max Temperature Percentile Rank (NBM vs Superensemble)\n{title_valid} | {title_inits}", fontsize=13, loc='left', weight='bold')
    ax1.text(1.15, -0.05, f"Generated: {run_timestamp}", transform=ax1.transAxes, ha='right', va='top', fontsize=10, color='gray')
    fig1.savefig(f"output/day_{offset}_percentile.png", bbox_inches='tight', dpi=150)

    # --- PLOT 2: VERIFICATION PANELS (Updated Color Range: -20 to 120) ---
    fig2, axs = plt.subplots(2, 2, figsize=(16, 11), subplot_kw={'projection': ccrs.LambertConformal(central_longitude=-96.0, central_latitude=39.2)})
    axs = axs.flatten()
    
    # Wider typical temperature bounds mapping -20 to 120 F
    t_levels = np.arange(-20, 122, 2)
    t_ticks = np.arange(-20, 140, 20)
    plot_data = [(gefs_mean_f, "GEFS Ensemble Mean"), (ifs_mean_f, "IFS Ensemble Mean"), (aifs_mean_f, "AIFS Ensemble Mean (Corrected)"), (nbm_f, "NBM Native QMD Median")]

    for i, (dataset, title) in enumerate(plot_data):
        axs[i].set_extent([lon_min, lon_max, lat_min, lat_max], crs=ccrs.PlateCarree())
        axs[i].add_feature(cfeature.COASTLINE, linewidth=0.8); axs[i].add_feature(cfeature.STATES, linewidth=0.4, linestyle="--")
        cf = axs[i].contourf(target_lons, target_lats, dataset, levels=t_levels, cmap='turbo', transform=ccrs.PlateCarree(), extend='both')
        axs[i].set_title(title, fontsize=12, weight='bold')

    cbar_ax = fig2.add_axes([0.15, 0.05, 0.7, 0.02])
    cbar2 = fig2.colorbar(cf, cax=cbar_ax, orientation='horizontal', ticks=t_ticks)
    cbar2.set_label('Max Temperature (°F)', fontsize=14)
    fig2.suptitle(f"Input Verification Checklist: Max Temperature Forecast Mapping\n{title_valid} | {title_inits}", fontsize=15, weight='bold', y=0.96)
    axs[3].text(1.0, -0.15, f"Generated: {run_timestamp}", transform=axs[3].transAxes, ha='right', va='top', fontsize=10, color='gray')
    fig2.savefig(f"output/day_{offset}_verification.png", bbox_inches='tight', dpi=150)

    # --- PLOT 3: 3-PANEL DIFFERENCE MAP ---
    diff_gefs, diff_ifs, diff_aifs = nbm_f - gefs_mean_f, nbm_f - ifs_mean_f, nbm_f - aifs_mean_f
    fig3, axs3 = plt.subplots(1, 3, figsize=(24, 7), subplot_kw={'projection': ccrs.LambertConformal(central_longitude=-96.0, central_latitude=39.2)})
    axs3 = axs3.flatten()
    diff_levels = np.arange(-16, 18, 2) 
    
    diff_data = [(diff_gefs, "NBM minus GEFS Mean"), (diff_ifs, "NBM minus IFS Mean"), (diff_aifs, "NBM minus AIFS Mean (Corrected)")]
    for i, (dataset, title) in enumerate(diff_data):
        axs3[i].set_extent([lon_min, lon_max, lat_min, lat_max], crs=ccrs.PlateCarree())
        axs3[i].add_feature(cfeature.COASTLINE, linewidth=0.8); axs3[i].add_feature(cfeature.STATES, linewidth=0.4, linestyle="--")
        dataset_masked = np.ma.masked_where(np.isnan(nbm_regridded), dataset)
        cf3 = axs3[i].contourf(target_lons, target_lats, dataset_masked, levels=diff_levels, cmap='RdBu_r', transform=ccrs.PlateCarree(), extend='both')
        axs3[i].set_title(title, fontsize=12, weight='bold')

    fig3.subplots_adjust(bottom=0.15)
    cbar_ax3 = fig3.add_axes([0.15, 0.05, 0.7, 0.03])
    cbar3 = fig3.colorbar(cf3, cax=cbar_ax3, orientation='horizontal')
    cbar3.set_label('Temperature Difference (°F) [Positive/Red = NBM is Warmer]', fontsize=14)
    fig3.suptitle(f"Ensemble Bias Check: NBM QMD vs Global Means\n{title_valid} | {title_inits}", fontsize=15, weight='bold', y=0.98)
    axs3[2].text(1.0, -0.25, f"Generated: {run_timestamp}", transform=axs3[2].transAxes, ha='right', va='top', fontsize=10, color='gray')
    fig3.savefig(f"output/day_{offset}_difference.png", bbox_inches='tight', dpi=150)
    plt.close('all')
    
    completed_offsets.append(offset)

# --- 6. WRITE MANIFEST FOR HTML DASHBOARD ---
with open("output/manifest.json", "w") as f:
    json.dump({
        "offsets": completed_offsets,
        "valid_dates": valid_dates_map,
        "run_timestamp": run_timestamp,
        "nbm_init": display_nbm_str,
        "global_init": global_init_time,
    }, f)
print("✅ Run Complete. Manifest written.")
