import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from herbie import Herbie
from scipy.interpolate import griddata
from datetime import datetime, timedelta
import concurrent.futures
import requests
import tempfile
import os
import warnings
warnings.filterwarnings('ignore')

# --- 1. DIRECTORY SETUP FOR GITHUB ACTIONS ---
# Ensures the output directory exists so headless files save correctly
os.makedirs("output", exist_ok=True)

# --- 2. AUTOMATIC SHIFT DETECTOR & CALENDAR SYNC ---
now = datetime.utcnow()

# Determine operational window based on WPC arrival schedules
if (now.hour == 1 and now.minute > 30) or (2 <= now.hour <= 12) or (now.hour == 13 and now.minute <= 30):
    print("⏰ Active Session: WPC Night Shift (18z NBM QMD / 12z Global Package)")
    base_date = now - timedelta(days=1)
    global_cycle = "12"
    nbm_cycle = "18"
    mode = "18z_package"
    date_offsets = [4, 5, 6, 7, 8] 
else:
    print("⏰ Active Session: WPC Day Shift (06z NBM QMD / 00z Global Package)")
    base_date = now - timedelta(days=1) if now.hour <= 1 else now
    global_cycle = "00"
    nbm_cycle = "06"
    mode = "06z_package"
    date_offsets = [3, 4, 5, 6, 7]

global_init_time = base_date.strftime(f'%Y-%m-%d {global_cycle}:00')
print(f"📊 Global Model Baseline Cycle: {global_init_time}")

# CONUS Domain Setup
lon_min, lon_max = -125.0, -65.0
lat_min, lat_max = 24.0, 50.0

grid_lats = np.arange(lat_min, lat_max + 0.25, 0.25)
grid_lons = np.arange(lon_min, lon_max + 0.25, 0.25)
target_lons, target_lats = np.meshgrid(grid_lons, grid_lats)

# --- 3. CUSTOM COLORMAP ENGINE (OPTION B1) ---
levels = np.arange(0, 105, 5)
base_cmap = plt.cm.RdYlBu_r
base_colors = base_cmap(np.linspace(0, 1, len(levels) - 1))
white = np.array([1.0, 1.0, 1.0, 1.0])
colors_b1 = base_colors.copy()

for i in range(len(levels) - 1):
    val = levels[i]
    if 25 <= val < 75:
        # Fade the 25th-75th percentiles (30% color, 70% white)
        colors_b1[i] = colors_b1[i] * 0.30 + white * 0.70
        colors_b1[i][3] = 1.0  
custom_percentile_cmap = mcolors.ListedColormap(colors_b1)

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
    if len(daily_grids) == len(fxx_hours):
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
                if 'number' in ds.dims: pert_grids.append(grid_data)
                else: ctrl_grids.append(grid_data)
        except Exception:
            pass 
    results = []
    if len(pert_grids) == len(fxx_hours):
        pert_max = np.max(np.stack(pert_grids), axis=0) 
        for i in range(pert_max.shape[0]): results.append(pert_max[i])
    if len(ctrl_grids) == len(fxx_hours):
        results.insert(0, np.max(np.stack(ctrl_grids), axis=0))
    return results

def get_ifs_max_grids(fxx_6hr_list):
    queries = []
    for f in fxx_6hr_list:
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
                if 'number' in ds.dims: all_perts.append(grid_data)
                else: all_ctrls.append(grid_data)
        except Exception:
            pass 
            
    results = []
    if len(all_perts) == len(queries):
        pert_max = np.max(np.stack(all_perts), axis=0) 
        for i in range(pert_max.shape[0]): results.append(pert_max[i])
    if len(all_ctrls) == len(queries):
        results.insert(0, np.max(np.stack(all_ctrls), axis=0))
    return results

# --- 5. PRODUCTION MULTI-DAY LOOP ---
for offset in date_offsets:
    valid_date = base_date + timedelta(days=offset)
    valid_date_str = valid_date.strftime('%A, %B %d, %Y')
    print(f"\n🚀 Processing Forecast Package for Valid Date: {valid_date_str}")
    
    if mode == "18z_package":
        nbm_fxx = offset * 24 + 12
        global_base_fxx = offset * 24
        fxx_list_6hr = [global_base_fxx, global_base_fxx + 6, global_base_fxx + 12, global_base_fxx + 18] 
    else:  
        nbm_fxx = offset * 24 + 24
        global_base_fxx = offset * 24
        fxx_list_6hr = [global_base_fxx + 12, global_base_fxx + 18, global_base_fxx + 24, global_base_fxx + 30] 

    # Fetch Core Data & Calculate Spatial Diurnal Correction Grid
    gefs_input_grids = []
    gefs_ctrl_tmax = get_gefs_grid("c00", fxx_list_6hr, ":TMAX:2 m")
    gefs_ctrl_synop = get_gefs_grid("c00", fxx_list_6hr, ":TMP:2 m")
    
    if gefs_ctrl_tmax is not None and gefs_ctrl_synop is not None:
        gefs_delta = np.clip(gefs_ctrl_tmax - gefs_ctrl_synop, 0, None) 
        gefs_input_grids.append(gefs_ctrl_tmax) 
    else:
        gefs_delta = 0.0

    ifs_input_grids = get_ifs_max_grids(fxx_list_6hr)
    ifs_synop_grids = get_ecmwf_synoptic_grids("ifs", fxx_list_6hr, ":2t:")
    
    if ifs_input_grids and ifs_synop_grids:
        ifs_delta = np.clip(ifs_input_grids[0] - ifs_synop_grids[0], 0, None)
    else:
        ifs_delta = 0.0
        
    diurnal_correction_grid = (gefs_delta + ifs_delta) / 2.0

    # Parallel Fetch GEFS
    with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
        gefs_members = [f"p{str(i).zfill(2)}" for i in range(1, 31)]
        def fetch_gefs_worker(mem): return get_gefs_grid(mem, fxx_list_6hr, ":TMAX:2 m")
        results = executor.map(fetch_gefs_worker, gefs_members)
        for res in results:
            if res is not None: gefs_input_grids.append(res)

    # Fetch AIFS
    aifs_raw_results = get_ecmwf_synoptic_grids("aifs", fxx_list_6hr, ":2t:")
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
        nbm_date_str = base_date.strftime('%Y%m%d')
        fxx_str = f"f{nbm_fxx:03d}"
        base_url = f"https://nomads.ncep.noaa.gov/pub/data/nccf/com/blend/prod/blend.{nbm_date_str}/{nbm_cycle}/qmd/"
        file_name = f"blend.t{nbm_cycle}z.qmd.{fxx_str}.co.grib2"
        
        r_idx = requests.get(f"{base_url}{file_name}.idx")
        r_idx.raise_for_status()
        idx_lines = r_idx.text.splitlines()
        
        target_idx = None
        for i, line in enumerate(idx_lines):
            if ":TMP:2 m above ground:" in line and "max fcst" in line and "50% level" in line:
                target_idx = i
                break
        if target_idx is None: raise ValueError("Could not locate 50th percentile MaxT grid.")
            
        start_byte = int(idx_lines[target_idx].split(':')[1])
        end_byte = int(idx_lines[target_idx + 1].split(':')[1]) - 1 if target_idx + 1 < len(idx_lines) else ""
        headers = {"Range": f"bytes={start_byte}-{end_byte}"}
        
        response = requests.get(base_url + file_name, headers=headers)
        response.raise_for_status()
        
        with tempfile.NamedTemporaryFile(delete=False, suffix=".grib2") as tmp:
            tmp.write(response.content)
            tmp_path = tmp.name
            
        ds_nbm = xr.open_dataset(tmp_path, engine="cfgrib")
        var_key = list(ds_nbm.data_vars)[0]
        nbm_lat = ds_nbm.latitude.values
        nbm_lon = np.where(ds_nbm.longitude.values > 180, ds_nbm.longitude.values - 360, ds_nbm.longitude.values)
        nbm_data = ds_nbm[var_key].values
        os.remove(tmp_path)
        
        mask = (nbm_lat >= lat_min - 3) & (nbm_lat <= lat_max + 3) & (nbm_lon >= lon_min - 3) & (nbm_lon <= lon_max + 3)
        nbm_regridded = griddata((nbm_lon[mask], nbm_lat[mask]), nbm_data[mask], (target_lons, target_lats), method='linear')
        nbm_f = (nbm_regridded - 273.15) * 9/5 + 32
    except Exception as e:
        print(f"❌ Failed to parse NBM for {valid_date_str}: {e}")
        continue

    # Computations
    members_below_nbm = np.sum(super_matrix < nbm_regridded, axis=0)
    percentile_rank = (members_below_nbm / super_matrix.shape[0]) * 100

    # --- PLOT 1: PERCENTILES WITH CUSTOM COLORMAP ---
    fig1 = plt.figure(figsize=(14, 9))
    ax1 = plt.axes(projection=ccrs.LambertConformal(central_longitude=-96.0, central_latitude=39.2))
    ax1.set_extent([lon_min, lon_max, lat_min, lat_max], crs=ccrs.PlateCarree())
    ax1.add_feature(cfeature.COASTLINE, linewidth=1.2); ax1.add_feature(cfeature.BORDERS, linewidth=1.2)
    ax1.add_feature(cfeature.STATES, linewidth=0.4, linestyle="--")
    
    contour1 = ax1.contourf(
        target_lons, target_lats, percentile_rank, 
        levels=levels, cmap=custom_percentile_cmap, transform=ccrs.PlateCarree(), extend='both'
    )
    
    cbar1 = plt.colorbar(contour1, shrink=0.7, pad=0.02, ticks=[0, 10, 25, 50, 75, 90, 100])
    cbar1.set_label('NBM Percentile Rank (%)', fontsize=12)
    cbar1.ax.hlines([10, 25, 75, 90], 0, 1, colors='black', linewidth=1.5, linestyles='--')
    ax1.set_title(f"NBM Max Temperature Percentile Rank vs Superensemble\nValid Date: {valid_date_str} | Init: {global_init_time}", fontsize=13, loc='left', weight='bold')
    
    # [CHANGE]: Headless Save Engine replacing plt.show()
    fig1.savefig(f"output/day_{offset}_percentile.png", bbox_inches='tight', dpi=150)

    # --- PLOT 2: VERIFICATION PANELS ---
    fig2, axs = plt.subplots(2, 2, figsize=(16, 11), subplot_kw={'projection': ccrs.LambertConformal(central_longitude=-96.0, central_latitude=39.2)})
    axs = axs.flatten()
    t_levels = np.arange(50, 116, 2)
    plot_data = [(gefs_mean_f, "GEFS Ensemble Mean"), (ifs_mean_f, "IFS Ensemble Mean"), (aifs_mean_f, "AIFS Ensemble Mean (Corrected)"), (nbm_f, "NBM Native QMD Median")]

    for i, (dataset, title) in enumerate(plot_data):
        axs[i].set_extent([lon_min, lon_max, lat_min, lat_max], crs=ccrs.PlateCarree())
        axs[i].add_feature(cfeature.COASTLINE, linewidth=0.8); axs[i].add_feature(cfeature.STATES, linewidth=0.4, linestyle="--")
        cf = axs[i].contourf(target_lons, target_lats, dataset, levels=t_levels, cmap='turbo', transform=ccrs.PlateCarree(), extend='both')
        axs[i].set_title(title, fontsize=12, weight='bold')

    cbar_ax = fig2.add_axes([0.15, 0.05, 0.7, 0.02])
    cbar2 = fig2.colorbar(cf, cax=cbar_ax, orientation='horizontal'); cbar2.set_label('Max Temperature (°F)', fontsize=14)
    fig2.suptitle(f"Input Verification Checklist: Max Temperature Forecast Mapping\nValid Date: {valid_date_str} | Init: {global_init_time}", fontsize=15, weight='bold', y=0.96)
    
    # [CHANGE]: Headless Save Engine replacing plt.show()
    fig2.savefig(f"output/day_{offset}_verification.png", bbox_inches='tight', dpi=150)
    
    plt.close('all')
