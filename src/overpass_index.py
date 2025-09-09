from pathlib import Path
import xarray as xr

def index_overpasses(folder, buoys, R_km=25):
    """
    Build a shortlist of SWOT granule files per buoy based on bbox overlap.

    Parameters
    ----------
    folder : str or Path
        Folder with SWOT NetCDF granules.
    buoys : dict
        Dict {buoy_name: (lon, lat)} in degrees.
    R_km : float
        Search radius [km] around each buoy.

    Returns
    -------
    dict
        {buoy_name: [Path(file1), Path(file2), ...]}
    """
    folder = Path(folder)
    deg_pad = R_km / 111.0  # rough deg/km

    candidates = {name: [] for name in buoys}
    for f in sorted(folder.glob("*.nc")):
        try:
            ds = xr.open_dataset(f)  # lazy open
            lon_min = float(ds["lon"].min())
            lon_max = float(ds["lon"].max())
            lat_min = float(ds["lat"].min())
            lat_max = float(ds["lat"].max())
            for name, (blon, blat) in buoys.items():
                if (lon_min - deg_pad <= blon <= lon_max + deg_pad) and \
                   (lat_min - deg_pad <= blat <= lat_max + deg_pad):
                    candidates[name].append(f)
            ds.close()
        except Exception as e:
            print(f"Skipping {f.name}: {e}")

    return candidates
