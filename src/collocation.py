import numpy as np
import pandas as pd
import xarray as xr

def haversine_km(lon1, lat1, lon2, lat2):
    """Great-circle distance in km between (lon1,lat1) and (lon2,lat2)."""
    R = 6371.0
    to_rad = np.pi / 180.0
    dlon = (lon2 - lon1) * to_rad
    dlat = (lat2 - lat1) * to_rad
    a = np.sin(dlat/2)**2 + np.cos(lat1*to_rad)*np.cos(lat2*to_rad) * np.sin(dlon/2)**2
    return 2 * R * np.arcsin(np.sqrt(a))

def collocate_buoy_to_file(nc_path, buoy_name, blon, blat, R_km=25, variables=None):
    """
    Find nearest SWOT pixel to buoy within radius.

    Parameters
    ----------
    nc_path : Path or str
        SWOT NetCDF file.
    buoy_name : str
        Name of buoy.
    blon, blat : float
        Buoy lon/lat [deg].
    R_km : float
        Search radius [km].
    variables : list[str]
        Variables to extract (e.g. ['ssh_karin','swh_karin']).

    Returns
    -------
    dict or None
        One-row dict with collocation info, or None if no match.
    """
    ds = xr.open_dataset(nc_path)

    deg_pad = R_km / 111.0
    ds_cut = ds.where(
        (ds["lon"] >= blon - deg_pad) & (ds["lon"] <= blon + deg_pad) &
        (ds["lat"] >= blat - deg_pad) & (ds["lat"] <= blat + deg_pad),
        drop=True
    )

    if ds_cut.dims.get("x", 0) == 0 and ds_cut.dims.get("y", 0) == 0:
        ds.close()
        return None

    d_km = xr.apply_ufunc(
        haversine_km,
        ds_cut["lon"], ds_cut["lat"],
        xr.zeros_like(ds_cut["lon"]) + blon,
        xr.zeros_like(ds_cut["lat"]) + blat,
        dask="parallelized", output_dtypes=[float]
    )

    mask = d_km <= R_km
    if mask.sum() == 0:
        ds.close()
        return None

    argmin = d_km.where(mask).argmin(dim=[dim for dim in d_km.dims])
    idx = {dim: int(argmin[dim]) for dim in argmin.dims}

    row = {
        "buoy": buoy_name,
        "file": str(nc_path),
        "pixel_lon": float(ds_cut["lon"].isel(**idx).values),
        "pixel_lat": float(ds_cut["lat"].isel(**idx).values),
        "pixel_time": pd.to_datetime(ds_cut["time"].isel(**idx).values),
        "distance_km": float(d_km.isel(**idx).values),
    }

    if variables:
        for v in variables:
            if v in ds_cut:
                val = ds_cut[v].isel(**idx).values
                row[v] = float(val) if np.isfinite(val) else None

    ds.close()
    return row

def collocate_all(candidates, buoys, R_km=25, variables=None):
    """
    Run collocation across all candidate files & buoys.

    Parameters
    ----------
    candidates : dict
        {buoy_name: [Path(file1), ...]} as from index_overpasses().
    buoys : dict
        {buoy_name: (lon, lat)}.
    R_km : float
        Search radius.
    variables : list[str]
        Variables to extract.

    Returns
    -------
    DataFrame
        Collocations with columns: buoy, file, pixel_lon, pixel_lat,
        pixel_time, distance_km, plus variables.
    """
    rows = []
    for name, (blon, blat) in buoys.items():
        for f in candidates.get(name, []):
            r = collocate_buoy_to_file(f, name, blon, blat, R_km, variables)
            if r:
                rows.append(r)
    return pd.DataFrame(rows).sort_values(["buoy", "pixel_time"])
