############## GPT ATTEMPT ######################
import os
import tarfile
import gzip
import io
import datetime
import requests
import numpy as np
import rasterio
from rasterio.transform import from_origin

def download_and_convert_snodas(target_dates, output_dir):
    """
    Downloads SNODAS tar files for specified dates, extracts the daily SWE data,
    and converts the raw binary (.dat) into a projected GeoTIFF.
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # SNODAS structural constants (Unmasked/Masked CONUS Grid specs)
    # Reference: https://nsidc.org/sites/default/files/g02158-v001-userguide_1.pdf
    COLS = 6935
    ROWS = 3351
    UL_LON = -124.73333333333333
    UL_LAT = 52.87500000000000
    PIXEL_SIZE = 0.00833333333333  # 30-arc seconds (~1 km)
    
    # Define affine transformation matrix for mapping pixels to geographic coordinates
    transform = from_origin(UL_LON, UL_LAT, PIXEL_SIZE, PIXEL_SIZE)
    crs = "EPSG:4326"  # WGS 84 Lat/Lon
    
    # Base URL for NSIDC NOAA HTTP server
    base_url = "https://noaadata.apps.nsidc.org/NOAA/G02158/masked"
    
    for dt in target_dates:
        date_str = dt.strftime("%Y%m%d")
        year = dt.strftime("%Y")
        # Format month folder name matching NSIDC convention (e.g., '10_Oct', '01_Jan')
        month_folder = f"{dt.strftime('%m')}_{dt.strftime('%b')}"
        
        tar_url = f"{base_url}/{year}/{month_folder}/SNODAS_{date_str}.tar"
        print(f"\nProcessing {date_str}...")
        print(f"Fetching: {tar_url}")
        
        try:
            # 1. Stream the tar file down from the server into memory
            response = requests.get(tar_url, timeout=30)
            if response.status_code != 200:
                print(f"--> Failed to download data for {date_str} (HTTP {response.status_code})")
                continue
                
            # Open the byte stream as a tarfile
            tar_stream = io.BytesIO(response.content)
            with tarfile.open(fileobj=tar_stream, mode="r") as tar:
                # Find the internal file matching the SWE identifier (11034tS)
                swe_filename = f"us_ssmv11034tST0001TTNATS{date_str}05HP001.dat.gz"
                
                try:
                    tar_member = tar.getmember(swe_filename)
                except KeyError:
                    # Fallback check: sometimes timestamps are slightly different (e.g. 05H vs others)
                    swe_member_search = [m for m in tar.getmembers() if "11034tS" in m.name and m.name.endswith(".dat.gz")]
                    if swe_member_search:
                        tar_member = swe_member_search[0]
                        swe_filename = tar_member.name
                    else:
                        print(f"--> Could not locate SWE file inside tar archive for {date_str}")
                        continue

                    
                # 2. Extract and decompress the .gz file directly from memory
                with tar.extractfile(tar_member) as f_gz:
                    with gzip.open(f_gz, 'rb') as f_dat:
                        # SNODAS data is 16-bit signed integers, Big-Endian format ('>i2')
                        binary_data = f_dat.read()
                        grid_data = np.frombuffer(binary_data, dtype='>i2').reshape((ROWS, COLS))
                        
            # 3. Create metadata and write out to local GeoTIFF format
            # Note: SNODAS SWE data units are millimeters (mm). Negative values (-9999) represent no-data.
            out_tiff_path = os.path.join(output_dir, f"SNODAS_SWE_{date_str}.tif")

            crs_fixed = {
        'proj': 'longlat',
        'datum': 'WGS84',
        'no_defs': True
    }
            
            meta = {
                'driver': 'GTiff',
                'dtype': 'int16',
                'nodata': -9999,
                'width': COLS,
                'height': ROWS,
                'count': 1,
                'crs': crs_fixed,
                'transform': transform,
                'compress': 'lzw' # Keeps file size highly optimized
            }
            
            with rasterio.open(out_tiff_path, 'w', **meta) as dst:
                dst.write(grid_data, 1)
                
            print(f"--> Successfully saved: {out_tiff_path}")
            
        except Exception as e:
            print(f"--> An error occurred processing {date_str}: {e}")

# ==========================================
# SCALABLE CONFIGURATION BLOCK
# ==========================================

# Define your output target directory path 
# (Using raw string format `r""` to handle Windows backslash syntax safely)
destination = r"N:\Research\Kampf\Private\KeenanW\SNODAS"


# Alternative strategy to generate a sequential block scale:
start_date = datetime.date(2003, 10, 3)
days_to_extract = [start_date + datetime.timedelta(days=x) for x in range(8292)] # 90 straight days

download_and_convert_snodas(days_to_extract, destination)

