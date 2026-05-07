library(sf)
library(elevatr)
library(terra)

# ── Config ────────────────────────────────────────────────────────────────────
OUTPUT_PATH <- r"(C:\Users\C830645719\OneDrive - Colostate\documents\GitHub\CO_natural_streamflow\data\terrain\dem_ucol_30m.tif)"
Z_LEVEL     <- 11  # zoom level — z=11 ≈ 30 m resolution in the American West

# ── 1. Reproject GDF to WGS84 (elevatr expects EPSG:4326) ────────────────────
gdf <- read_sf(r"(C:\Users\C830645719\OneDrive - Colostate\documents\GitHub\CO_natural_streamflow\data\shapefiles\UCOL_basin\layers\globalwatershed.shp)")
gdf_wgs84 <- st_transform(gdf, crs = 4326)

bb <- st_bbox(gdf_wgs84)
cat(sprintf("Bounding box (WGS84): %.4f, %.4f, %.4f, %.4f\n",
            bb["xmin"], bb["ymin"], bb["xmax"], bb["ymax"]))

# ── 2. Download 30 m DEM via elevatr (USGS 3DEP / AWS Terrain Tiles) ─────────
# get_elev_raster() returns a RasterLayer; pass your sf object as the location
dem_raster <- get_elev_raster(
  locations = gdf_wgs84,
  z         = Z_LEVEL,   # zoom 11 ≈ 28 m/pixel near equator; ~20–30 m in CONUS
  src       = "aws",     # AWS Terrain Tiles (backed by USGS 3DEP for CONUS)
  clip      = "bbox"     # use "locations" to clip to exact polygon shapes
)

# ── 3. Convert to terra SpatRaster for modern workflows ──────────────────────
dem <- rast(dem_raster)

cat(sprintf("DEM dimensions : %d rows × %d cols\n", nrow(dem), ncol(dem)))
cat(sprintf("DEM CRS        : %s\n", crs(dem, describe = TRUE)$code))
cat(sprintf("Elevation range: %.1f – %.1f m\n", global(dem, "min")[[1]],
            global(dem, "max")[[1]]))

# ── 4. (Optional) Clip to exact polygon geometry instead of bbox ──────────────
# Uncomment to clip tightly to your actual shapes:
#
# gdf_vect <- vect(gdf_wgs84)
# dem      <- mask(crop(dem, gdf_vect), gdf_vect)

# ── 5. Save to GeoTIFF ────────────────────────────────────────────────────────
writeRaster(dem, OUTPUT_PATH, overwrite = TRUE, gdal = "COMPRESS=LZW")
cat(sprintf("Saved → %s\n", normalizePath(OUTPUT_PATH)))