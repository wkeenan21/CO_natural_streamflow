import pandas as pd
from shapely.geometry import Polygon, MultiPolygon

def fill_missing_days(df):
    try:
        df.index.freq = 'D' # if there's missing days, this will error
        return df
    except:
        full_date_range = pd.date_range(start=df.index.min(), end=df.index.max(), freq='D')
        # Reindex to include all dates, filling missing values with NaN
        try:
            df = df.reindex(full_date_range)
            df.index.freq = 'D'
            return df
        except:
            # this might trigger if the index has duplicates
            dup_mask = df.index.duplicated(keep='first')
            df = df[~dup_mask]
            df = df.reindex(full_date_range)
            df.index.freq = 'D'
            return df

def fix_gage_series(s: pd.Series) -> pd.Series:
    def normalize_val(x):
        # Convert everything to string first
        val = str(x).strip()

        # Case 1: If contains any alphabetic characters, leave as-is
        if any(c.isalpha() for c in val):
            return val

        # Case 2: If it's numeric (digits only)
        if val.isdigit():
            if len(val) <= 7:
                return val.zfill(8)  # pad to 8 digits
            else:
                return val  # leave ≥ 9 digits as-is

        # Otherwise return as-is (covers weird cases like NaN, decimals, etc.)
        return val

    return s.apply(normalize_val).astype(str)

def to_single(geom):
    if geom.geom_type == "MultiPolygon":
        # Take the largest polygon (by area) if there are multiple parts
        geom = max(geom.geoms, key=lambda a: a.area)
    return geom

def remove_holes(geom):
    if geom.geom_type == "Polygon":
        return Polygon(geom.exterior)
    elif geom.geom_type == "MultiPolygon":
        # Apply recursively to all parts
        parts = [Polygon(p.exterior) for p in geom.geoms]
        return max(parts, key=lambda a: a.area)  # keep largest
    else:
        return geom