import geopandas as gpd




def load_csv(path):
    df = pd.read_csv(path, header=None, names=["wkt"])
    df = pd.read_csv(uploaded_file, header=None, names=["wkt"])

    df["geometry"] = df["wkt"].apply(safe_loads)

    # Convert to GeoDataFrame
    gdf = gpd.GeoDataFrame(df, geometry="geometry", crs="EPSG:3857")

    return gdf

from shapely.validation import explain_validity

def validate_geometries(gdf):
    gdf["is_valid"] = gdf.geometry.is_valid
    gdf["error_info"] = gdf.geometry.apply(
        lambda geom: explain_validity(geom) if not geom.is_valid else "Valid"
    )
    return gdf

def get_invalid(gdf):
    return gdf[gdf["is_valid"] == False]

def fix_geometries(gdf):
    gdf["fixed_geometry"] = gdf.geometry.buffer(0)
    return gdf

def validate_fixed(gdf):
    gdf["fixed_is_valid"] = gdf.fixed_geometry.is_valid
    return gdf