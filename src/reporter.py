def save_error_report(gdf, path="output/error_report.csv"):
    invalid = gdf[gdf["is_valid"] == False]
    invalid[["error_info"]].to_csv(path)