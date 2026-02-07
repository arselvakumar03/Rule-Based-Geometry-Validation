from validator import load_data, load_csv, validate_geometries, get_invalid, fix_geometries, validate_fixed
from reporter import save_error_report
from visualizer import plot_invalid

def run(path):
    print("Loading data...")
    #gdf = load_data(path)
    gdf = load_csv(path)

    print("Validating geometries...")
    gdf = validate_geometries(gdf)

    invalid = get_invalid(gdf)
    print(f"Invalid geometries found: {len(invalid)}")

    print("Saving error report...")
    save_error_report(gdf)

    print("Generating visualization...")
    plot_invalid(gdf)

    print("Attempting auto-fix...")
    gdf = fix_geometries(gdf)
    gdf = validate_fixed(gdf)

    print("Invalid after fix:", len(gdf[gdf["fixed_is_valid"] == False]))

if __name__ == "__main__":
    run("data/streets_xgen.wkt")