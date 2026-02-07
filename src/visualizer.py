import matplotlib.pyplot as plt

def plot_invalid(gdf, output_path="output/invalid_plot.png"):
    fig, ax = plt.subplots(figsize=(10, 10))

    # Plot all geometries
    gdf.plot(ax=ax, color="lightgray", edgecolor="black")

    # Highlight invalid ones
    invalid = gdf[gdf["is_valid"] == False]
    invalid.plot(ax=ax, color="red", edgecolor="black")

    plt.title("Invalid Polygons Highlighted in Red")
    plt.savefig(output_path)
    plt.close()