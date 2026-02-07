import streamlit as st
import pandas as pd
import geopandas as gpd
from shapely import wkt
from shapely.validation import explain_validity
import matplotlib.pyplot as plt
import contextily as ctx
import io
from pandas import ExcelWriter
import matplotlib.patches as mpatches
from shapely import wkt
from shapely.validation import explain_validity
import pandas as pd
from sklearn.ensemble import IsolationForest
from shapely.geometry.base import BaseGeometry




st.set_page_config(page_title="Line Validation", page_icon="📏", layout="wide")
st.sidebar.markdown("**Focus Error Type:** Invalid LINESTRING syntax (malformed WKT)")
with st.sidebar.expander("Training Instructions"):
    st.markdown(
        """
        <div style="font-size:12px; line-height:1.3;">
        To add more examples:
        <ol>
          <li>Create a <code>.wkt</code> file with one LINESTRING per line.</li>
          <li>Include both valid and invalid geometries.</li>
          <li>Upload the file here to test detection.</li>
          <li>Label new examples as valid/invalid for supervised learning.</li>
        </ol>
        </div>
        """,
        unsafe_allow_html=True
    )


st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Upload & Validate", "Visualization", "Auto-Fix"])

uploaded_file = st.sidebar.file_uploader("Upload WKT File", type=["wkt", "csv", "txt"])

# -----------------------------
# Safe WKT loader
# -----------------------------
def safe_loads(wkt_str):
    try:
        if isinstance(wkt_str, str) and wkt_str.strip():
            return wkt.loads(wkt_str.strip())
        else:
            return None
    except Exception:
        return None
    
# Error classifier
def classify_error(geom, wkt_str):
    if geom is None or not wkt_str.strip():
        return "Syntax error: malformed WKT"
    elif geom.geom_type == "LineString" and len(geom.coords) < 2:
        return "Invalid LINESTRING: requires at least 2 points"
    elif not geom.is_valid:
        return explain_validity(geom)
    else:
        return "Valid"
    


# -----------------------------
# File parsing
# -----------------------------
if uploaded_file:
    raw_text = uploaded_file.getvalue().decode("utf-8")

    wkts = []
    buffer = ""

    for line in raw_text.splitlines():
        line = line.strip()
        if not line:
            continue
        if line.startswith("LINESTRING"):
            if buffer:
                wkts.append(buffer)
            buffer = line
        else:
            buffer += " " + line

    if buffer:
        wkts.append(buffer)

    # Build DataFrame
    df = pd.DataFrame(wkts, columns=["wkt"])

    df["geometry"] = df["wkt"].apply(safe_loads)

    # Convert to GeoDataFrame
    gdf = gpd.GeoDataFrame(df, geometry="geometry", crs="EPSG:4326")

    # Add row IDs
    gdf["row_id"] = gdf.index + 1

    # Feature extraction
    gdf["num_coords"] = gdf.geometry.apply(lambda g: len(g.coords) if g else 0)
    gdf["bbox_area"] = gdf.geometry.apply(lambda g: (g.bounds[2]-g.bounds[0])*(g.bounds[3]-g.bounds[1]) if g else 0)
    gdf["length"] = gdf.geometry.apply(lambda g: g.length if g else 0)

    # Validity + error classification
    gdf["is_valid"] = gdf.geometry.apply(lambda g: g.is_valid if g else False)
    gdf["error_info"] = gdf.apply(lambda row: classify_error(row["geometry"], row["wkt"]), axis=1)




    # -----------------------------
    # Geometry property extraction
    # -----------------------------
    gdf["num_coords"] = gdf.geometry.apply(lambda g: len(g.coords) if g else 0)
    gdf["bbox_area"] = gdf.geometry.apply(
        lambda g: (g.bounds[2]-g.bounds[0])*(g.bounds[3]-g.bounds[1]) if g else 0
    )
    gdf["length"] = gdf.geometry.apply(lambda g: g.length if g else 0)
    gdf["is_valid"] = gdf.geometry.apply(lambda g: g.is_valid if g else False)

    # -----------------------------
    # Rule-based validation
    # -----------------------------
    def rule_check(g):
        if g is None or g.is_empty:
            return "Empty geometry"
        if not g.is_valid:
            return explain_validity(g)
        # coordinate range check (assuming lat/lon)
        minx, miny, maxx, maxy = g.bounds
        if not (-180 <= minx <= 180 and -90 <= miny <= 90):
            return "Invalid coordinate range"
        return "Valid"

    gdf["rule_result"] = gdf.geometry.apply(rule_check)





    features = gdf[["num_coords", "bbox_area", "length"]]
    clf = IsolationForest(contamination=0.1, random_state=42)
    clf.fit(features)
    gdf["ml_flag"] = clf.predict(features)  # -1 anomaly, 1 normal



    # -----------------------------
    # ML anomaly detection
    # -----------------------------
    from sklearn.ensemble import IsolationForest

    features = gdf[["num_coords", "bbox_area", "length"]]
    clf = IsolationForest(contamination=0.1, random_state=42)
    clf.fit(features)

    gdf["ml_flag"] = clf.predict(features)  # -1 anomaly, 1 normal

    # -----------------------------
    # Combine Rule + ML
    # -----------------------------
    def combine_results(row):
        if row["ml_flag"] == -1 and row["rule_result"] != "Valid":
            return "ML anomaly + rule error"
        elif row["ml_flag"] == -1 and row["rule_result"] == "Valid":
            return "ML flagged anomaly"
        else:
            return row["rule_result"]

    gdf["qa_result"] = gdf.apply(combine_results, axis=1)

    # Extract invalid geometries
    invalid = gdf[gdf["is_valid"] == False]

    # -----------------------------
    # Optional: Feature type classification
    # -----------------------------
    # If you have a 'feature_type' column (Road, River, Track), you can train a classifier:
    # from sklearn.ensemble import RandomForestClassifier
    # X = gdf[["num_coords", "bbox_area", "length"]]
    # y = gdf["feature_type"]   # must exist in your data
    # clf_type = RandomForestClassifier(random_state=42)
    # clf_type.fit(X, y)
    # gdf["predicted_type"] = clf_type.predict(X)

    def validate_geometry(g, gdf=None):
        if g is None or g.is_empty:
            return "Empty geometry"
        if not g.is_valid:
            return explain_validity(g)
        minx, miny, maxx, maxy = g.bounds
        if not (-180 <= minx <= 180 and -90 <= miny <= 90):
            return "Invalid coordinate range"
        if gdf is not None:
            try:
                union = gdf.unary_union
                if g.intersects(union) and not g.equals(union):
                    return "Overlap detected"
            except Exception:
                pass
        return "Valid"
    
# -----------------------------
# Page 1 — Upload & Validate
# -----------------------------
if page == "Upload & Validate":
    st.title("📏 LINESTRING Geometry Validation")
    st.write("Upload a WKT file containing LINESTRING geometries.")

    if not uploaded_file:
        st.warning("Please upload a WKT file to begin.")
    else:
        st.subheader("Validation Results")

        col1, col2, col3 = st.columns(3)
        col1.metric("Total Geometries", len(gdf))
        col2.metric("Valid Geometries", len(gdf[gdf["is_valid"]]))
        col3.metric("Invalid Geometries", len(invalid))



        #st.dataframe(gdf[["wkt", "is_valid", "error_info"]], use_container_width=True)

        st.dataframe(gdf[["row_id", "wkt", "is_valid", "error_info", "ml_flag", "qa_result"]],
             use_container_width=True)


        st.success(f"Total geometries: {len(gdf)}")
        st.error(f"Invalid geometries: {len(invalid)}")

        # Download error report
        csv = gdf[["wkt", "is_valid", "error_info"]].to_csv(index=False).encode("utf-8")
        st.download_button("📥 Download Error Report", csv, "error_report.csv", "text/csv")

# -----------------------------
# Page 2 — Visualization
# -----------------------------
elif page == "Visualization":
    st.title("🗺️ Invalid LINESTRING Visualization")

    if not uploaded_file:
        st.warning("Please upload a WKT file first.")
    else:
        fig, ax = plt.subplots(figsize=(10, 10))

        # ✅ Clean geometries before plotting
        plot_geoms = gdf[gdf.geometry.notna() & ~gdf.geometry.is_empty]

        def in_range(g):
            if g is None or g.is_empty:
                return False
            minx, miny, maxx, maxy = g.bounds
            return (-180 <= minx <= 180 and -90 <= miny <= 90)

        plot_geoms = plot_geoms[plot_geoms.geometry.apply(in_range)]

        # ✅ Plot valid geometries
        valid_geoms = plot_geoms[plot_geoms["is_valid"]]
        if valid_geoms.empty:
            st.warning("No valid geometries to display.")
        else:
            valid_geoms.plot(ax=ax, color="lightgray", linewidth=1, label="Valid")

        # ✅ Plot invalid geometries by error type
        error_colors = {
            "Syntax error: malformed WKT": "red",
            "Invalid LINESTRING: requires at least 2 points": "purple",
            "Empty geometry": "black",
            "Invalid coordinate range": "orange",
            "ML flagged anomaly": "blue",
            "ML anomaly + rule error": "cyan"
        }

        plotted_any = False
        for err_type, color in error_colors.items():
            subset = gdf[gdf["qa_result"] == err_type]
            subset = subset[subset.geometry.notna() & ~subset.geometry.is_empty]
            if not subset.empty:
                subset.plot(ax=ax, color=color, linewidth=2, label=err_type)
                plotted_any = True
                for _, row in subset.iterrows():
                    x, y = row.geometry.coords[0]
                    ax.text(x, y, f"Row {row['row_id']}", fontsize=8, color=color)

        if not plotted_any:
            st.warning("No invalid geometries to display.")

        # ✅ Legend
        ax.legend(loc="upper left", bbox_to_anchor=(0, 1), ncol=1)

        # ✅ Aspect ratio
        try:
            ax.set_aspect("equal", adjustable="box")
        except Exception:
            ax.set_aspect("auto")

        plt.title("LINESTRINGs with Error Types Highlighted")
        st.pyplot(fig)






# -----------------------------
# Page 3 — Auto-Fix
# -----------------------------
elif page == "Auto-Fix":
    st.title("🛠️ Auto-Fix LINESTRING Geometries")

    if not uploaded_file:
        st.warning("Please upload a WKT file first.")
    else:
        st.write("Attempting to fix invalid geometries using buffer(0)...")

        def fix_geom(g):
            if isinstance(g, BaseGeometry) and not g.is_empty:
                try:
                    return g.buffer(0)
                except Exception:
                    return None
            return None

        # Auto-fix invalid geometries
        gdf["fixed_geometry"] = gdf.geometry.apply(fix_geom)
        gdf["fixed_is_valid"] = gdf.fixed_geometry.apply(lambda g: g.is_valid if g else False)

        st.subheader("Fix Results")
        st.dataframe(gdf[["wkt", "is_valid", "fixed_is_valid", "error_info"]])

        # Show before/after counts clearly
        st.error(f"Invalid before fix: {len(invalid)}")
        st.success(f"Still invalid after fix: {len(gdf[gdf['fixed_is_valid'] == False])}")

        # Convert fixed geometries back to WKT strings
        fixed_df = gdf.copy()
        fixed_df["wkt_fixed"] = fixed_df.apply(
            lambda row: row["fixed_geometry"].wkt if row["fixed_geometry"] and not row["fixed_geometry"].is_empty
            else row["wkt"],  # fallback to original WKT if fix failed
            axis=1
        )

        # Join into plain text content (one WKT per line)
        txt_content = "\n".join(fixed_df["wkt_fixed"].tolist())

        # Download as TXT file
        st.download_button(
            label="📥 Download Fixed WKT File (TXT)",
            data=txt_content,
            file_name="fixed_lines.txt",
            mime="text/plain"
        )
