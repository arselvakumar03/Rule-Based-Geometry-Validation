# 🗺️ Geometry Validation Web App (WKT QA System)

This project provides a **rule‑based geometry validation system** for vector data stored in **WKT format**.  
It detects invalid geometries (e.g., self‑intersections, duplicate points, zero‑length segments), highlights them visually, and optionally auto‑fixes them using Shapely’s `buffer(0)` trick.

Built with **Streamlit**, **GeoPandas**, and **Shapely**, this app is designed for hackathon demos and quick QA automation.

---

## 🚀 Features
- Upload `.wkt` or `.csv` files containing geometries (LINESTRING, POLYGON, MULTIPOLYGON).
- Validate geometries and report errors.
- Visualize invalid geometries (highlighted in red).
- Auto‑fix invalid geometries (optional).
- Download error reports and corrected WKT files.

---


---

## ⚙️ Installation

Clone the repository and install dependencies:

```bash
git clone <your-repo-url>
cd project
pip install -r requirements.txt


## Run the project in streamlit

streamlit run app.py
