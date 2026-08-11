```python
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.cluster import DBSCAN
from scipy.spatial import ConvexHull


# ============================================================
# 🐦 AT2 CLUSTER COLIBRI
# Kleine Kalibrierungs-App für DBSCAN
#
# Zweck:
#   EPS und min_samples visuell kalibrieren,
#   ohne die Hauptanalyse-App zu verändern.
#
# Eingabe:
#   QuPath MASTER CSV
#
# Verwendung:
#   1. CSV laden
#   2. Image auswählen
#   3. ROI auswählen
#   4. EPS verschieben
#   5. Cluster direkt kontrollieren
# ============================================================


st.set_page_config(
    page_title="AT2 Cluster Colibri",
    page_icon="🐦",
    layout="wide"
)


st.title("🐦 AT2 Cluster Colibri")

st.markdown(
    """
    **DBSCAN-Kalibrierung für AT2-Zellcluster**

    Diese kleine App verändert **nicht** deine Hauptanalyse.
    Sie dient ausschließlich dazu, einen biologisch plausiblen
    `eps`-Wert und `min_samples` visuell festzulegen.
    """
)


# ============================================================
# CSV LADEN
# ============================================================

uploaded_file = st.file_uploader(
    "📂 QuPath MASTER-CSV laden",
    type=["csv"]
)

if uploaded_file is None:

    st.info(
        """
        Bitte deine **Positive_Centroids_MASTER.csv**
        laden.
        """
    )

    st.stop()


# ============================================================
# CSV EINLESEN
# ============================================================

try:

    df = pd.read_csv(
        uploaded_file,
        sep=None,
        engine="python"
    )

except Exception as e:

    st.error(
        f"CSV konnte nicht gelesen werden:\n{e}"
    )

    st.stop()


# ============================================================
# SPALTENNAMEN
# ============================================================

df.columns = (
    df.columns
    .astype(str)
    .str.strip()
)


required = [
    "Image",
    "ROI_ID"
]


missing = [
    col
    for col in required
    if col not in df.columns
]


if missing:

    st.error(
        "Folgende Spalten fehlen:\n\n"
        + "\n".join(missing)
    )

    st.stop()


# ============================================================
# KOORDINATEN PRÜFEN
# ============================================================

has_um = (
    "X_um" in df.columns
    and
    "Y_um" in df.columns
)


has_pixel = (
    "X_pixel" in df.columns
    and
    "Y_pixel" in df.columns
)


if not has_um and not has_pixel:

    st.error(
        """
        Keine AT2-Koordinaten gefunden.

        Benötigt werden entweder:

        X_um / Y_um

        oder:

        X_pixel / Y_pixel
        """
    )

    st.stop()


# ============================================================
# NUMERISCHE KONVERTIERUNG
# ============================================================

for col in [
    "X_um",
    "Y_um",
    "X_pixel",
    "Y_pixel",
    "PixelWidth_um",
    "PixelHeight_um",
    "ROI_Area_mm2"
]:

    if col in df.columns:

        df[col] = pd.to_numeric(
            df[col],
            errors="coerce"
        )


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.header("🐦 Colibri Kalibrierung")


# ============================================================
# IMAGE AUSWÄHLEN
# ============================================================

images = sorted(
    df["Image"]
    .dropna()
    .astype(str)
    .unique()
)


selected_image = st.sidebar.selectbox(
    "Bild / Image",
    images
)


image_df = df[
    df["Image"].astype(str)
    == selected_image
].copy()


# ============================================================
# ROI AUSWÄHLEN
# ============================================================

rois = sorted(
    image_df["ROI_ID"]
    .dropna()
    .astype(str)
    .unique()
)


selected_roi = st.sidebar.selectbox(
    "ROI",
    rois
)


roi_df = image_df[
    image_df["ROI_ID"].astype(str)
    == selected_roi
].copy()


# ============================================================
# KALIBRIERUNG
# ============================================================

if has_um:

    roi_df = roi_df.dropna(
        subset=[
            "X_um",
            "Y_um"
        ]
    )

    xy = roi_df[
        [
            "X_um",
            "Y_um"
        ]
    ].to_numpy(
        dtype=float
    )

    coordinate_mode = "X_um / Y_um"

else:

    roi_df = roi_df.dropna(
        subset=[
            "X_pixel",
            "Y_pixel"
        ]
    )

    # --------------------------------------------------------
    # Pixelkalibrierung
    # --------------------------------------------------------

    if (
        "PixelWidth_um" in roi_df.columns
        and
        "PixelHeight_um" in roi_df.columns
    ):

        px_w = roi_df[
            "PixelWidth_um"
        ].dropna()

        px_h = roi_df[
            "PixelHeight_um"
        ].dropna()

        if len(px_w) > 0:
            pixel_width = float(
                px_w.iloc[0]
            )
        else:
            pixel_width = 0.2128

        if len(px_h) > 0:
            pixel_height = float(
                px_h.iloc[0]
            )
        else:
            pixel_height = 0.2128

    else:

        pixel_width = 0.2128
        pixel_height = 0.2128


    xy_pixel = roi_df[
        [
            "X_pixel",
            "Y_pixel"
        ]
    ].to_numpy(
        dtype=float
    )


    xy = np.column_stack(
        [
            xy_pixel[:, 0] *
            pixel_width,

            xy_pixel[:, 1] *
            pixel_height
        ]
    )

    coordinate_mode = (
        "X_pixel / Y_pixel → µm"
    )


# ============================================================
# PARAMETER
# ============================================================

st.sidebar.markdown("---")
st.sidebar.subheader("🔵 DBSCAN")


eps_um = st.sidebar.slider(
    "EPS – maximaler Abstand (µm)",
    min_value=5.0,
    max_value=150.0,
    value=50.0,
    step=1.0
)


min_samples = st.sidebar.slider(
    "min_samples",
    min_value=2,
    max_value=10,
    value=3,
    step=1
)


# ============================================================
# DBSCAN
# ============================================================

if len(xy) >= 2:

    dbscan = DBSCAN(
        eps=float(eps_um),
        min_samples=int(min_samples)
    )

    labels = dbscan.fit_predict(
        xy
    )

else:

    labels = np.full(
        len(xy),
        -1
    )


# ============================================================
# CLUSTER
# ============================================================

cluster_ids = sorted(
    [
        x
        for x in np.unique(labels)
        if x != -1
    ]
)


cluster_count = len(
    cluster_ids
)


clustered_mask = (
    labels != -1
)


clustered_count = int(
    clustered_mask.sum()
)


total_count = len(
    xy
)


if total_count > 0:

    clustered_percent = (
        clustered_count
        /
        total_count
        *
        100
    )

else:

    clustered_percent = 0


# ============================================================
# CLUSTERGRÖSSEN
# ============================================================

cluster_sizes = []


for cluster_id in cluster_ids:

    size = int(
        np.sum(
            labels == cluster_id
        )
    )

    cluster_sizes.append(
        size
    )


if cluster_sizes:

    median_cluster_size = float(
        np.median(
            cluster_sizes
        )
    )

else:

    median_cluster_size = np.nan


# ============================================================
# ÜBERSCHRIFT
# ============================================================

st.subheader(
    f"🔬 {selected_image}  |  ROI {selected_roi}"
)


st.caption(
    f"Koordinaten: {coordinate_mode}"
)


# ============================================================
# KENNZAHLEN
# ============================================================

c1, c2, c3, c4, c5 = st.columns(5)


c1.metric(
    "AT2 gesamt",
    total_count
)


c2.metric(
    "AT2 im Cluster",
    clustered_count
)


c3.metric(
    "Clustered AT2",
    f"{clustered_percent:.1f} %"
)


c4.metric(
    "Cluster",
    cluster_count
)


c5.metric(
    "Median AT2 / Cluster",
    (
        f"{median_cluster_size:.1f}"
        if not np.isnan(
            median_cluster_size
        )
        else "—"
    )
)


# ============================================================
# PLOT
# ============================================================

st.markdown("---")

fig, ax = plt.subplots(
    figsize=(11, 8)
)


# ------------------------------------------------------------
# Nicht geclusterte AT2
# ------------------------------------------------------------

non_clustered = (
    labels == -1
)


if np.any(non_clustered):

    ax.scatter(
        xy[non_clustered, 0],
        xy[non_clustered, 1],
        s=25,
        alpha=0.45,
        label="nicht geclustert"
    )


# ------------------------------------------------------------
# Cluster
# ------------------------------------------------------------

for cluster_id in cluster_ids:

    mask = (
        labels == cluster_id
    )

    points = xy[mask]


    ax.scatter(
        points[:, 0],
        points[:, 1],
        s=55,
        label=f"Cluster {cluster_id + 1}"
    )


    # --------------------------------------------------------
    # Cluster-Hülle
    # --------------------------------------------------------

    if len(points) >= 3:

        try:

            hull = ConvexHull(
                points
            )

            hull_points = points[
                hull.vertices
            ]

            hull_points = np.vstack(
                [
                    hull_points,
                    hull_points[0]
                ]
            )

            ax.plot(
                hull_points[:, 0],
                hull_points[:, 1],
                linewidth=1.5
            )

        except Exception:

            pass


# ============================================================
# DARSTELLUNG
# ============================================================

ax.set_xlabel(
    "X (µm)"
)

ax.set_ylabel(
    "Y (µm)"
)

ax.set_title(
    f"DBSCAN: EPS = {eps_um:.0f} µm | "
    f"min_samples = {min_samples}"
)


ax.set_aspect(
    "equal",
    adjustable="box"
)


ax.legend(
    bbox_to_anchor=(
        1.02,
        1
    ),
    loc="upper left"
)


st.pyplot(
    fig,
    use_container_width=True
)


# ============================================================
# CLUSTERDETAILS
# ============================================================

st.markdown("---")

st.subheader(
    "🔎 Erkannte Cluster"
)


if cluster_ids:

    cluster_table = []


    for cluster_id in cluster_ids:

        mask = (
            labels == cluster_id
        )

        points = xy[mask]


        area = np.nan


        if len(points) >= 3:

            try:

                hull = ConvexHull(
                    points
                )

                area = float(
                    hull.volume
                )

            except Exception:

                pass


        cluster_table.append(
            {
                "Cluster":
                    cluster_id + 1,

                "AT2":
                    int(
                        mask.sum()
                    ),

                "Fläche_µm²":
                    area
            }
        )


    cluster_table = pd.DataFrame(
        cluster_table
    )


    st.dataframe(
        cluster_table.round(2),
        use_container_width=True,
        hide_index=True
    )

else:

    st.info(
        "Bei diesen Parametern wurde kein Cluster erkannt."
    )


# ============================================================
# AKTUELLEN PARAMETER ANZEIGEN
# ============================================================

st.markdown("---")

st.subheader(
    "📌 Aktuelle Kalibrierung"
)


st.code(
    f"""
AT2 Cluster Calibration

Image:
{selected_image}

ROI:
{selected_roi}

EPS:
{eps_um:.1f} µm

min_samples:
{min_samples}

AT2 gesamt:
{total_count}

AT2 im Cluster:
{clustered_count}

Clustered AT2:
{clustered_percent:.1f} %

Cluster:
{cluster_count}

Median AT2 / Cluster:
{
    f"{median_cluster_size:.1f}"
    if not np.isnan(median_cluster_size)
    else "—"
}
"""
)


# ============================================================
# HINWEIS
# ============================================================

st.info(
    """
    💡 **Workflow**

    1. Repräsentatives Bild auswählen.
    2. EPS langsam verändern.
    3. Prüfen, ob die markierten Gruppen deiner
       biologischen Definition eines AT2-Clusters entsprechen.
    4. Dasselbe mit mehreren repräsentativen Bildern prüfen.
    5. EPS und min_samples anschließend für die komplette
       Serie festlegen und nicht mehr pro Maus verändern.
    """
)
```
