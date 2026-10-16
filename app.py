"""Interface Streamlit de Computer Vision — Panorama Studio."""

from __future__ import annotations

import hashlib
from typing import Any

import cv2
import numpy as np
import pandas as pd
import streamlit as st

from src.comparison import PipelineResult, run_pipeline
from src.detectors import ALGORITHMS
from src.visualization import bgr_to_rgb, draw_inlier_matches, draw_keypoints, draw_matches


st.set_page_config(page_title="Panorama Studio", page_icon="🧭", layout="wide")
st.markdown(
    """
    <style>
      .stApp {background: linear-gradient(145deg, #07111f 0%, #0d1b2a 48%, #101d2b 100%);}
      [data-testid="stSidebar"] {background: #091522; border-right: 1px solid #21354a;}
      .hero {padding: 1.4rem 1.6rem; border: 1px solid #27435c; border-radius: 18px;
             background: linear-gradient(120deg, rgba(19,56,78,.8), rgba(8,20,34,.8)); margin-bottom: 1rem;}
      .hero h1 {margin: 0; font-size: 2.25rem; color: #f3f7fb;}
      .hero p {margin: .45rem 0 0; color: #a9c2d8;}
      div[data-testid="stMetric"] {background: #102437; border: 1px solid #29465e; padding: .7rem; border-radius: 12px;}
      .small-note {color:#98aec1; font-size:.9rem;}
    </style>
    <div class="hero">
      <h1>Computer Vision — Panorama Studio</h1>
      <p>Explorez le pipeline complet et comparez SIFT, FAST + BRIEF et ORB.</p>
    </div>
    """,
    unsafe_allow_html=True,
)


def decode_upload(upload, max_dimension: int) -> np.ndarray:
    raw = np.frombuffer(upload.getvalue(), dtype=np.uint8)
    image = cv2.imdecode(raw, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"{upload.name} n'est pas une image JPG/PNG valide.")
    height, width = image.shape[:2]
    if max_dimension > 0 and max(height, width) > max_dimension:
        scale = max_dimension / max(height, width)
        image = cv2.resize(image, (round(width * scale), round(height * scale)), interpolation=cv2.INTER_AREA)
    return image


def algorithm_controls(algorithm: str) -> dict[str, Any]:
    params: dict[str, Any] = {}
    if algorithm == "SIFT":
        params["nfeatures"] = st.number_input("Nombre maximal (0 = illimité)", 0, 10000, 0, 250)
        params["contrast_threshold"] = st.slider("Seuil de contraste", 0.001, 0.15, 0.04, 0.001)
        params["edge_threshold"] = st.slider("Seuil des contours", 2, 30, 10)
        params["sigma"] = st.slider("Sigma initial", 0.8, 2.5, 1.6, 0.1)
    elif algorithm == "FAST + BRIEF":
        params["fast_threshold"] = st.slider("Seuil FAST", 1, 100, 20)
        params["nonmax_suppression"] = st.checkbox("Suppression des non-maxima", True)
        params["max_features"] = st.number_input("Points transmis à BRIEF", 100, 10000, 3000, 100)
        params["brief_bytes"] = st.selectbox("Taille BRIEF (octets)", [16, 32, 64], index=1)
        params["brief_orientation"] = st.checkbox("BRIEF orienté (expérimental)", False)
    else:
        params["nfeatures"] = st.number_input("Nombre maximal de points", 100, 10000, 2000, 100)
        params["scale_factor"] = st.slider("Facteur de pyramide", 1.05, 2.0, 1.2, 0.05)
        params["nlevels"] = st.slider("Niveaux de pyramide", 2, 16, 8)
        params["wta_k"] = st.selectbox("WTA_K", [2, 3, 4], index=0)
        params["fast_threshold"] = st.slider("Seuil FAST interne", 1, 100, 20)
    return params


def default_params(algorithm: str) -> dict[str, Any]:
    if algorithm == "SIFT":
        return {"nfeatures": 0, "contrast_threshold": 0.04, "edge_threshold": 10, "sigma": 1.6}
    if algorithm == "FAST + BRIEF":
        return {"fast_threshold": 20, "nonmax_suppression": True, "max_features": 3000, "brief_bytes": 32}
    return {"nfeatures": 2000, "scale_factor": 1.2, "nlevels": 8, "wta_k": 2, "fast_threshold": 20}


def show_error(result: PipelineResult) -> None:
    if result.error:
        st.error(result.error)


with st.sidebar:
    st.header("Réglages du pipeline")
    algorithm = st.selectbox("Algorithme", ALGORITHMS)
    with st.expander("Paramètres de détection", expanded=True):
        detector_params = algorithm_controls(algorithm)
    ratio = st.slider("Ratio de Lowe", 0.50, 0.95, 0.75, 0.01)
    ransac_threshold = st.slider("Seuil RANSAC (pixels)", 0.5, 12.0, 4.0, 0.5)
    blending = st.selectbox("Méthode de fusion", ["Feather", "Moyenne"])
    max_dimension = st.select_slider(
        "Dimension maximale",
        options=[0, 800, 1200, 1600, 2400, 3200],
        value=1600,
        format_func=lambda value: "Taille originale" if value == 0 else f"{value} px",
    )
    run_clicked = st.button("🚀 Lancer le traitement", type="primary", width="stretch")
    st.caption("Pour plusieurs images, chargez-les de gauche à droite avec un recouvrement entre chaque paire.")

uploads = st.file_uploader(
    "Chargez au moins deux photos qui se chevauchent",
    type=["jpg", "jpeg", "png"],
    accept_multiple_files=True,
)

images: list[np.ndarray] = []
image_signature = None
if uploads:
    try:
        images = [decode_upload(upload, max_dimension) for upload in uploads]
        digest = hashlib.sha1()
        for upload in uploads:
            digest.update(upload.name.encode("utf-8", errors="replace"))
            digest.update(upload.getvalue())
        image_signature = (digest.hexdigest(), max_dimension)
    except ValueError as exc:
        st.error(str(exc))

pipeline_context = (
    image_signature,
    algorithm,
    repr(sorted(detector_params.items())),
    ratio,
    ransac_threshold,
    blending,
)

if run_clicked:
    if len(images) < 2:
        st.warning("Chargez au moins deux images avant de lancer le traitement.")
    else:
        with st.spinner(f"Pipeline {algorithm} en cours…"):
            st.session_state["pipeline"] = run_pipeline(
                images, algorithm, detector_params, ratio, ransac_threshold, blending
            )
            st.session_state["pipeline_context"] = pipeline_context

tabs = st.tabs(
    [
        "Images originales",
        "1 · Points d’intérêt",
        "2 · Matching",
        "3 · Homographie",
        "4 · Panorama",
        "Comparaison SIFT vs FAST + BRIEF vs ORB",
        "Comprendre les algorithmes",
    ]
)

result: PipelineResult | None = (
    st.session_state.get("pipeline")
    if st.session_state.get("pipeline_context") == pipeline_context
    else None
)

with tabs[0]:
    if not images:
        st.info("Les images chargées apparaîtront ici.")
    else:
        columns = st.columns(min(3, len(images)))
        for index, (upload, image) in enumerate(zip(uploads, images)):
            with columns[index % len(columns)]:
                st.image(bgr_to_rgb(image), caption=f"{index + 1}. {upload.name} — {image.shape[1]}×{image.shape[0]}")

with tabs[1]:
    if result is None:
        st.info("Lancez le traitement pour afficher les points d’intérêt.")
    else:
        show_error(result)
        for index, feature in enumerate(result.features):
            st.subheader(f"Image {index + 1}")
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Points détectés", feature.detected_count)
            c2.metric("Points décrits", feature.descriptor_count)
            c3.metric("Dimension", feature.descriptor_size or "—")
            c4.metric("Temps", f"{feature.elapsed_ms:.1f} ms")
            st.image(bgr_to_rgb(draw_keypoints(images[index], feature.keypoints)), width="stretch")
            if feature.descriptors is not None:
                label = "128 valeurs flottantes" if result.algorithm == "SIFT" else "valeurs binaires compactées en octets"
                st.caption(f"Aperçu des descripteurs ({label})")
                st.dataframe(pd.DataFrame(feature.descriptors[:5]), width="stretch", height=210)

with tabs[2]:
    if result is None or not result.pairs:
        st.info("Aucun résultat de matching disponible.")
    else:
        show_error(result)
        for pair in result.pairs:
            st.subheader(f"Image {pair.index_a + 1} → image {pair.index_b + 1}")
            c1, c2, c3 = st.columns(3)
            c1.metric("Requêtes KNN", pair.matching.query_count)
            c2.metric("Bons matches", len(pair.matching.good_matches))
            c3.metric("Temps", f"{pair.matching.elapsed_ms:.1f} ms")
            view = draw_matches(
                images[pair.index_a], result.features[pair.index_a].keypoints,
                images[pair.index_b], result.features[pair.index_b].keypoints,
                pair.matching.good_matches,
            )
            st.image(bgr_to_rgb(view), caption="Les 120 meilleures correspondances après le ratio de Lowe", width="stretch")

with tabs[3]:
    if result is None or not result.pairs:
        st.info("Aucune homographie disponible.")
    else:
        show_error(result)
        for pair in result.pairs:
            homography = pair.homography
            st.subheader(f"Image {pair.index_a + 1} → image {pair.index_b + 1}")
            c1, c2, c3 = st.columns(3)
            c1.metric("Inliers", homography.inliers)
            c2.metric("Outliers", homography.outliers)
            c3.metric("Taux d’inliers", f"{homography.inlier_ratio:.1f} %")
            if homography.matrix is not None:
                st.code(np.array2string(homography.matrix, precision=5, suppress_small=True), language="text")
                view = draw_inlier_matches(
                    images[pair.index_a], result.features[pair.index_a].keypoints,
                    images[pair.index_b], result.features[pair.index_b].keypoints,
                    pair.matching.good_matches, homography.inlier_mask,
                )
                st.image(bgr_to_rgb(view), caption="Correspondances conservées par RANSAC", width="stretch")
            elif homography.error:
                st.warning(homography.error)

with tabs[4]:
    if result is None:
        st.info("Le panorama final apparaîtra ici.")
    elif result.success and result.stitch is not None:
        panorama = result.stitch.panorama
        st.success(f"Panorama généré avec {result.algorithm} en {result.total_ms:.1f} ms.")
        st.image(bgr_to_rgb(panorama), width="stretch")
        ok, encoded = cv2.imencode(".png", panorama)
        if ok:
            st.download_button(
                "⬇️ Télécharger le panorama PNG",
                data=encoded.tobytes(),
                file_name=f"panorama_{result.algorithm.lower().replace(' + ', '_').replace(' ', '_')}.png",
                mime="image/png",
            )
    else:
        show_error(result)

with tabs[5]:
    st.markdown("Chaque méthode exécute son propre pipeline sur les mêmes pixels et les mêmes seuils de matching/RANSAC.")
    compare_clicked = st.button("Comparer les 3 méthodes", type="primary", disabled=len(images) < 2)
    if compare_clicked:
        comparison_results = {}
        progress = st.progress(0, text="Comparaison en cours…")
        for index, name in enumerate(ALGORITHMS):
            comparison_results[name] = run_pipeline(
                images, name, default_params(name), ratio, ransac_threshold, blending
            )
            progress.progress((index + 1) / len(ALGORITHMS), text=f"{name} terminé")
        progress.empty()
        st.session_state["comparison"] = comparison_results
        st.session_state["comparison_context"] = (image_signature, ratio, ransac_threshold, blending)

    comparison_context = (image_signature, ratio, ransac_threshold, blending)
    comparison: dict[str, PipelineResult] | None = (
        st.session_state.get("comparison")
        if st.session_state.get("comparison_context") == comparison_context
        else None
    )
    if comparison:
        metrics = {name: item.metrics() for name, item in comparison.items()}
        table = pd.DataFrame(metrics)
        st.dataframe(
            table.style.format(
                lambda value: f"{value:.2f}" if isinstance(value, float) else value
            ),
            width="stretch",
        )
        chart_specs = {
            "Temps d’exécution (ms)": ["Temps de détection (ms)", "Temps de matching (ms)", "Temps total (ms)"],
            "Volumes de données": ["Keypoints détectés", "Bons matches", "Inliers RANSAC"],
        }
        chart_columns = st.columns(2)
        for column, (title, rows) in zip(chart_columns, chart_specs.items()):
            with column:
                st.subheader(title)
                chart_data = table.loc[rows].T.apply(pd.to_numeric, errors="coerce")
                st.bar_chart(chart_data)
        st.subheader("Panoramas et diagnostics")
        panorama_columns = st.columns(3)
        for column, name in zip(panorama_columns, ALGORITHMS):
            item = comparison[name]
            with column:
                st.markdown(f"#### {name}")
                if item.success:
                    st.image(bgr_to_rgb(item.stitch.panorama), width="stretch")
                else:
                    st.error(item.error or "Échec non déterminé")
        st.info("Un plus grand nombre de points ne garantit pas un meilleur panorama : la géométrie, la répartition des inliers et la parallaxe comptent aussi.")

with tabs[6]:
    st.warning("Les schémas ci-dessous sont des simulations pédagogiques simplifiées ; ils ne reproduisent pas exactement les calculs internes d’OpenCV.")
    sift_tab, fast_tab, orb_tab = st.tabs(["SIFT", "FAST", "ORB"])
    with sift_tab:
        st.markdown(
            """SIFT construit un **espace d’échelles gaussien**, soustrait des niveaux voisins (Difference of Gaussians),
            puis cherche les extrema 3D. Une orientation est attribuée grâce aux gradients $I_x$ et $I_y$ :
            magnitude $\\sqrt{I_x^2+I_y^2}$ et angle $\\operatorname{atan2}(I_y,I_x)$. Les histogrammes locaux
            forment finalement un descripteur de **128 valeurs** (4×4 cellules × 8 orientations)."""
        )
        demo = np.zeros((160, 240), np.uint8)
        cv2.circle(demo, (75, 80), 35, 230, -1)
        cv2.rectangle(demo, (145, 40), (210, 120), 160, -1)
        levels = [cv2.GaussianBlur(demo, (0, 0), sigma) for sigma in (1.0, 2.0, 4.0)]
        dog = cv2.normalize(cv2.subtract(levels[0], levels[1]), None, 0, 255, cv2.NORM_MINMAX)
        st.image(np.hstack(levels + [dog]), caption="Trois échelles gaussiennes, puis une DoG simplifiée", width="stretch")
    with fast_tab:
        st.markdown(
            """FAST examine un cercle de **16 pixels** autour d’un candidat. Un arc contigu nettement plus clair ou plus sombre
            que le centre indique un coin. Le seuil règle cette différence d’intensité. FAST ne produit aucun vecteur :
            **BRIEF est donc nécessaire** pour décrire puis comparer les points."""
        )
        fast_demo = np.full((260, 260, 3), 24, np.uint8)
        center = np.array([130, 130])
        for number, angle in enumerate(np.linspace(0, 2 * np.pi, 16, endpoint=False), 1):
            point = center + np.rint(82 * np.array([np.cos(angle), np.sin(angle)])).astype(int)
            color = (70, 220, 255) if number <= 10 else (100, 100, 110)
            cv2.circle(fast_demo, tuple(point), 10, color, -1)
            cv2.putText(fast_demo, str(number), tuple(point + [-7, 5]), cv2.FONT_HERSHEY_SIMPLEX, .35, (5, 12, 18), 1)
        cv2.circle(fast_demo, tuple(center), 12, (80, 120, 255), -1)
        st.image(bgr_to_rgb(fast_demo), caption="Cercle FAST simplifié : un arc de pixels satisfait le test", width=360)
    with orb_tab:
        st.markdown(
            """ORB applique FAST dans une **pyramide d’images**, classe les coins et estime leur orientation par centroïde
            d’intensité. Il fait pivoter ses paires de tests BRIEF (rBRIEF), ce qui améliore l’invariance à la rotation.
            Les chaînes binaires se comparent rapidement par distance de **Hamming**."""
        )
        rng = np.random.default_rng(7)
        pairs = rng.integers(25, 215, size=(14, 4))
        brief_demo = np.full((240, 240, 3), 17, np.uint8)
        for x1, y1, x2, y2 in pairs:
            cv2.line(brief_demo, (x1, y1), (x2, y2), (80, 145, 235), 1)
            cv2.circle(brief_demo, (x1, y1), 3, (65, 230, 170), -1)
            cv2.circle(brief_demo, (x2, y2), 3, (245, 185, 70), -1)
        st.image(bgr_to_rgb(brief_demo), caption="Paires de comparaison binaires BRIEF simplifiées", width=360)

st.markdown(
    '<p class="small-note">Limites : faible texture ou recouvrement, objets mobiles, changements d’exposition et parallaxe importante peuvent empêcher une homographie globale fiable.</p>',
    unsafe_allow_html=True,
)
