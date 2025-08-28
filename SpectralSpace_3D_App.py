import streamlit as st
import os
import pickle
import numpy as np
import pandas as pd
import re
from tqdm import tqdm
from sklearn.neighbors import NearestNeighbors
from scipy.interpolate import interp1d
import plotly.express as px
import plotly.graph_objects as go

# ------------------------
# Utility Functions
# ------------------------
def sanitize_filename(filename):
    invalid_chars = r'[<>:"/\\|?*]'
    return re.sub(invalid_chars, '_', filename)

def load_model(model_path):
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    return model

def extract_molecule_formula(header):
    pattern = r"molecules=['\"]([^,'\"]+)"
    match = re.search(pattern, header)
    if match:
        formula = match.group(1)
        if ',' in formula:
            formula = formula.split(',')[0]
        return formula
    return "Unknown"

def load_and_interpolate_spectrum(filepath, reference_frequencies):
    with open(filepath, 'r') as f:
        lines = f.readlines()

    first_line = lines[0].strip()
    formula = "Unknown"
    param_dict = {}
    data_start_line = 0

    # Case 1: molecule header
    if first_line.startswith('//') and 'molecules=' in first_line:
        header = first_line[2:].strip()
        formula = extract_molecule_formula(header)
        for part in header.split():
            if '=' in part:
                try:
                    key, value = part.split('=')
                    key = key.strip()
                    value = value.strip("'")
                    if key in ['molecules', 'sourcesize']:
                        continue
                    try:
                        param_dict[key] = float(value)
                    except ValueError:
                        param_dict[key] = value
                except:
                    continue
        data_start_line = 1

    # Case 2: column header
    elif first_line.startswith('!') or first_line.startswith('#'):
        if 'molecules=' in first_line:
            formula = extract_molecule_formula(first_line)
        data_start_line = 1

    # Case 3: no header
    else:
        data_start_line = 0
        formula = os.path.basename(filepath).split('.')[0]

    spectrum_data = []
    for line in lines[data_start_line:]:
        line = line.strip()
        if not line or line.startswith('!') or line.startswith('#'):
            continue
        try:
            parts = line.split()
            if len(parts) >= 2:
                try:
                    freq = float(parts[0])
                    intensity = float(parts[1])
                except ValueError:
                    freq = float(parts[0].replace('D','E').replace('d','E'))
                    intensity = float(parts[1].replace('D','E').replace('d','E'))
                if np.isfinite(freq) and np.isfinite(intensity):
                    spectrum_data.append([freq, intensity])
        except:
            continue

    if not spectrum_data:
        raise ValueError("No valid data points found")

    spectrum_data = np.array(spectrum_data)

    # Convert GHz → Hz if needed
    if np.max(spectrum_data[:,0]) < 1e11:
        spectrum_data[:,0] *= 1e9

    interpolator = interp1d(spectrum_data[:,0], spectrum_data[:,1],
                            kind='linear', bounds_error=False, fill_value=0.0)
    interpolated = interpolator(reference_frequencies)

    params = [
        param_dict.get('logn', np.nan),
        param_dict.get('tex', np.nan),
        param_dict.get('velo', np.nan),
        param_dict.get('fwhm', np.nan)
    ]
    return spectrum_data, interpolated, formula, params, os.path.basename(filepath)

def find_knn_neighbors(training_embeddings, new_embeddings, k=5):
    if len(training_embeddings) == 0 or len(new_embeddings) == 0:
        return []
    k = min(k, len(training_embeddings))
    knn = NearestNeighbors(n_neighbors=k, metric='euclidean')
    knn.fit(training_embeddings)
    all_neighbor_indices = []
    for new_embedding in new_embeddings:
        distances, indices = knn.kneighbors([new_embedding])
        valid_indices = [idx for idx in indices[0] if idx < len(training_embeddings)]
        all_neighbor_indices.append(valid_indices)
    return all_neighbor_indices

# ------------------------
# Streamlit UI
# ------------------------
st.set_page_config(layout="wide", page_title="3D Spectra UMAP Explorer")

st.title("🔬 Molecular Spectra Explorer with PCA + UMAP")

# Upload model
model_file = st.file_uploader("Upload trained PCA+UMAP model (.pkl)", type="pkl")
if model_file:
    model = pickle.load(model_file)
    st.success("Model loaded successfully ✅")

    scaler = model['scaler']
    pca = model['pca']
    umap_model = model['umap']
    ref_freqs = model['reference_frequencies']

    # Upload spectra
    spectra_files = st.file_uploader("Upload spectrum files (.txt)", type="txt", accept_multiple_files=True)
    if spectra_files:
        new_embeddings, new_params, new_formulas, new_filenames = [], [], [], []
        spectra_dict = {}
        for uploaded in spectra_files:
            filepath = uploaded.name
            try:
                content = uploaded.read().decode("utf-8").splitlines()
                with open("tmp.txt","w") as f: f.write("\n".join(content))
                spectrum_data, interpolated, formula, params, filename = load_and_interpolate_spectrum("tmp.txt", ref_freqs)
                X_scaled = scaler.transform([interpolated])
                X_pca = pca.transform(X_scaled)
                X_umap = umap_model.transform(X_pca)
                new_embeddings.append(X_umap[0])
                new_params.append(params)
                new_formulas.append(formula)
                new_filenames.append(filename)
                spectra_dict[filename] = spectrum_data
            except Exception as e:
                st.error(f"Error processing {filepath}: {e}")

        if new_embeddings:
            new_embeddings = np.array(new_embeddings)
            new_params = np.array(new_params)
            knn_indices = find_knn_neighbors(model['embedding'], new_embeddings, k=5)

            # Interactive 3D Plot by formula
            df_plot = pd.DataFrame({
                "UMAP1": model['embedding'][:,0],
                "UMAP2": model['embedding'][:,1],
                "UMAP3": model['embedding'][:,2],
                "Formula": model['formulas']
            })
            fig = px.scatter_3d(df_plot, x="UMAP1", y="UMAP2", z="UMAP3",
                                color="Formula", opacity=0.4, size_max=5,
                                title="Training Data")

            # Add new points
            df_new = pd.DataFrame({
                "UMAP1": new_embeddings[:,0],
                "UMAP2": new_embeddings[:,1],
                "UMAP3": new_embeddings[:,2],
                "Formula": new_formulas
            })
            fig.add_trace(go.Scatter3d(
                x=df_new["UMAP1"], y=df_new["UMAP2"], z=df_new["UMAP3"],
                mode='markers+text',
                text=new_filenames,
                marker=dict(size=8, color="red", symbol="diamond"),
                name="New Spectra"
            ))
            st.plotly_chart(fig, use_container_width=True)

            # Show KNN tables
            for i, neighbors in enumerate(knn_indices):
                st.subheader(f"📊 Nearest Neighbors for {new_filenames[i]} ({new_formulas[i]})")
                if neighbors:
                    neighbor_data = []
                    for idx in neighbors:
                        neighbor_data.append({
                            "Formula": model['formulas'][idx],
                            "logn": model['y'][idx,0],
                            "Tex": model['y'][idx,1],
                            "Velocity": model['y'][idx,2],
                            "FWHM": model['y'][idx,3]
                        })
                    st.dataframe(pd.DataFrame(neighbor_data))
                else:
                    st.info("No neighbors found")
