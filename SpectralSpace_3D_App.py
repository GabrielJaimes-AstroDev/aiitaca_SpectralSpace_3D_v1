import streamlit as st
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os
import re
from scipy.interpolate import interp1d
from sklearn.neighbors import NearestNeighbors
from io import BytesIO
import base64
import plotly.express as px

# Set page configuration
st.set_page_config(
    page_title="C.3D Spectral Space Analyzer",
    page_icon="🔭",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1E88E5;
        text-align: center;
        margin-bottom: 2rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #0D47A1;
        margin-bottom: 1rem;
        margin-top: 1.5rem;
    }
    .info-box {
        background-color: #E3F2FD;
        padding: 1rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
    }
    .stButton>button {
        background-color: #1E88E5;
        color: white;
    }
    .main-title {
        font-size: 1.8rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 1rem;
        font-weight: bold;
    }
    .plot-container {
        background-color: #FAFAFA;
        padding: 1.5rem;
        border-radius: 0.5rem;
        margin-bottom: 1.5rem;
    }
</style>
""", unsafe_allow_html=True)

if 'model' not in st.session_state:
    st.session_state.model = None
if 'spectra_files' not in st.session_state:
    st.session_state.spectra_files = None
if 'results' not in st.session_state:
    st.session_state.results = None

def load_model(model_file):
    try:
        model = pickle.load(model_file)
        return model
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None

def sanitize_filename(filename):
    invalid_chars = r'[<>:"/\\|?*]'
    return re.sub(invalid_chars, '_', filename)

def extract_molecule_formula(header):
    pattern = r"molecules=['\"]([^,'\"]+)"
    match = re.search(pattern, header)
    if match:
        formula = match.group(1)
        if ',' in formula:
            formula = formula.split(',')[0]
        return formula
    return "Unknown"

def process_uploaded_spectrum(file, reference_frequencies):
    try:
        content = file.getvalue().decode("utf-8")
        lines = content.split('\n')
        
        first_line = lines[0].strip()
        second_line = lines[1].strip() if len(lines) > 1 else ""
        
        formula = "Unknown"
        param_dict = {}
        data_start_line = 0
        
        # Format 1
        if first_line.startswith('//') and 'molecules=' in first_line:
            header = first_line[2:].strip()  # Remove the '//'
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
        
        # Format 2
        elif first_line.startswith('!') or first_line.startswith('#'):
            if 'molecules=' in first_line:
                formula = extract_molecule_formula(first_line)
            data_start_line = 1
        
        # Format 3
        else:
            data_start_line = 0
            formula = file.name.split('.')[0] 

        spectrum_data = []
        for line in lines[data_start_line:]:
            line = line.strip()
            # Saltar líneas de comentario o vacías
            if not line or line.startswith('!') or line.startswith('#'):
                continue
                
            try:
                parts = line.split()
                if len(parts) >= 2:
                    # Intentar diferentes formatos de números
                    try:
                        freq = float(parts[0])
                        intensity = float(parts[1])
                    except ValueError:
                        # Intentar con notación científica que pueda tener D instead of E
                        freq_str = parts[0].replace('D', 'E').replace('d', 'E')
                        intensity_str = parts[1].replace('D', 'E').replace('d', 'E')
                        freq = float(freq_str)
                        intensity = float(intensity_str)
                    
                    if np.isfinite(freq) and np.isfinite(intensity):
                        spectrum_data.append([freq, intensity])
            except Exception as e:
                st.warning(f"Could not parse line '{line}': {e}")
                continue

        if not spectrum_data:
            st.error("No valid data points found in spectrum file")
            return None, None, None, None, None

        spectrum_data = np.array(spectrum_data)

        if np.max(spectrum_data[:, 0]) < 1e11: 
            spectrum_data[:, 0] = spectrum_data[:, 0] * 1e9  # Convertir GHz to Hz
            st.info(f"Converted frequencies from GHz to Hz for {file.name}")

        interpolator = interp1d(spectrum_data[:, 0], spectrum_data[:, 1],
                                kind='linear', bounds_error=False, fill_value=0.0)
        interpolated = interpolator(reference_frequencies)

        params = [
            param_dict.get('logn', np.nan),
            param_dict.get('tex', np.nan),
            param_dict.get('velo', np.nan),
            param_dict.get('fwhm', np.nan)
        ]

        return spectrum_data, interpolated, formula, params, file.name
        
    except Exception as e:
        st.error(f"Error processing spectrum file: {str(e)}")
        return None, None, None, None, None

def find_knn_neighbors(training_embeddings, new_embeddings, k=5):
    if len(training_embeddings) == 0 or len(new_embeddings) == 0:
        return []
    
    k = min(k, len(training_embeddings))
    
    knn = NearestNeighbors(n_neighbors=k, metric='euclidean')
    knn.fit(training_embeddings)
    
    all_neighbor_indices = []
    for new_embedding in new_embeddings:
        distances, indices = knn.kneighbors([new_embedding])
        # Verificar que los índices estén dentro del rango válido
        valid_indices = [idx for idx in indices[0] if idx < len(training_embeddings)]
        all_neighbor_indices.append(valid_indices)
    
    return all_neighbor_indices

def analyze_spectra(model, spectra_files, knn_neighbors):
    new_spectra_data = []
    new_formulas = []
    new_params = []
    new_filenames = []
    new_embeddings = []
    new_pca_components = []
    
    for file in spectra_files:
        spectrum_data, interpolated, formula, params, filename = process_uploaded_spectrum(
            file, model['reference_frequencies'])
        
        if interpolated is not None:
            # Transform the spectrum
            scaler = model['scaler']
            pca = model['pca']
            umap_model = model['umap']
            
            X_scaled = scaler.transform([interpolated])
            X_pca = pca.transform(X_scaled)
            X_umap = umap_model.transform(X_pca)
            
            new_spectra_data.append(interpolated)
            new_formulas.append(formula)
            new_params.append(params)
            new_filenames.append(filename)
            new_embeddings.append(X_umap[0])
            new_pca_components.append(X_pca[0])
    
    if len(new_embeddings) == 0:
        return None
    
    new_embeddings = np.array(new_embeddings)
    new_params = np.array(new_params)
    new_formulas = np.array(new_formulas)
    new_pca_components = np.array(new_pca_components)
    
    # Find KNN neighbors
    knn_indices = find_knn_neighbors(model['embedding'], new_embeddings, k=knn_neighbors)
    
    avg_new_params = []
    uncertainties_new_params = []
    for i in range(len(new_embeddings)):
        if knn_indices and len(knn_indices) > i:
            neighbor_indices = knn_indices[i]
            if neighbor_indices:
                expected_values, uncertainties = calculate_parameter_uncertainty(model, neighbor_indices)
                avg_new_params.append(expected_values)
                uncertainties_new_params.append(uncertainties)
            else:
                avg_new_params.append([np.nan, np.nan, np.nan, np.nan])
                uncertainties_new_params.append([np.nan, np.nan, np.nan, np.nan])
        else:
            avg_new_params.append([np.nan, np.nan, np.nan, np.nan])
            uncertainties_new_params.append([np.nan, np.nan, np.nan, np.nan])
    avg_new_params = np.array(avg_new_params)
    uncertainties_new_params = np.array(uncertainties_new_params)
    return {
        'new_spectra_data': new_spectra_data,
        'new_formulas': new_formulas,
        'new_params': new_params,
        'new_filenames': new_filenames,
        'new_embeddings': new_embeddings,
        'new_pca_components': new_pca_components,
        'knn_indices': knn_indices,
        'avg_new_params': avg_new_params,
        'uncertainties_new_params': uncertainties_new_params
    }

def calculate_parameter_uncertainty(model, neighbor_indices):
    uncertainties = []
    expected_values = []
    for i in range(4):  # For each parameter (logn, tex, velo, fwhm)
        param_values = model['y'][neighbor_indices, i]
        valid_values = param_values[~np.isnan(param_values)]
        if len(valid_values) > 0:
            expected_value = np.mean(valid_values)
            uncertainty = np.std(valid_values)
        else:
            expected_value = np.nan
            uncertainty = np.nan
        expected_values.append(expected_value)
        uncertainties.append(uncertainty)
    return expected_values, uncertainties

def create_3d_scatter(embeddings, color_values, title, color_label, color_scale='viridis', 
                      marker_size=5, selected_indices=None, selected_color='red', selected_size=10,
                      formulas=None, params=None, is_training=True, show_legend=False, legend_dict=None,
                      color_param=None):  # Añadir color_param como parámetro
    fig = go.Figure()
    
    hover_text = []
    for i in range(len(embeddings)):
        if is_training and formulas is not None and params is not None:
            text = f"Index: {i}<br>Formula: {formulas[i]}<br>log(n): {params[i, 0]:.2f}<br>T_ex: {params[i, 1]:.2f} K<br>Velocity: {params[i, 2]:.2f}<br>FWHM: {params[i, 3]:.2f}"
        elif not is_training and formulas is not None:
            text = f"New Spectrum: {formulas[i]}"
        else:
            text = f"Index: {i}"
        hover_text.append(text)
    
    if show_legend and legend_dict is not None and color_param == 'formula':
        unique_formulas = list(legend_dict.keys())
        
        import plotly.express as px
        colors = px.colors.qualitative.Set1 + px.colors.qualitative.Set2 + px.colors.qualitative.Set3
        formula_colors = {formula: colors[i % len(colors)] for i, formula in enumerate(unique_formulas)}
        
        for formula in unique_formulas:
            indices = [i for i, f in enumerate(formulas) if f == formula]
            if indices:
                fig.add_trace(go.Scatter3d(
                    x=embeddings[indices, 0],
                    y=embeddings[indices, 1],
                    z=embeddings[indices, 2],
                    mode='markers',
                    marker=dict(
                        size=marker_size,
                        color=formula_colors[formula],  
                        opacity=0.7,
                        line=dict(width=0)
                    ),
                    text=[hover_text[i] for i in indices],
                    hovertemplate=
                    '<b>X</b>: %{x}<br>' +
                    '<b>Y</b>: %{y}<br>' +
                    '<b>Z</b>: %{z}<br>' +
                    '%{text}' +
                    '<extra></extra>',
                    name=formula,
                    showlegend=True
                ))
    elif show_legend and legend_dict is not None:
        unique_formulas = list(legend_dict.keys())
        for formula in unique_formulas:
            indices = [i for i, f in enumerate(formulas) if f == formula]
            if indices:
                fig.add_trace(go.Scatter3d(
                    x=embeddings[indices, 0],
                    y=embeddings[indices, 1],
                    z=embeddings[indices, 2],
                    mode='markers',
                    marker=dict(
                        size=marker_size,
                        color=[color_values[i] for i in indices],
                        colorscale=color_scale,
                        opacity=0.7,
                        line=dict(width=0)
                    ),
                    text=[hover_text[i] for i in indices],
                    hovertemplate=
                    '<b>X</b>: %{x}<br>' +
                    '<b>Y</b>: %{y}<br>' +
                    '<b>Z</b>: %{z}<br>' +
                    '%{text}' +
                    '<extra></extra>',
                    name=formula,
                    showlegend=True
                ))
    else:
        fig.add_trace(go.Scatter3d(
            x=embeddings[:, 0],
            y=embeddings[:, 1],
            z=embeddings[:, 2],
            mode='markers',
            marker=dict(
                size=marker_size,
                color=color_values,
                colorscale=color_scale,
                opacity=0.7,
                colorbar=dict(
                    title=color_label,
                    len=0.5,  # Make colorbar shorter vertically
                    yanchor='middle',
                    y=0.5
                ),
                line=dict(width=0)
            ),
            text=hover_text,
            hovertemplate=
            '<b>X</b>: %{x}<br>' +
            '<b>Y</b>: %{y}<br>' +
            '<b>Z</b>: %{z}<br>' +
            '%{text}' +
            '<extra></extra>',
            name='Data points',
            showlegend=False
        ))
    
    if selected_indices is not None and len(selected_indices) > 0:
        selected_embeddings = embeddings[selected_indices]
        selected_values = color_values[selected_indices] if hasattr(color_values, '__len__') and len(color_values) == len(embeddings) else color_values
        
        fig.add_trace(go.Scatter3d(
            x=selected_embeddings[:, 0],
            y=selected_embeddings[:, 1],
            z=selected_embeddings[:, 2],
            mode='markers',
            marker=dict(
                size=selected_size,
                color=selected_color,
                opacity=1.0,
                line=dict(width=2, color='black')
            ),
            name='Selected points',
            showlegend=True
        ))
    
    fig.update_layout(
        title=title,
        scene=dict(
            xaxis_title='UMAP 1',
            yaxis_title='UMAP 2',
            zaxis_title='UMAP 3',
            camera=dict(eye=dict(x=1.5, y=1.5, z=1.5))
        ),
        height=600,
        margin=dict(l=0, r=0, b=0, t=30)
    )
    
    return fig

def create_2d_scatter(embeddings, color_values, title, color_label, color_scale='viridis', 
                      marker_size=5, selected_indices=None, selected_color='red', selected_size=10):
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=embeddings[:, 0],
        y=embeddings[:, 1],
        mode='markers',
        marker=dict(
            size=marker_size,
            color=color_values,
            colorscale=color_scale,
            opacity=0.7,
            colorbar=dict(title=color_label)
        ),
        text=[f"Index: {i}" for i in range(len(embeddings))],
        hovertemplate=
        '<b>X</b>: %{x}<br>' +
        '<b>Y</b>: %{y}<br>' +
        '<b>Value</b>: %{marker.color}<br>' +
        '<extra></extra>',
        name='Data points'
    ))
    
    if selected_indices is not None and len(selected_indices) > 0:
        selected_embeddings = embeddings[selected_indices]
        selected_values = color_values[selected_indices] if hasattr(color_values, '__len__') and len(color_values) == len(embeddings) else color_values
        
        fig.add_trace(go.Scatter(
            x=selected_embeddings[:, 0],
            y=selected_embeddings[:, 1],
            mode='markers',
            marker=dict(
                size=selected_size,
                color=selected_color,
                opacity=1.0,
                line=dict(width=2, color='black')
            ),
            name='Selected points'
        ))
    
    fig.update_layout(
        title=title,
        xaxis_title='UMAP 1',
        yaxis_title='UMAP 2',
        height=500,
        margin=dict(l=0, r=0, b=0, t=30)
    )
    
    return fig

def create_spectrum_plot(frequencies, intensities, title):
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=frequencies,
        y=intensities,
        mode='lines',
        line=dict(width=2),
        name='Spectrum'
    ))
    
    fig.update_layout(
        title=title,
        xaxis_title='Frequency (Hz)',
        yaxis_title='Intensity',
        height=400,
        margin=dict(l=0, r=0, b=0, t=30)
    )
    
    return fig

def main():
    
    st.image("NGC6523_BVO_2.jpg", use_column_width=True)
    
    col1, col2 = st.columns([1, 3])
    with col1:
        st.empty()
        
    with col2:
        st.markdown('<p class="main-title">AI-ITACA | Artificial Intelligence Integral Tool for AstroChemical Analysis</p>', unsafe_allow_html=True)
    
    st.markdown("""
    A remarkable upsurge in the complexity of molecules identified in the interstellar medium (ISM) is currently occurring, with over 80 new species discovered in the last three years. A number of them have been emphasized by prebiotic experiments as vital molecular building blocks of life. Since our Solar System was formed from a molecular cloud in the ISM, it prompts the query as to whether the rich interstellar chemical reservoir could have played a role in the emergence of life. The improved sensitivities of state-of-the-art astronomical facilities, such as the Atacama Large Millimeter/submillimeter Array (ALMA) and the James Webb Space Telescope (JWST), are revolutionizing the discovery of new molecules in space. However, we are still just scraping the tip of the iceberg. We are far from knowing the complete catalogue of molecules that astrochemistry can offer, as well as the complexity they can reach.<br><br>
    <strong>Artificial Intelligence Integral Tool for AstroChemical Analysis (AI-ITACA)</strong>, proposes to combine complementary machine learning (ML) techniques to address all the challenges that astrochemistry is currently facing. AI-ITACA will significantly contribute to the development of new AI-based cutting-edge analysis software that will allow us to make a crucial leap in the characterization of the level of chemical complexity in the ISM, and in our understanding of the contribution that interstellar chemistry might have in the origin of life.
    """, unsafe_allow_html=True)
    
    st.markdown("""
    <div class="info-box">
    <h4>About GUAPOS</h4>
    <p>The G31.41+0.31 Unbiased ALMA sPectral Observational Survey (GUAPOS) project targets the hot molecular core (HMC) G31.41+0.31 (G31) to reveal the complex chemistry of one of the most chemically rich high-mass star-forming regions outside the Galactic center (GC).</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Header
    st.markdown('<h1 class="main-header">🧪 3D Spectral Space Analyzer</h1>', unsafe_allow_html=True)
    
    with st.sidebar:
        st.header("Input Parameters")
        
        # Model upload
        st.subheader("1. Upload Model")
        model_file = st.file_uploader("Upload trained model (PKL file)", type=['pkl'])
        
        if model_file is not None:
            if st.button("Load Model") or st.session_state.model is None:
                with st.spinner("Loading model..."):
                    st.session_state.model = load_model(model_file)
                    if st.session_state.model is not None:
                        st.success("Model loaded successfully!")
        
        # Spectra upload
        st.subheader("2. Upload Spectra")
        spectra_files = st.file_uploader("Upload spectrum files (TXT)", type=['txt'], accept_multiple_files=True)
        
        if spectra_files:
            st.session_state.spectra_files = spectra_files
        
        st.subheader("3. Analysis Parameters")
        knn_neighbors = st.slider("Number of KNN neighbors", min_value=1, max_value=20, value=5)
        
        if st.button("Analyze Spectra") and st.session_state.model is not None and st.session_state.spectra_files:
            with st.spinner("Analyzing spectra..."):
                try:
                    model = st.session_state.model
                    results = analyze_spectra(model, st.session_state.spectra_files, knn_neighbors)
                    st.session_state.results = results
                    st.success("Analysis completed!")
                except Exception as e:
                    st.error(f"Error during analysis: {str(e)}")
    
    if st.session_state.model is None:
        st.info("Please upload a model file to get started.")
        return
    
    model = st.session_state.model
    
    with st.expander("Model Information", expanded=True):
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Training Samples", model.get('sample_size', 'N/A'))
        with col2:
            st.metric("PCA Components", model.get('n_components', 'N/A'))
        with col3:
            st.metric("Variance Threshold", f"{model.get('variance_threshold', 0.99)*100:.1f}%")
    
    if st.session_state.results is None:
        st.info("Upload spectrum files and click 'Analyze Spectra' to see results.")
        return
    
    results = st.session_state.results
    
    # Main content
    st.markdown('<div class="info-box">', unsafe_allow_html=True)
    st.write(f"**Analysis Results:** {len(results['new_embeddings'])} spectra processed and projected into 3D space")
    st.markdown('</div>', unsafe_allow_html=True)
    
    tab1, tab2, tab3, tab4 = st.tabs(["3D Projection", "2D Projection", "Spectrum View", "KNN Analysis"])
    
    with tab1:
        st.markdown('<h2 class="sub-header">3D UMAP Projection</h2>', unsafe_allow_html=True)
        
        param_options = ['logn', 'tex', 'velo', 'fwhm', 'formula']
        color_param = st.selectbox("Color by", param_options, index=4)
        
        combined_embeddings = np.vstack([model['embedding'], results['new_embeddings']])
        
        if color_param == 'formula':
            # For formula coloring, we need to create a numeric mapping
            all_formulas = np.concatenate([model['formulas'], results['new_formulas']])
            unique_formulas = np.unique(all_formulas)
            formula_to_num = {formula: i for i, formula in enumerate(unique_formulas)}
            color_values = np.array([formula_to_num[f] for f in all_formulas])
            color_label = "Formula"
            color_scale = 'viridis'
            
            legend_dict = {formula: formula_to_num[formula] for formula in unique_formulas}
            show_legend = True
        else:
            param_idx = param_options.index(color_param)
            if param_idx < 4:  # It's a parameter
                # For training data
                training_params = model['y'][:, param_idx]
                # For new data, use average from neighbors
                new_data_params = results['avg_new_params'][:, param_idx]
                color_values = np.concatenate([training_params, new_data_params])
                color_label = param_options[param_idx]
                color_scale = 'viridis'
                show_legend = False
                legend_dict = None
        
        # Create the plot
        selected_indices = list(range(len(model['embedding']), len(combined_embeddings)))
        
        all_formulas = np.concatenate([model['formulas'], results['new_formulas']])
        all_params = np.vstack([model['y'], results['avg_new_params']])
        
        fig_3d = create_3d_scatter(
            combined_embeddings, 
            color_values, 
            "3D UMAP Projection (Training + New Spectra)", 
            color_label,
            color_scale=color_scale,
            selected_indices=selected_indices,
            formulas=all_formulas,
            params=all_params,
            is_training=True,
            show_legend=show_legend,
            legend_dict=legend_dict,
            color_param=color_param  # Añadir este parámetro
        )
        
        st.plotly_chart(fig_3d, use_container_width=True)
        
        st.markdown('<h3 class="sub-header">New Spectrum Details</h3>', unsafe_allow_html=True)
        
        for i in range(len(results['new_embeddings'])):
            col1, col2 = st.columns([1, 2])
            
            with col1:
                st.write(f"**Spectrum {i+1}:** {results['new_filenames'][i]}")
                st.write(f"**Formula:** {results['new_formulas'][i]}")
                
                # Use average parameters from neighbors
                if results['knn_indices'] and len(results['knn_indices']) > i and len(results['knn_indices'][i]) > 0:
                    st.write(f"**log(n):** {results['avg_new_params'][i, 0]:.2f}")
                    st.write(f"**T_ex (K):** {results['avg_new_params'][i, 1]:.2f}")
                    st.write(f"**Velocity:** {results['avg_new_params'][i, 2]:.2f}")
                    st.write(f"**FWHM:** {results['avg_new_params'][i, 3]:.2f}")
                else:
                    st.write("**log(n):** No neighbors found")
                    st.write("**T_ex (K):** No neighbors found")
                    st.write("**Velocity:** No neighbors found")
                    st.write("**FWHM:** No neighbors found")
            
            with col2:
                spectrum_fig = create_spectrum_plot(
                    model['reference_frequencies'],
                    results['new_spectra_data'][i],
                    f"Spectrum: {results['new_filenames'][i]}"
                )
                st.plotly_chart(spectrum_fig, use_container_width=True)
    
    with tab2:
        st.markdown('<h2 class="sub-header">2D UMAP Projection</h2>', unsafe_allow_html=True)
        
        color_param_2d = st.selectbox("Color by", param_options, index=4, key='color_2d')
        
        if color_param_2d == 'formula':
            color_values_2d = color_values
            color_label_2d = "Formula"
            color_scale_2d = 'viridis'
        else:
            param_idx = param_options.index(color_param_2d)
            if param_idx < 4:  # It's a parameter
                color_values_2d = np.concatenate([model['y'][:, param_idx], results['avg_new_params'][:, param_idx]])
                color_label_2d = param_options[param_idx]
                color_scale_2d = 'viridis'
        
        # Create the plot
        fig_2d = create_2d_scatter(
            combined_embeddings, 
            color_values_2d, 
            "2D UMAP Projection (Training + New Spectra)", 
            color_label_2d,
            color_scale=color_scale_2d,
            selected_indices=selected_indices
        )
        
        st.plotly_chart(fig_2d, use_container_width=True)
    
    with tab3:
        st.markdown('<h2 class="sub-header">Spectrum Comparison</h2>', unsafe_allow_html=True)
        
        spectrum_idx = st.selectbox("Select spectrum", range(len(results['new_embeddings'])), 
                                  format_func=lambda x: results['new_filenames'][x])
        
        if spectrum_idx is not None:
            col1, col2 = st.columns(2)
            
            with col1:
                spectrum_fig = create_spectrum_plot(
                    model['reference_frequencies'],
                    results['new_spectra_data'][spectrum_idx],
                    f"Spectrum: {results['new_filenames'][spectrum_idx]}"
                )
                st.plotly_chart(spectrum_fig, use_container_width=True)
            
            with col2:
                if results['knn_indices'] and len(results['knn_indices']) > spectrum_idx:
                    neighbor_indices = results['knn_indices'][spectrum_idx]
                    
                    if neighbor_indices:
                        st.write("**K-Nearest Neighbors:**")
                        
                        neighbor_data = []
                        for idx in neighbor_indices:
                            neighbor_data.append({
                                'Formula': model['formulas'][idx],
                                'log(n)': f"{model['y'][idx, 0]:.2f}",
                                'T_ex (K)': f"{model['y'][idx, 1]:.2f}",
                                'Velocity': f"{model['y'][idx, 2]:.2f}",
                                'FWHM': f"{model['y'][idx, 3]:.2f}"
                            })
                        
                        neighbor_df = pd.DataFrame(neighbor_data)
                        st.dataframe(neighbor_df, use_container_width=True)
    
    with tab4:
        st.markdown('<h2 class="sub-header">K-Nearest Neighbors Analysis</h2>', unsafe_allow_html=True)
        
        for i in range(len(results['new_embeddings'])):
            st.markdown(f"**{results['new_filenames'][i]}** ({results['new_formulas'][i]})")
            
            if results['knn_indices'] and len(results['knn_indices']) > i:
                neighbor_indices = results['knn_indices'][i]
                
                if neighbor_indices:
                    neighbor_data = []
                    for idx in neighbor_indices:
                        neighbor_data.append({
                            'Formula': model['formulas'][idx],
                            'log(n)': f"{model['y'][idx, 0]:.2f}",
                            'T_ex (K)': f"{model['y'][idx, 1]:.2f}",
                            'Velocity': f"{model['y'][idx, 2]:.2f}",
                            'FWHM': f"{model['y'][idx, 3]:.2f}",
                            'Distance': f"{np.linalg.norm(model['embedding'][idx] - results['new_embeddings'][i]):.4f}"
                        })
                    
                    neighbor_df = pd.DataFrame(neighbor_data)
                    st.dataframe(neighbor_df, use_container_width=True)
                    
                    st.write("**Average parameters of neighbors:**")
                    avg_params = {
                        'log(n)': np.mean([model['y'][idx, 0] for idx in neighbor_indices]),
                        'T_ex (K)': np.mean([model['y'][idx, 1] for idx in neighbor_indices]),
                        'Velocity': np.mean([model['y'][idx, 2] for idx in neighbor_indices]),
                        'FWHM': np.mean([model['y'][idx, 3] for idx in neighbor_indices])
                    }
                    
                    avg_df = pd.DataFrame([avg_params])
                    st.dataframe(avg_df, use_container_width=True)
                    
                    comparison_data = {
                        'Parameter': ['log(n)', 'T_ex (K)', 'Velocity', 'FWHM'],
                        'New Spectrum': [results['avg_new_params'][i, 0], results['avg_new_params'][i, 1], results['avg_new_params'][i, 2], results['avg_new_params'][i, 3]],
                        'Neighbors Average': [avg_params['log(n)'], avg_params['T_ex (K)'], avg_params['Velocity'], avg_params['FWHM']],
                        'Difference': [
                            results['avg_new_params'][i, 0] - avg_params['log(n)'],
                            results['avg_new_params'][i, 1] - avg_params['T_ex (K)'],
                            results['avg_new_params'][i, 2] - avg_params['Velocity'],
                            results['avg_new_params'][i, 3] - avg_params['FWHM']
                        ]
                    }
                    
                    comparison_df = pd.DataFrame(comparison_data)
                    st.dataframe(comparison_df, use_container_width=True)
            
            st.markdown("---")

if __name__ == "__main__":
    main()









