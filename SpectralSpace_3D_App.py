import os
import pickle
import numpy as np
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
from scipy.interpolate import interp1d
import re
from sklearn.neighbors import NearestNeighbors
import base64
from io import StringIO

# Set page configuration
st.set_page_config(
    page_title="3D Spectrum Analysis Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #1f77b4;
        margin-bottom: 1rem;
    }
    .info-box {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
    }
    .stButton>button {
        width: 100%;
    }
</style>
""", unsafe_allow_html=True)

def sanitize_filename(filename):
    """Elimina caracteres inválidos de los nombres de archivo"""
    invalid_chars = r'[<>:"/\\|?*]'
    return re.sub(invalid_chars, '_', filename)

def load_model(model_path):
    """Carga el modelo entrenado desde un .pkl"""
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    return model

def extract_molecule_formula(header):
    """
    Extract molecule formula from header string.
    Example: "molecules='C2H5OH,V=0|1'" returns "C2H5OH"
    """
    pattern = r"molecules=['\"]([^,'\"]+)"
    match = re.search(pattern, header)
    if match:
        formula = match.group(1)
        if ',' in formula:
            formula = formula.split(',')[0]
        return formula
    return "Unknown"

def load_and_interpolate_spectrum(file_content, filename, reference_frequencies):
    """Carga un espectro desde contenido de archivo y lo interpola a las frecuencias de referencia"""
    lines = file_content.split('\n')
    
    # Determinar el formato del archivo
    first_line = lines[0].strip() if lines else ""
    second_line = lines[1].strip() if len(lines) > 1 else ""
    
    formula = "Unknown"
    param_dict = {}
    data_start_line = 0
    
    # Formato 1: con header de molécula y parámetros
    if first_line.startswith('//') and 'molecules=' in first_line:
        header = first_line[2:].strip()  # Remove the '//'
        formula = extract_molecule_formula(header)
        
        # Extraer parámetros del header
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
    
    # Formato 2: con header de columnas
    elif first_line.startswith('!') or first_line.startswith('#'):
        # Intentar extraer información del header si está disponible
        if 'molecules=' in first_line:
            formula = extract_molecule_formula(first_line)
        data_start_line = 1
    
    # Formato 3: sin header, solo datos
    else:
        data_start_line = 0
        formula = filename.split('.')[0]  # Usar nombre del archivo como fórmula

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
        raise ValueError("No valid data points found in spectrum file")

    spectrum_data = np.array(spectrum_data)

    # Ajustar frecuencia si está en GHz (convertir a Hz)
    if np.max(spectrum_data[:, 0]) < 1e11:  # Si las frecuencias son menores a 100 GHz, probablemente están en GHz
        spectrum_data[:, 0] = spectrum_data[:, 0] * 1e9  # Convertir GHz to Hz
        st.info(f"Converted frequencies from GHz to Hz for {filename}")

    interpolator = interp1d(spectrum_data[:, 0], spectrum_data[:, 1],
                            kind='linear', bounds_error=False, fill_value=0.0)
    interpolated = interpolator(reference_frequencies)

    # Extraer parámetros con valores por defecto si faltan
    params = [
        param_dict.get('logn', np.nan),
        param_dict.get('tex', np.nan),
        param_dict.get('velo', np.nan),
        param_dict.get('fwhm', np.nan)
    ]

    return spectrum_data, interpolated, formula, params, filename

def find_knn_neighbors(training_embeddings, new_embeddings, k=5):
    """Encuentra los k vecinos más cercanos usando KNN en 3D"""
    if len(training_embeddings) == 0 or len(new_embeddings) == 0:
        return []
    
    # Asegurar que k no sea mayor que el número de puntos de entrenamiento
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

def create_3d_scatter(model, new_embeddings, new_formulas, new_params, param_index=0, param_name="logn"):
    """Create a 3D scatter plot with training and prediction data"""
    param_names = ['logn', 'tex', 'velo', 'fwhm']
    param_labels = ['log(n)', 'T_ex (K)', 'Velocity (km/s)', 'FWHM (km/s)']
    
    # Create figure
    fig = go.Figure()
    
    # Add training data
    fig.add_trace(go.Scatter3d(
        x=model['embedding'][:, 0],
        y=model['embedding'][:, 1],
        z=model['embedding'][:, 2],
        mode='markers',
        marker=dict(
            size=4,
            color=model['y'][:, param_index],
            colorscale='Viridis',
            opacity=0.6,
            colorbar=dict(title=param_labels[param_index])
        ),
        name='Training Data',
        hovertemplate=
        '<b>Formula</b>: %{text}<br>' +
        f'<b>{param_labels[param_index]}</b>: %{{marker.color:.2f}}<br>' +
        '<b>X</b>: %{x:.2f}<br>' +
        '<b>Y</b>: %{y:.2f}<br>' +
        '<b>Z</b>: %{z:.2f}<extra></extra>',
        text=model['formulas']
    ))
    
    # Add new predictions
    fig.add_trace(go.Scatter3d(
        x=new_embeddings[:, 0],
        y=new_embeddings[:, 1],
        z=new_embeddings[:, 2],
        mode='markers',
        marker=dict(
            size=8,
            color=new_params[:, param_index],
            colorscale='Plasma',
            opacity=1.0,
            symbol='diamond'
        ),
        name='New Predictions',
        hovertemplate=
        '<b>Formula</b>: %{text}<br>' +
        f'<b>{param_labels[param_index]}</b>: %{{marker.color:.2f}}<br>' +
        '<b>X</b>: %{x:.2f}<br>' +
        '<b>Y</b>: %{y:.2f}<br>' +
        '<b>Z</b>: %{z:.2f}<extra></extra>',
        text=new_formulas
    ))
    
    # Update layout
    fig.update_layout(
        title=f'3D UMAP: {param_name} (Training + Predictions)',
        scene=dict(
            xaxis_title='UMAP 1',
            yaxis_title='UMAP 2',
            zaxis_title='UMAP 3'
        ),
        height=700,
        margin=dict(l=0, r=0, b=0, t=40)
    )
    
    return fig

def create_formula_scatter(model, new_embeddings, new_formulas):
    """Create a 3D scatter plot colored by molecular formula"""
    # Combine all formulas
    all_formulas = np.concatenate([model['formulas'], new_formulas])
    unique_formulas = np.unique(all_formulas)
    
    # Create a color map for formulas
    colors = px.colors.qualitative.Plotly
    formula_to_color = {formula: colors[i % len(colors)] for i, formula in enumerate(unique_formulas)}
    
    # Create figure
    fig = go.Figure()
    
    # Add training data
    for formula in np.unique(model['formulas']):
        mask = model['formulas'] == formula
        fig.add_trace(go.Scatter3d(
            x=model['embedding'][mask, 0],
            y=model['embedding'][mask, 1],
            z=model['embedding'][mask, 2],
            mode='markers',
            marker=dict(
                size=4,
                color=formula_to_color[formula],
                opacity=0.6
            ),
            name=f'{formula} (Train)',
            hovertemplate=
            '<b>Formula</b>: ' + formula + '<br>' +
            '<b>X</b>: %{x:.2f}<br>' +
            '<b>Y</b>: %{y:.2f}<br>' +
            '<b>Z</b>: %{z:.2f}<extra></extra>'
        ))
    
    # Add new predictions
    for formula in np.unique(new_formulas):
        mask = new_formulas == formula
        fig.add_trace(go.Scatter3d(
            x=new_embeddings[mask, 0],
            y=new_embeddings[mask, 1],
            z=new_embeddings[mask, 2],
            mode='markers',
            marker=dict(
                size=8,
                color=formula_to_color[formula],
                symbol='diamond',
                opacity=1.0
            ),
            name=f'{formula} (New)',
            hovertemplate=
            '<b>Formula</b>: ' + formula + '<br>' +
            '<b>X</b>: %{x:.2f}<br>' +
            '<b>Y</b>: %{y:.2f}<br>' +
            '<b>Z</b>: %{z:.2f}<extra></extra>'
        ))
    
    # Update layout
    fig.update_layout(
        title='3D UMAP: Molecular Formula (Training + Predictions)',
        scene=dict(
            xaxis_title='UMAP 1',
            yaxis_title='UMAP 2',
            zaxis_title='UMAP 3'
        ),
        height=700,
        margin=dict(l=0, r=0, b=0, t=40)
    )
    
    return fig

def create_spectrum_plot(spectrum_data, interpolated, ref_freqs, formula, filename):
    """Create a plot of original vs interpolated spectrum"""
    fig = go.Figure()
    
    # Add original spectrum
    fig.add_trace(go.Scatter(
        x=spectrum_data[:, 0],
        y=spectrum_data[:, 1],
        mode='lines',
        name='Original',
        line=dict(width=1, color='blue')
    ))
    
    # Add interpolated spectrum
    fig.add_trace(go.Scatter(
        x=ref_freqs,
        y=interpolated,
        mode='lines',
        name='Interpolated',
        line=dict(width=2, color='red')
    ))
    
    # Update layout
    fig.update_layout(
        title=f'Spectrum: {filename} - Formula: {formula}',
        xaxis_title='Frequency (Hz)',
        yaxis_title='Intensity',
        height=400,
        showlegend=True
    )
    
    return fig

def main():
    # Header
    st.markdown('<h1 class="main-header">3D Spectrum Analysis Dashboard</h1>', unsafe_allow_html=True)
    
    # Sidebar
    st.sidebar.title("Configuration")
    
    # File uploaders
    st.sidebar.subheader("Upload Model File")
    model_file = st.sidebar.file_uploader("Upload trained model (.pkl)", type=['pkl'])
    
    st.sidebar.subheader("Upload Spectrum Files")
    spectrum_files = st.sidebar.file_uploader("Upload spectrum files (.txt)", type=['txt'], accept_multiple_files=True)
    
    # Parameters
    st.sidebar.subheader("Parameters")
    knn_neighbors = st.sidebar.slider("Number of KNN Neighbors", min_value=1, max_value=20, value=5)
    
    # Initialize session state
    if 'model' not in st.session_state:
        st.session_state.model = None
    if 'predictions' not in st.session_state:
        st.session_state.predictions = None
    
    # Load model if provided
    if model_file is not None:
        try:
            model = pickle.load(model_file)
            st.session_state.model = model
            st.sidebar.success("Model loaded successfully!")
        except Exception as e:
            st.sidebar.error(f"Error loading model: {e}")
    
    # Process spectra if model and files are provided
    if st.session_state.model is not None and spectrum_files:
        model = st.session_state.model
        
        # Display model info
        with st.expander("Model Information"):
            st.write(f"Training samples: {model.get('sample_size', 'N/A')}")
            st.write(f"Number of components: {model.get('n_components', 'N/A')}")
            st.write(f"Variance threshold: {model.get('variance_threshold', 'N/A')}")
            st.write(f"Is 3D: {model['embedding'].shape[1] == 3}")
        
        # Process spectra
        scaler = model['scaler']
        pca = model['pca']
        umap_model = model['umap']
        ref_freqs = model['reference_frequencies']
        
        new_spectra_data = []
        new_formulas = []
        new_params = []
        new_filenames = []
        new_embeddings = []
        new_pca_components = []
        spectrum_plots = []
        
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        for i, spectrum_file in enumerate(spectrum_files):
            status_text.text(f"Processing {spectrum_file.name} ({i+1}/{len(spectrum_files)})")
            
            try:
                # Read file content
                content = spectrum_file.getvalue().decode("utf-8")
                
                # Process spectrum
                spectrum_data, interpolated, formula, params, filename = load_and_interpolate_spectrum(
                    content, spectrum_file.name, ref_freqs
                )
                
                # Transform the spectrum
                X_scaled = scaler.transform([interpolated])
                X_pca = pca.transform(X_scaled)
                X_umap = umap_model.transform(X_pca)
                
                # Store results
                new_spectra_data.append(interpolated)
                new_formulas.append(formula)
                new_params.append(params)
                new_filenames.append(filename)
                new_embeddings.append(X_umap[0])
                new_pca_components.append(X_pca[0])
                
                # Create spectrum plot
                spectrum_plot = create_spectrum_plot(spectrum_data, interpolated, ref_freqs, formula, filename)
                spectrum_plots.append(spectrum_plot)
                
            except Exception as e:
                st.error(f"Error processing {spectrum_file.name}: {e}")
            
            progress_bar.progress((i + 1) / len(spectrum_files))
        
        status_text.text("Processing complete!")
        progress_bar.empty()
        
        if new_embeddings:
            new_embeddings = np.array(new_embeddings)
            new_params = np.array(new_params)
            new_formulas = np.array(new_formulas)
            new_pca_components = np.array(new_pca_components)
            
            # Find KNN neighbors
            knn_indices = find_knn_neighbors(model['embedding'], new_embeddings, k=knn_neighbors)
            
            # Store predictions in session state
            st.session_state.predictions = {
                'embeddings': new_embeddings,
                'formulas': new_formulas,
                'params': new_params,
                'filenames': new_filenames,
                'pca_components': new_pca_components,
                'knn_indices': knn_indices,
                'spectrum_plots': spectrum_plots
            }
    
    # Display results if available
    if st.session_state.predictions is not None and st.session_state.model is not None:
        predictions = st.session_state.predictions
        model = st.session_state.model
        
        # Create tabs for different visualizations
        tab1, tab2, tab3, tab4 = st.tabs(["3D Visualizations", "Spectrum Plots", "KNN Analysis", "Data Tables"])
        
        with tab1:
            st.subheader("3D Visualizations")
            
            # Parameter selection for coloring
            param_options = ['logn', 'tex', 'velo', 'fwhm']
            param_index = st.selectbox("Select parameter for coloring", range(len(param_options)), 
                                      format_func=lambda x: param_options[x])
            
            # Create 3D scatter plot
            fig_3d = create_3d_scatter(model, predictions['embeddings'], predictions['formulas'], 
                                      predictions['params'], param_index, param_options[param_index])
            st.plotly_chart(fig_3d, use_container_width=True)
            
            # Create formula-colored 3D scatter plot
            st.subheader("Molecular Formula Visualization")
            fig_formula = create_formula_scatter(model, predictions['embeddings'], predictions['formulas'])
            st.plotly_chart(fig_formula, use_container_width=True)
        
        with tab2:
            st.subheader("Spectrum Plots")
            
            # Display spectrum plots in a grid
            cols = st.columns(2)
            for i, plot in enumerate(predictions['spectrum_plots']):
                with cols[i % 2]:
                    st.plotly_chart(plot, use_container_width=True)
        
        with tab3:
            st.subheader("K-Nearest Neighbors Analysis")
            
            # Select a spectrum to analyze
            if len(predictions['filenames']) > 0:
                selected_idx = st.selectbox("Select spectrum for KNN analysis", 
                                           range(len(predictions['filenames'])), 
                                           format_func=lambda x: predictions['filenames'][x])
                
                if selected_idx < len(predictions['knn_indices']) and predictions['knn_indices'][selected_idx]:
                    neighbor_indices = predictions['knn_indices'][selected_idx]
                    
                    # Create table with neighbor information
                    neighbor_data = []
                    for idx in neighbor_indices:
                        neighbor_data.append({
                            'Formula': model['formulas'][idx],
                            'log(n)': f"{model['y'][idx, 0]:.2f}",
                            'T_ex (K)': f"{model['y'][idx, 1]:.2f}",
                            'Velocity': f"{model['y'][idx, 2]:.2f}",
                            'FWHM': f"{model['y'][idx, 3]:.2f}"
                        })
                    
                    df_neighbors = pd.DataFrame(neighbor_data)
                    st.dataframe(df_neighbors, use_container_width=True)
                    
                    # Create a plot showing the selected spectrum and its neighbors
                    fig_knn = go.Figure()
                    
                    # Add all training data
                    fig_knn.add_trace(go.Scatter3d(
                        x=model['embedding'][:, 0],
                        y=model['embedding'][:, 1],
                        z=model['embedding'][:, 2],
                        mode='markers',
                        marker=dict(size=3, color='lightgray', opacity=0.3),
                        name='Training Data',
                        hovertemplate='<b>Formula</b>: %{text}<extra></extra>',
                        text=model['formulas']
                    ))
                    
                    # Add KNN neighbors
                    fig_knn.add_trace(go.Scatter3d(
                        x=model['embedding'][neighbor_indices, 0],
                        y=model['embedding'][neighbor_indices, 1],
                        z=model['embedding'][neighbor_indices, 2],
                        mode='markers',
                        marker=dict(size=8, color='blue', opacity=0.8),
                        name='KNN Neighbors',
                        hovertemplate='<b>Formula</b>: %{text}<extra></extra>',
                        text=model['formulas'][neighbor_indices]
                    ))
                    
                    # Add selected prediction
                    fig_knn.add_trace(go.Scatter3d(
                        x=[predictions['embeddings'][selected_idx, 0]],
                        y=[predictions['embeddings'][selected_idx, 1]],
                        z=[predictions['embeddings'][selected_idx, 2]],
                        mode='markers',
                        marker=dict(size=12, color='red', symbol='diamond'),
                        name='Selected Prediction',
                        hovertemplate=f'<b>Formula</b>: {predictions["formulas"][selected_idx]}<extra></extra>'
                    ))
                    
                    # Update layout
                    fig_knn.update_layout(
                        title=f'K-Nearest Neighbors for {predictions["filenames"][selected_idx]}',
                        scene=dict(
                            xaxis_title='UMAP 1',
                            yaxis_title='UMAP 2',
                            zaxis_title='UMAP 3'
                        ),
                        height=600
                    )
                    
                    st.plotly_chart(fig_knn, use_container_width=True)
                else:
                    st.info("No KNN neighbors found for the selected spectrum.")
            else:
                st.info("No spectra available for KNN analysis.")
        
        with tab4:
            st.subheader("Data Tables")
            
            # Create prediction results table
            prediction_data = []
            for i in range(len(predictions['filenames'])):
                prediction_data.append({
                    'Filename': predictions['filenames'][i],
                    'Formula': predictions['formulas'][i],
                    'log(n)': f"{predictions['params'][i, 0]:.2f}",
                    'T_ex (K)': f"{predictions['params'][i, 1]:.2f}",
                    'Velocity': f"{predictions['params'][i, 2]:.2f}",
                    'FWHM': f"{predictions['params'][i, 3]:.2f}",
                    'UMAP X': f"{predictions['embeddings'][i, 0]:.4f}",
                    'UMAP Y': f"{predictions['embeddings'][i, 1]:.4f}",
                    'UMAP Z': f"{predictions['embeddings'][i, 2]:.4f}"
                })
            
            df_predictions = pd.DataFrame(prediction_data)
            st.dataframe(df_predictions, use_container_width=True)
            
            # Add download button
            csv = df_predictions.to_csv(index=False)
            b64 = base64.b64encode(csv.encode()).decode()
            href = f'<a href="data:file/csv;base64,{b64}" download="predictions.csv">Download Predictions CSV</a>'
            st.markdown(href, unsafe_allow_html=True)
    
    else:
        # Show instructions if no data is loaded
        st.info("""
        ## Instructions:
        1. Upload a trained model file (.pkl) in the sidebar
        2. Upload one or more spectrum files (.txt) for analysis
        3. Adjust parameters as needed
        4. View the interactive 3D visualizations and analysis results
        """)
        
        # Example of what the app can do
        st.subheader("About this Application")
        st.write("""
        This application provides interactive 3D visualization of spectral data analysis using 
        dimensionality reduction techniques like PCA and UMAP. It allows you to:
        
        - Project new spectra into a pre-trained 3D model space
        - Visualize the results in interactive 3D plots
        - Analyze K-nearest neighbors for each prediction
        - View original and interpolated spectra
        - Export results as CSV files
        
        The application is particularly useful for analyzing molecular spectra and identifying
        patterns in high-dimensional spectral data.
        """)

if __name__ == "__main__":
    main()
