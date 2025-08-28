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
import tempfile
from io import StringIO

# Set page configuration
st.set_page_config(
    page_title="3D Spectrum Analysis",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
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
        color: #ff7f0e;
        margin-top: 1.5rem;
        margin-bottom: 1rem;
    }
    .info-box {
        background-color: #f0f2f6;
        padding: 15px;
        border-radius: 10px;
        margin-bottom: 15px;
    }
    .stButton button {
        width: 100%;
    }
</style>
""", unsafe_allow_html=True)

def sanitize_filename(filename):
    """Remove invalid characters from filenames"""
    invalid_chars = r'[<>:"/\\|?*]'
    return re.sub(invalid_chars, '_', filename)

def load_model(model_file):
    """Load trained model from uploaded file"""
    try:
        model = pickle.load(model_file)
        return model
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None

def extract_molecule_formula(header):
    """Extract molecule formula from header string"""
    pattern = r"molecules=['\"]([^,'\"]+)"
    match = re.search(pattern, header)
    if match:
        formula = match.group(1)
        if ',' in formula:
            formula = formula.split(',')[0]
        return formula
    return "Unknown"

def parse_spectrum_content(content, filename, reference_frequencies):
    """Parse spectrum content from uploaded file"""
    lines = content.split('\n')
    
    # Determine file format
    first_line = lines[0].strip() if len(lines) > 0 else ""
    second_line = lines[1].strip() if len(lines) > 1 else ""
    
    formula = "Unknown"
    param_dict = {}
    data_start_line = 0
    
    # Format 1: with molecule and parameters header
    if first_line.startswith('//') and 'molecules=' in first_line:
        header = first_line[2:].strip()
        formula = extract_molecule_formula(header)
        
        # Extract parameters from header
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
    
    # Format 2: with column header
    elif first_line.startswith('!') or first_line.startswith('#'):
        if 'molecules=' in first_line:
            formula = extract_molecule_formula(first_line)
        data_start_line = 1
    
    # Format 3: no header, just data
    else:
        data_start_line = 0
        formula = filename.split('.')[0]  # Use filename as formula

    spectrum_data = []
    for line in lines[data_start_line:]:
        line = line.strip()
        # Skip comment or empty lines
        if not line or line.startswith('!') or line.startswith('#'):
            continue
            
        try:
            parts = line.split()
            if len(parts) >= 2:
                # Try different number formats
                try:
                    freq = float(parts[0])
                    intensity = float(parts[1])
                except ValueError:
                    # Try with scientific notation that might have D instead of E
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

    # Adjust frequency if in GHz (convert to Hz)
    if np.max(spectrum_data[:, 0]) < 1e11:  # If frequencies are less than 100 GHz, probably in GHz
        spectrum_data[:, 0] = spectrum_data[:, 0] * 1e9  # Convert GHz to Hz
        st.info(f"Converted frequencies from GHz to Hz for {filename}")

    interpolator = interp1d(spectrum_data[:, 0], spectrum_data[:, 1],
                            kind='linear', bounds_error=False, fill_value=0.0)
    interpolated = interpolator(reference_frequencies)

    # Extract parameters with default values if missing
    params = [
        param_dict.get('logn', np.nan),
        param_dict.get('tex', np.nan),
        param_dict.get('velo', np.nan),
        param_dict.get('fwhm', np.nan)
    ]

    return spectrum_data, interpolated, formula, params, filename

def find_knn_neighbors(training_embeddings, new_embeddings, k=5):
    """Find k nearest neighbors using KNN in 3D"""
    if len(training_embeddings) == 0 or len(new_embeddings) == 0:
        return []
    
    # Ensure k is not greater than the number of training points
    k = min(k, len(training_embeddings))
    
    knn = NearestNeighbors(n_neighbors=k, metric='euclidean')
    knn.fit(training_embeddings)
    
    all_neighbor_indices = []
    for new_embedding in new_embeddings:
        distances, indices = knn.kneighbors([new_embedding])
        # Verify that indices are within valid range
        valid_indices = [idx for idx in indices[0] if idx < len(training_embeddings)]
        all_neighbor_indices.append(valid_indices)
    
    return all_neighbor_indices

def create_3d_scatter(model, new_embeddings, new_formulas, new_params, param_index, param_name, param_label):
    """Create a 3D scatter plot with training data and new predictions"""
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
            colorbar=dict(title=param_label, x=0.8)
        ),
        name='Training Data',
        hovertemplate=
        '<b>Formula</b>: %{customdata[0]}<br>' +
        f'<b>{param_label}</b>: %{{customdata[1]:.2f}}<br>' +
        '<b>X</b>: %{x:.2f}<br>' +
        '<b>Y</b>: %{y:.2f}<br>' +
        '<b>Z</b>: %{z:.2f}<extra></extra>',
        customdata=np.column_stack((model['formulas'], model['y'][:, param_index]))
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
            symbol='diamond',
            line=dict(color='red', width=2)
        ),
        name='New Predictions',
        hovertemplate=
        '<b>Formula</b>: %{customdata[0]}<br>' +
        f'<b>{param_label}</b>: %{{customdata[1]:.2f}}<br>' +
        '<b>X</b>: %{x:.2f}<br>' +
        '<b>Y</b>: %{y:.2f}<br>' +
        '<b>Z</b>: %{z:.2f}<extra></extra>',
        customdata=np.column_stack((new_formulas, new_params[:, param_index]))
    ))
    
    fig.update_layout(
        title=f'3D UMAP: {param_name} (Training + Predictions)',
        scene=dict(
            xaxis_title='UMAP 1',
            yaxis_title='UMAP 2',
            zaxis_title='UMAP 3'
        ),
        width=1000,
        height=800,
        margin=dict(l=0, r=0, b=0, t=50)
    )
    
    return fig

def create_formula_plot(model, new_embeddings, new_formulas):
    """Create a 3D plot colored by molecular formula"""
    all_formulas = np.concatenate([model['formulas'], new_formulas])
    unique_formulas = np.unique(all_formulas)
    
    fig = go.Figure()
    
    # Add training data
    for formula in np.unique(model['formulas']):
        mask = model['formulas'] == formula
        fig.add_trace(go.Scatter3d(
            x=model['embedding'][mask, 0],
            y=model['embedding'][mask, 1],
            z=model['embedding'][mask, 2],
            mode='markers',
            marker=dict(size=4, opacity=0.6),
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
            marker=dict(size=8, symbol='diamond', line=dict(color='black', width=2)),
            name=f'{formula} (New)',
            hovertemplate=
            '<b>Formula</b>: ' + formula + '<br>' +
            '<b>X</b>: %{x:.2f}<br>' +
            '<b>Y</b>: %{y:.2f}<br>' +
            '<b>Z</b>: %{z:.2f}<extra></extra>'
        ))
    
    fig.update_layout(
        title='3D UMAP: Molecular Formula (Training + Predictions)',
        scene=dict(
            xaxis_title='UMAP 1',
            yaxis_title='UMAP 2',
            zaxis_title='UMAP 3'
        ),
        width=1000,
        height=800,
        margin=dict(l=0, r=0, b=0, t=50)
    )
    
    return fig

def create_knn_plot(model, new_embeddings, new_formulas, knn_indices):
    """Create a 3D plot showing KNN relationships"""
    fig = go.Figure()
    
    # Add training data
    fig.add_trace(go.Scatter3d(
        x=model['embedding'][:, 0],
        y=model['embedding'][:, 1],
        z=model['embedding'][:, 2],
        mode='markers',
        marker=dict(size=4, color='lightgray', opacity=0.3),
        name='Training Data',
        hovertemplate=
        '<b>Formula</b>: %{customdata}<br>' +
        '<b>X</b>: %{x:.2f}<br>' +
        '<b>Y</b>: %{y:.2f}<br>' +
        '<b>Z</b>: %{z:.2f}<extra></extra>',
        customdata=model['formulas']
    ))
    
    # Add KNN connections and neighbors
    for i, (new_embedding, indices) in enumerate(zip(new_embeddings, knn_indices)):
        if indices:  # Only if there are valid neighbors
            # Add connections
            for idx in indices:
                fig.add_trace(go.Scatter3d(
                    x=[new_embedding[0], model['embedding'][idx, 0]],
                    y=[new_embedding[1], model['embedding'][idx, 1]],
                    z=[new_embedding[2], model['embedding'][idx, 2]],
                    mode='lines',
                    line=dict(color='gray', width=2, dash='dash'),
                    opacity=0.5,
                    showlegend=False,
                    hoverinfo='skip'
                ))
            
            # Add neighbors
            fig.add_trace(go.Scatter3d(
                x=model['embedding'][indices, 0],
                y=model['embedding'][indices, 1],
                z=model['embedding'][indices, 2],
                mode='markers',
                marker=dict(size=6, color='blue', opacity=0.7),
                name='KNN Neighbors' if i == 0 else '',
                hovertemplate=
                '<b>Formula</b>: %{customdata}<br>' +
                '<b>X</b>: %{x:.2f}<br>' +
                '<b>Y</b>: %{y:.2f}<br>' +
                '<b>Z</b>: %{z:.2f}<extra></extra>',
                customdata=model['formulas'][indices]
            ))
    
    # Add new predictions
    fig.add_trace(go.Scatter3d(
        x=new_embeddings[:, 0],
        y=new_embeddings[:, 1],
        z=new_embeddings[:, 2],
        mode='markers',
        marker=dict(size=10, color='red', symbol='diamond', line=dict(color='black', width=2)),
        name='New Predictions',
        hovertemplate=
        '<b>Formula</b>: %{customdata}<br>' +
        '<b>X</b>: %{x:.2f}<br>' +
        '<b>Y</b>: %{y:.2f}<br>' +
        '<b>Z</b>: %{z:.2f}<extra></extra>',
        customdata=new_formulas
    ))
    
    fig.update_layout(
        title='3D UMAP: K-Nearest Neighbors Analysis',
        scene=dict(
            xaxis_title='UMAP 1',
            yaxis_title='UMAP 2',
            zaxis_title='UMAP 3'
        ),
        width=1000,
        height=800,
        margin=dict(l=0, r=0, b=0, t=50)
    )
    
    return fig

def create_spectrum_plot(spectrum_data, interpolated, reference_frequencies, formula, filename):
    """Create a plot showing original vs interpolated spectrum"""
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=spectrum_data[:, 0],
        y=spectrum_data[:, 1],
        mode='lines',
        name='Original',
        line=dict(color='blue', width=1),
        opacity=0.7
    ))
    
    fig.add_trace(go.Scatter(
        x=reference_frequencies,
        y=interpolated,
        mode='lines',
        name='Interpolated',
        line=dict(color='red', width=2)
    ))
    
    fig.update_layout(
        title=f'Spectrum: {filename}<br>Formula: {formula}',
        xaxis_title='Frequency (Hz)',
        yaxis_title='Intensity',
        width=800,
        height=500,
        margin=dict(l=50, r=50, b=50, t=80)
    )
    
    return fig

def main():
    st.markdown('<h1 class="main-header">3D Spectrum Analysis and Visualization</h1>', unsafe_allow_html=True)
    
    # Sidebar for inputs
    with st.sidebar:
        st.header("Input Parameters")
        
        # Model upload
        model_file = st.file_uploader("Upload Trained Model (PKL)", type=['pkl'])
        
        # Spectrum files upload
        spectrum_files = st.file_uploader("Upload Spectrum Files (TXT)", type=['txt'], accept_multiple_files=True)
        
        # KNN parameter
        knn_neighbors = st.slider("Number of KNN Neighbors", min_value=1, max_value=20, value=5)
        
        # Process button
        process_btn = st.button("Process Spectra", type="primary")
    
    # Main content area
    if model_file and spectrum_files and process_btn:
        # Load model
        with st.spinner("Loading model..."):
            model = load_model(model_file)
        
        if model is None:
            st.error("Failed to load model. Please check the file format.")
            return
        
        # Check if model is 3D
        if model['embedding'].shape[1] != 3:
            st.warning("Warning: The loaded model is not 3D. Some visualizations may be limited.")
            use_3d = False
        else:
            use_3d = True
            st.success("3D UMAP model loaded successfully!")
        
        # Process spectra
        new_spectra_data = []
        new_formulas = []
        new_params = []
        new_filenames = []
        new_embeddings = []
        new_pca_components = []
        spectrum_plots = []
        
        with st.spinner("Processing spectra..."):
            for spectrum_file in spectrum_files:
                try:
                    # Read file content
                    content = spectrum_file.getvalue().decode("utf-8")
                    
                    # Parse spectrum
                    spectrum_data, interpolated, formula, params, filename = parse_spectrum_content(
                        content, spectrum_file.name, model['reference_frequencies']
                    )
                    
                    # Transform the spectrum
                    X_scaled = model['scaler'].transform([interpolated])
                    X_pca = model['pca'].transform(X_scaled)
                    X_umap = model['umap'].transform(X_pca)
                    
                    new_spectra_data.append(interpolated)
                    new_formulas.append(formula)
                    new_params.append(params)
                    new_filenames.append(filename)
                    new_embeddings.append(X_umap[0])
                    new_pca_components.append(X_pca[0])
                    
                    # Create spectrum plot
                    spectrum_plot = create_spectrum_plot(
                        spectrum_data, interpolated, model['reference_frequencies'], formula, filename
                    )
                    spectrum_plots.append(spectrum_plot)
                    
                except Exception as e:
                    st.error(f"Error processing {spectrum_file.name}: {str(e)}")
                    continue
        
        if not new_embeddings:
            st.error("No valid spectra found for prediction")
            return
        
        new_embeddings = np.array(new_embeddings)
        new_params = np.array(new_params)
        new_formulas = np.array(new_formulas)
        new_pca_components = np.array(new_pca_components)
        
        st.success(f"Successfully processed {len(new_embeddings)} spectra!")
        
        # Find KNN neighbors
        with st.spinner("Finding nearest neighbors..."):
            knn_indices = find_knn_neighbors(model['embedding'], new_embeddings, k=knn_neighbors)
        
        # Display results
        st.markdown('<h2 class="sub-header">Results</h2>', unsafe_allow_html=True)
        
        # Create tabs for different visualizations
        tab1, tab2, tab3, tab4, tab5 = st.tabs([
            "Parameter Visualizations", 
            "Formula Visualization", 
            "KNN Analysis", 
            "Spectrum Plots", 
            "Data Tables"
        ])
        
        with tab1:
            st.markdown("### Parameter Visualizations")
            
            param_names = ['logn', 'tex', 'velo', 'fwhm']
            param_labels = ['log(n)', 'T_ex (K)', 'Velocity (km/s)', 'FWHM (km/s)']
            
            for i, (param_name, param_label) in enumerate(zip(param_names, param_labels)):
                if use_3d:
                    fig = create_3d_scatter(model, new_embeddings, new_formulas, new_params, i, param_name, param_label)
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.warning(f"3D visualization not available for 2D model: {param_name}")
        
        with tab2:
            st.markdown("### Formula Visualization")
            
            if use_3d:
                fig = create_formula_plot(model, new_embeddings, new_formulas)
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning("3D formula visualization not available for 2D model")
        
        with tab3:
            st.markdown("### KNN Analysis")
            
            if use_3d:
                fig = create_knn_plot(model, new_embeddings, new_formulas, knn_indices)
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning("3D KNN visualization not available for 2D model")
            
            # Display KNN tables for each prediction
            for i in range(len(new_embeddings)):
                if i < len(knn_indices) and knn_indices[i]:
                    st.markdown(f"**KNN Neighbors for {new_formulas[i]} ({new_filenames[i]})**")
                    
                    # Create table data
                    table_data = []
                    for idx in knn_indices[i]:
                        table_data.append([
                            model['formulas'][idx],
                            f"{model['y'][idx, 0]:.2f}",
                            f"{model['y'][idx, 1]:.2f}",
                            f"{model['y'][idx, 2]:.2f}",
                            f"{model['y'][idx, 3]:.2f}"
                        ])
                    
                    # Create DataFrame
                    df = pd.DataFrame(
                        table_data,
                        columns=['Formula', 'log(n)', 'T_ex (K)', 'Velocity', 'FWHM']
                    )
                    
                    st.dataframe(df, use_container_width=True)
        
        with tab4:
            st.markdown("### Spectrum Plots")
            
            for i, plot in enumerate(spectrum_plots):
                st.plotly_chart(plot, use_container_width=True)
        
        with tab5:
            st.markdown("### Data Tables")
            
            # Create prediction coordinates table
            prediction_data = []
            for i in range(len(new_embeddings)):
                prediction_data.append({
                    'Filename': new_filenames[i],
                    'Formula': new_formulas[i],
                    'UMAP X': new_embeddings[i, 0],
                    'UMAP Y': new_embeddings[i, 1],
                    'UMAP Z': new_embeddings[i, 2] if use_3d else 0,
                    'log(n)': new_params[i, 0],
                    'T_ex (K)': new_params[i, 1],
                    'Velocity': new_params[i, 2],
                    'FWHM': new_params[i, 3]
                })
            
            df_predictions = pd.DataFrame(prediction_data)
            st.dataframe(df_predictions, use_container_width=True)
            
            # Add download button for predictions
            csv = df_predictions.to_csv(index=False)
            b64 = base64.b64encode(csv.encode()).decode()
            href = f'<a href="data:file/csv;base64,{b64}" download="predictions.csv">Download Predictions CSV</a>'
            st.markdown(href, unsafe_allow_html=True)
            
            # Display model information
            st.markdown("#### Model Information")
            model_info = {
                'Training Samples': model['sample_size'],
                'Number of Components': model['n_components'],
                'Variance Threshold': model['variance_threshold'],
                'Is 3D Model': use_3d
            }
            st.json(model_info)
    
    else:
        # Display instructions when no files are uploaded
        st.info("""
        ## Instructions:
        1. Upload a trained model file (.pkl) in the sidebar
        2. Upload one or more spectrum files (.txt) for analysis
        3. Adjust the number of KNN neighbors if needed
        4. Click the 'Process Spectra' button to generate visualizations
        
        ## Expected Spectrum Format:
        Spectra can be in one of these formats:
        
        **Format 1 (with header):**
        ```
        // molecules='C2H5OH' logn=13.5 tex=300.0 velo=0.0 fwhm=3.0
        84.0797306920    0.000000e+00
        84.0802239790    0.000000e+00
        ```
        
        **Format 2 (with column headers):**
        ```
        !xValues(GHz)    yValues(K)
        84.0797306920    0.000000e+00
        84.0802239790    0.000000e+00
        ```
        
        **Format 3 (just data):**
        ```
        84.0797306920    0.000000e+00
        84.0802239790    0.000000e+00
        ```
        
        Frequencies can be in Hz or GHz (automatically converted).
        """)

if __name__ == "__main__":
    main()
