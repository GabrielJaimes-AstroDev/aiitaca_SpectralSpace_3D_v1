# molecular_spectrum_analyzer.py
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

# Set page configuration
st.set_page_config(
    page_title="3D Molecular Spectrum Analyzer",
    page_icon="🧪",
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
    .plot-container {
        background-color: #FAFAFA;
        padding: 1.5rem;
        border-radius: 0.5rem;
        margin-bottom: 1.5rem;
    }
</style>
""", unsafe_allow_html=True)

def load_model(model_file):
    """Load the trained model from a pickle file"""
    try:
        model = pickle.load(model_file)
        return model
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None

def sanitize_filename(filename):
    """Remove invalid characters from filenames"""
    invalid_chars = r'[<>:"/\\|?*]'
    return re.sub(invalid_chars, '_', filename)

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

def process_uploaded_spectrum(file, reference_frequencies):
    """Process an uploaded spectrum file"""
    try:
        content = file.getvalue().decode("utf-8")
        lines = content.split('\n')
        
        # Determine file format
        first_line = lines[0].strip()
        second_line = lines[1].strip() if len(lines) > 1 else ""
        
        formula = "Unknown"
        param_dict = {}
        data_start_line = 0
        
        # Format 1: with molecule and parameters header
        if first_line.startswith('//') and 'molecules=' in first_line:
            header = first_line[2:].strip()  # Remove the '//'
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
            # Try to extract information from header if available
            if 'molecules=' in first_line:
                formula = extract_molecule_formula(first_line)
            data_start_line = 1
        
        # Format 3: no header, just data
        else:
            data_start_line = 0
            formula = file.name.split('.')[0]  # Use filename as formula

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
            st.error("No valid data points found in spectrum file")
            return None, None, None, None, None

        spectrum_data = np.array(spectrum_data)

        # Adjust frequency if in GHz (convert to Hz)
        if np.max(spectrum_data[:, 0]) < 1e11:  # If frequencies are less than 100 GHz, probably in GHz
            spectrum_data[:, 0] = spectrum_data[:, 0] * 1e9  # Convert GHz to Hz
            st.info(f"Converted frequencies from GHz to Hz for {file.name}")

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

        return spectrum_data, interpolated, formula, params, file.name
        
    except Exception as e:
        st.error(f"Error processing spectrum file: {str(e)}")
        return None, None, None, None, None

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
        # Verify indices are within valid range
        valid_indices = [idx for idx in indices[0] if idx < len(training_embeddings)]
        all_neighbor_indices.append(valid_indices)
    
    return all_neighbor_indices

def create_3d_scatter(embeddings, color_values, title, color_label, color_scale='viridis', 
                      marker_size=5, selected_indices=None, selected_color='red', selected_size=10):
    """Create an interactive 3D scatter plot"""
    fig = go.Figure()
    
    # Create main scatter plot
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
            colorbar=dict(title=color_label),
            line=dict(width=0)
        ),
        text=[f"Index: {i}" for i in range(len(embeddings))],
        hovertemplate=
        '<b>X</b>: %{x}<br>' +
        '<b>Y</b>: %{y}<br>' +
        '<b>Z</b>: %{z}<br>' +
        '<b>Value</b>: %{marker.color}<br>' +
        '<extra></extra>',
        name='Data points'
    ))
    
    # Highlight selected points if provided
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
            name='Selected points'
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
    """Create an interactive 2D scatter plot"""
    fig = go.Figure()
    
    # Create main scatter plot
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
    
    # Highlight selected points if provided
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
    """Create a spectrum plot"""
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
    # Header
    st.markdown('<h1 class="main-header">🧪 3D Molecular Spectrum Analyzer</h1>', unsafe_allow_html=True)
    
    # Sidebar
    st.sidebar.title("Configuration")
    
    # File uploaders
    st.sidebar.subheader("Upload Model and Spectra")
    model_file = st.sidebar.file_uploader("Upload trained model (PKL file)", type="pkl")
    
    if model_file is not None:
        # Load model
        with st.spinner("Loading model..."):
            model = load_model(model_file)
        
        if model is not None:
            st.sidebar.success("Model loaded successfully!")
            
            # Display model info
            st.sidebar.subheader("Model Information")
            st.sidebar.write(f"Training samples: {model.get('sample_size', 'N/A')}")
            st.sidebar.write(f"Number of components: {model.get('n_components', 'N/A')}")
            st.sidebar.write(f"Variance threshold: {model.get('variance_threshold', 'N/A')}")
            st.sidebar.write(f"3D Model: {model['embedding'].shape[1] == 3}")
            
            # Upload spectra files
            spectra_files = st.sidebar.file_uploader("Upload spectrum files (TXT)", type="txt", accept_multiple_files=True)
            
            # KNN neighbors parameter
            knn_neighbors = st.sidebar.slider("Number of KNN neighbors", min_value=1, max_value=20, value=5)
            
            if spectra_files and len(spectra_files) > 0:
                # Process spectra
                with st.spinner("Processing spectra..."):
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
                    
                    if len(new_embeddings) > 0:
                        new_embeddings = np.array(new_embeddings)
                        new_params = np.array(new_params)
                        new_formulas = np.array(new_formulas)
                        new_pca_components = np.array(new_pca_components)
                        
                        # Find KNN neighbors
                        knn_indices = find_knn_neighbors(model['embedding'], new_embeddings, k=knn_neighbors)
                        
                        # Display success message
                        st.sidebar.success(f"Processed {len(new_embeddings)} spectra successfully!")
                        
                        # Main content
                        st.markdown('<div class="info-box">', unsafe_allow_html=True)
                        st.write(f"**Analysis Results:** {len(new_embeddings)} spectra processed and projected into 3D space")
                        st.markdown('</div>', unsafe_allow_html=True)
                        
                        # Create tabs for different visualizations
                        tab1, tab2, tab3, tab4 = st.tabs(["3D Projection", "2D Projection", "Spectrum View", "KNN Analysis"])
                        
                        with tab1:
                            st.markdown('<h2 class="sub-header">3D UMAP Projection</h2>', unsafe_allow_html=True)
                            
                            # Parameter selection for coloring
                            param_options = ['logn', 'tex', 'velo', 'fwhm', 'formula']
                            color_param = st.selectbox("Color by", param_options, index=4)
                            
                            # Create combined data for plotting
                            combined_embeddings = np.vstack([model['embedding'], new_embeddings])
                            
                            if color_param == 'formula':
                                # For formula coloring, we need to create a numeric mapping
                                all_formulas = np.concatenate([model['formulas'], new_formulas])
                                unique_formulas = np.unique(all_formulas)
                                formula_to_num = {formula: i for i, formula in enumerate(unique_formulas)}
                                color_values = np.array([formula_to_num[f] for f in all_formulas])
                                color_label = "Formula"
                                color_scale = 'viridis'
                            else:
                                param_idx = param_options.index(color_param)
                                if param_idx < 4:  # It's a parameter
                                    color_values = np.concatenate([model['y'][:, param_idx], new_params[:, param_idx]])
                                    color_label = param_options[param_idx]
                                    color_scale = 'plasma'
                            
                            # Create the plot
                            selected_indices = list(range(len(model['embedding']), len(combined_embeddings)))
                            fig_3d = create_3d_scatter(
                                combined_embeddings, 
                                color_values, 
                                "3D UMAP Projection (Training + New Spectra)", 
                                color_label,
                                color_scale=color_scale,
                                selected_indices=selected_indices
                            )
                            
                            st.plotly_chart(fig_3d, use_container_width=True)
                            
                            # Display information about the new spectra
                            st.markdown('<h3 class="sub-header">New Spectrum Details</h3>', unsafe_allow_html=True)
                            
                            for i in range(len(new_embeddings)):
                                col1, col2 = st.columns([1, 2])
                                
                                with col1:
                                    st.write(f"**Spectrum {i+1}:** {new_filenames[i]}")
                                    st.write(f"**Formula:** {new_formulas[i]}")
                                    st.write(f"**log(n):** {new_params[i, 0]:.2f}")
                                    st.write(f"**T_ex (K):** {new_params[i, 1]:.2f}")
                                    st.write(f"**Velocity:** {new_params[i, 2]:.2f}")
                                    st.write(f"**FWHM:** {new_params[i, 3]:.2f}")
                                
                                with col2:
                                    spectrum_fig = create_spectrum_plot(
                                        model['reference_frequencies'],
                                        new_spectra_data[i],
                                        f"Spectrum: {new_filenames[i]}"
                                    )
                                    st.plotly_chart(spectrum_fig, use_container_width=True)
                        
                        with tab2:
                            st.markdown('<h2 class="sub-header">2D UMAP Projection</h2>', unsafe_allow_html=True)
                            
                            # Parameter selection for coloring
                            color_param_2d = st.selectbox("Color by", param_options, index=4, key='color_2d')
                            
                            if color_param_2d == 'formula':
                                color_values_2d = color_values
                                color_label_2d = "Formula"
                                color_scale_2d = 'viridis'
                            else:
                                param_idx = param_options.index(color_param_2d)
                                if param_idx < 4:  # It's a parameter
                                    color_values_2d = np.concatenate([model['y'][:, param_idx], new_params[:, param_idx]])
                                    color_label_2d = param_options[param_idx]
                                    color_scale_2d = 'plasma'
                            
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
                            
                            # Let user select which spectrum to view
                            spectrum_idx = st.selectbox("Select spectrum", range(len(new_embeddings)), 
                                                      format_func=lambda x: new_filenames[x])
                            
                            if spectrum_idx is not None:
                                col1, col2 = st.columns(2)
                                
                                with col1:
                                    # Show the selected spectrum
                                    spectrum_fig = create_spectrum_plot(
                                        model['reference_frequencies'],
                                        new_spectra_data[spectrum_idx],
                                        f"Spectrum: {new_filenames[spectrum_idx]}"
                                    )
                                    st.plotly_chart(spectrum_fig, use_container_width=True)
                                
                                with col2:
                                    # Show KNN neighbors if available
                                    if knn_indices and len(knn_indices) > spectrum_idx:
                                        neighbor_indices = knn_indices[spectrum_idx]
                                        
                                        if neighbor_indices:
                                            st.write("**K-Nearest Neighbors:**")
                                            
                                            # Create a DataFrame for the neighbors
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
                            
                            # Show KNN analysis for each spectrum
                            for i in range(len(new_embeddings)):
                                st.markdown(f"**{new_filenames[i]}** ({new_formulas[i]})")
                                
                                if knn_indices and len(knn_indices) > i:
                                    neighbor_indices = knn_indices[i]
                                    
                                    if neighbor_indices:
                                        # Create a DataFrame for the neighbors
                                        neighbor_data = []
                                        for idx in neighbor_indices:
                                            neighbor_data.append({
                                                'Formula': model['formulas'][idx],
                                                'log(n)': f"{model['y'][idx, 0]:.2f}",
                                                'T_ex (K)': f"{model['y'][idx, 1]:.2f}",
                                                'Velocity': f"{model['y'][idx, 2]:.2f}",
                                                'FWHM': f"{model['y'][idx, 3]:.2f}",
                                                'Distance': f"{np.linalg.norm(model['embedding'][idx] - new_embeddings[i]):.4f}"
                                            })
                                        
                                        neighbor_df = pd.DataFrame(neighbor_data)
                                        st.dataframe(neighbor_df, use_container_width=True)
                                        
                                        # Show average parameters
                                        st.write("**Average parameters of neighbors:**")
                                        avg_params = {
                                            'log(n)': np.mean([model['y'][idx, 0] for idx in neighbor_indices]),
                                            'T_ex (K)': np.mean([model['y'][idx, 1] for idx in neighbor_indices]),
                                            'Velocity': np.mean([model['y'][idx, 2] for idx in neighbor_indices]),
                                            'FWHM': np.mean([model['y'][idx, 3] for idx in neighbor_indices])
                                        }
                                        
                                        avg_df = pd.DataFrame([avg_params])
                                        st.dataframe(avg_df, use_container_width=True)
                                        
                                        # Compare with the new spectrum
                                        comparison_data = {
                                            'Parameter': ['log(n)', 'T_ex (K)', 'Velocity', 'FWHM'],
                                            'New Spectrum': [new_params[i, 0], new_params[i, 1], new_params[i, 2], new_params[i, 3]],
                                            'Neighbors Average': [avg_params['log(n)'], avg_params['T_ex (K)'], avg_params['Velocity'], avg_params['FWHM']],
                                            'Difference': [
                                                new_params[i, 0] - avg_params['log(n)'],
                                                new_params[i, 1] - avg_params['T_ex (K)'],
                                                new_params[i, 2] - avg_params['Velocity'],
                                                new_params[i, 3] - avg_params['FWHM']
                                            ]
                                        }
                                        
                                        comparison_df = pd.DataFrame(comparison_data)
                                        st.dataframe(comparison_df, use_container_width=True)
                                
                                st.markdown("---")
                    else:
                        st.error("No valid spectra could be processed. Please check your files.")
            else:
                st.info("Please upload spectrum files to analyze.")
        else:
            st.error("Failed to load the model. Please check the file format.")
    else:
        st.info("Please upload a trained model file to begin analysis.")
        
        # Show instructions
        st.markdown("""
        ### How to use this application:
        
        1. **Upload a trained model** - This should be a PKL file containing a trained PCA/UMAP model
        2. **Upload spectrum files** - Text files containing molecular spectrum data
        3. **Adjust parameters** - Use the sidebar to configure the analysis
        4. **Explore the results** - Use the tabs to view different visualizations and analyses
        
        The application will project your spectra into 3D space and help you analyze their properties
        and relationships to other spectra in the dataset.
        """)

if __name__ == "__main__":
    main()
