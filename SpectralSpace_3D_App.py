import os
import pickle
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
import re
from sklearn.neighbors import NearestNeighbors
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import base64
from io import BytesIO

# Configuración de la página
st.set_page_config(
    page_title="Visualización 3D de Espectros Moleculares",
    page_icon="🧪",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Función para sanitizar nombres de archivo
def sanitize_filename(filename):
    invalid_chars = r'[<>:"/\\|?*]'
    return re.sub(invalid_chars, '_', filename)

# Función para cargar el modelo
@st.cache_resource
def load_model(model_path):
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    return model

# Función para extraer fórmula molecular
def extract_molecule_formula(header):
    pattern = r"molecules=['\"]([^,'\"]+)"
    match = re.search(pattern, header)
    if match:
        formula = match.group(1)
        if ',' in formula:
            formula = formula.split(',')[0]
        return formula
    return "Unknown"

# Función para cargar e interpolar espectros
def load_and_interpolate_spectrum(filepath, reference_frequencies):
    with open(filepath, 'r') as f:
        lines = f.readlines()

    first_line = lines[0].strip()
    second_line = lines[1].strip() if len(lines) > 1 else ""
    
    formula = "Unknown"
    param_dict = {}
    data_start_line = 0
    
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
    
    elif first_line.startswith('!') or first_line.startswith('#'):
        if 'molecules=' in first_line:
            formula = extract_molecule_formula(first_line)
        data_start_line = 1
    
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
                    freq_str = parts[0].replace('D', 'E').replace('d', 'E')
                    intensity_str = parts[1].replace('D', 'E').replace('d', 'E')
                    freq = float(freq_str)
                    intensity = float(intensity_str)
                
                if np.isfinite(freq) and np.isfinite(intensity):
                    spectrum_data.append([freq, intensity])
        except Exception as e:
            continue

    if not spectrum_data:
        raise ValueError("No valid data points found in spectrum file")

    spectrum_data = np.array(spectrum_data)

    if np.max(spectrum_data[:, 0]) < 1e11:
        spectrum_data[:, 0] = spectrum_data[:, 0] * 1e9

    interpolator = interp1d(spectrum_data[:, 0], spectrum_data[:, 1],
                            kind='linear', bounds_error=False, fill_value=0.0)
    interpolated = interpolator(reference_frequencies)

    params = [
        param_dict.get('logn', np.nan),
        param_dict.get('tex', np.nan),
        param_dict.get('velo', np.nan),
        param_dict.get('fwhm', np.nan)
    ]

    return spectrum_data, interpolated, formula, params, os.path.basename(filepath)

# Función para encontrar vecinos KNN
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

# Función para crear gráfico 3D interactivo
def create_3d_plot(embeddings, formulas, param_values, param_name, param_label, new_embeddings=None, 
                  new_formulas=None, new_param_values=None, title="3D UMAP Visualization"):
    
    fig = go.Figure()
    
    # Training data
    if param_values is not None:
        fig.add_trace(go.Scatter3d(
            x=embeddings[:, 0],
            y=embeddings[:, 1],
            z=embeddings[:, 2],
            mode='markers',
            marker=dict(
                size=4,
                color=param_values,
                colorscale='Viridis',
                opacity=0.6,
                colorbar=dict(title=param_label)
            ),
            text=formulas,
            hovertemplate='<b>%{text}</b><br>' +
                         f'{param_label}: %{{marker.color:.2f}}<br>' +
                         'UMAP1: %{x:.2f}<br>' +
                         'UMAP2: %{y:.2f}<br>' +
                         'UMAP3: %{z:.2f}<extra></extra>',
            name='Training Data'
        ))
    else:
        # Color by formula
        unique_formulas = np.unique(formulas)
        colors = plt.cm.tab20(np.linspace(0, 1, len(unique_formulas)))
        
        for i, formula in enumerate(unique_formulas):
            mask = formulas == formula
            color = f'rgb({int(colors[i][0]*255)}, {int(colors[i][1]*255)}, {int(colors[i][2]*255)})'
            
            fig.add_trace(go.Scatter3d(
                x=embeddings[mask, 0],
                y=embeddings[mask, 1],
                z=embeddings[mask, 2],
                mode='markers',
                marker=dict(
                    size=4,
                    color=color,
                    opacity=0.6
                ),
                text=formula,
                hovertemplate='<b>%{text}</b><br>' +
                             'UMAP1: %{x:.2f}<br>' +
                             'UMAP2: %{y:.2f}<br>' +
                             'UMAP3: %{z:.2f}<extra></extra>',
                name=formula
            ))
    
    # New predictions
    if new_embeddings is not None and len(new_embeddings) > 0:
        if new_param_values is not None:
            fig.add_trace(go.Scatter3d(
                x=new_embeddings[:, 0],
                y=new_embeddings[:, 1],
                z=new_embeddings[:, 2],
                mode='markers',
                marker=dict(
                    size=8,
                    color=new_param_values,
                    colorscale='Plasma',
                    opacity=1.0,
                    symbol='diamond'
                ),
                text=new_formulas,
                hovertemplate='<b>%{text}</b><br>' +
                             f'{param_label}: %{{marker.color:.2f}}<br>' +
                             'UMAP1: %{x:.2f}<br>' +
                             'UMAP2: %{y:.2f}<br>' +
                             'UMAP3: %{z:.2f}<extra></extra>',
                name='New Predictions'
            ))
        else:
            for i, formula in enumerate(np.unique(new_formulas)):
                mask = new_formulas == formula
                fig.add_trace(go.Scatter3d(
                    x=new_embeddings[mask, 0],
                    y=new_embeddings[mask, 1],
                    z=new_embeddings[mask, 2],
                    mode='markers',
                    marker=dict(
                        size=8,
                        color='red',
                        opacity=1.0,
                        symbol='diamond'
                    ),
                    text=formula,
                    hovertemplate='<b>%{text}</b><br>' +
                                 'UMAP1: %{x:.2f}<br>' +
                                 'UMAP2: %{y:.2f}<br>' +
                                 'UMAP3: %{z:.2f}<extra></extra>',
                    name=f'{formula} (New)'
                ))
    
    fig.update_layout(
        title=title,
        scene=dict(
            xaxis_title='UMAP 1',
            yaxis_title='UMAP 2',
            zaxis_title='UMAP 3'
        ),
        width=1000,
        height=800,
        margin=dict(l=0, r=0, b=0, t=30)
    )
    
    return fig

# Función para crear gráfico 2D
def create_2d_projection(embeddings, formulas, dim1, dim2, plane, new_embeddings=None, new_formulas=None):
    fig = go.Figure()
    
    # Training data
    fig.add_trace(go.Scatter(
        x=embeddings[:, dim1],
        y=embeddings[:, dim2],
        mode='markers',
        marker=dict(
            size=6,
            color='lightgray',
            opacity=0.5
        ),
        text=formulas,
        hovertemplate='<b>%{text}</b><br>' +
                     f'UMAP{dim1+1}: %{{x:.2f}}<br>' +
                     f'UMAP{dim2+1}: %{{y:.2f}}<extra></extra>',
        name='Training Data'
    ))
    
    # New predictions
    if new_embeddings is not None and len(new_embeddings) > 0:
        fig.add_trace(go.Scatter(
            x=new_embeddings[:, dim1],
            y=new_embeddings[:, dim2],
            mode='markers',
            marker=dict(
                size=12,
                color='red',
                symbol='star'
            ),
            text=new_formulas,
            hovertemplate='<b>%{text}</b><br>' +
                         f'UMAP{dim1+1}: %{{x:.2f}}<br>' +
                         f'UMAP{dim2+1}: %{{y:.2f}}<extra></extra>',
            name='New Predictions'
        ))
    
    fig.update_layout(
        title=f'UMAP {plane} Projection',
        xaxis_title=f'UMAP {dim1+1}',
        yaxis_title=f'UMAP {dim2+1}',
        width=500,
        height=500,
        showlegend=True
    )
    
    return fig

# Función para crear tabla de datos
def create_data_table(data, title):
    df = pd.DataFrame(data)
    st.subheader(title)
    st.dataframe(df, use_container_width=True)

# Función principal de la aplicación
def main():
    st.title("🧪 Visualización 3D de Espectros Moleculares")
    st.markdown("""
    Esta aplicación permite visualizar espectros moleculares en un espacio 3D reducido mediante UMAP.
    Carga un modelo preentrenado y nuevos espectros para ver sus proyecciones en el espacio de características.
    """)
    
    # Sidebar para carga de archivos
    st.sidebar.header("Configuración")
    model_file = st.sidebar.file_uploader("Cargar modelo (.pkl)", type=["pkl"])
    
    if model_file is not None:
        # Guardar el modelo temporalmente
        with open("temp_model.pkl", "wb") as f:
            f.write(model_file.getbuffer())
        
        # Cargar el modelo
        try:
            model = load_model("temp_model.pkl")
            st.sidebar.success("Modelo cargado correctamente")
            
            # Mostrar información del modelo
            st.sidebar.subheader("Información del Modelo")
            st.sidebar.write(f"Muestras de entrenamiento: {model.get('sample_size', 'N/A')}")
            st.sidebar.write(f"Componentes PCA: {model.get('n_components', 'N/A')}")
            st.sidebar.write(f"Varianza explicada: {model.get('variance_threshold', 'N/A')}")
            
            # Cargar nuevos espectros
            new_spectra_files = st.sidebar.file_uploader(
                "Cargar nuevos espectros (.txt)", 
                type=["txt"], 
                accept_multiple_files=True
            )
            
            if new_spectra_files:
                # Crear directorio temporal para los espectros
                os.makedirs("temp_spectra", exist_ok=True)
                for file in new_spectra_files:
                    with open(os.path.join("temp_spectra", file.name), "wb") as f:
                        f.write(file.getbuffer())
                
                # Procesar los nuevos espectros
                ref_freqs = model['reference_frequencies']
                scaler = model['scaler']
                pca = model['pca']
                umap_model = model['umap']
                
                new_spectra_data = []
                new_formulas = []
                new_params = []
                new_filenames = []
                new_embeddings = []
                
                for file in new_spectra_files:
                    filepath = os.path.join("temp_spectra", file.name)
                    try:
                        spectrum_data, interpolated, formula, params, filename = load_and_interpolate_spectrum(filepath, ref_freqs)
                        
                        # Transformar el espectro
                        X_scaled = scaler.transform([interpolated])
                        X_pca = pca.transform(X_scaled)
                        X_umap = umap_model.transform(X_pca)
                        
                        new_spectra_data.append(interpolated)
                        new_formulas.append(formula)
                        new_params.append(params)
                        new_filenames.append(filename)
                        new_embeddings.append(X_umap[0])
                        
                    except Exception as e:
                        st.sidebar.error(f"Error procesando {file.name}: {str(e)}")
                
                if new_embeddings:
                    new_embeddings = np.array(new_embeddings)
                    new_params = np.array(new_params)
                    new_formulas = np.array(new_formulas)
                    
                    st.sidebar.success(f"{len(new_embeddings)} espectros procesados correctamente")
                    
                    # Encontrar vecinos KNN
                    knn_neighbors = st.sidebar.slider("Número de vecinos KNN", 1, 20, 5)
                    knn_indices = find_knn_neighbors(model['embedding'], new_embeddings, k=knn_neighbors)
                    
                    # Mostrar visualizaciones
                    st.header("Visualizaciones 3D")
                    
                    # Seleccionar tipo de visualización
                    viz_type = st.selectbox(
                        "Seleccionar tipo de visualización",
                        ["Por parámetro físico", "Por fórmula molecular", "Proyecciones 2D"]
                    )
                    
                    if viz_type == "Por parámetro físico":
                        param_names = ['logn', 'tex', 'velo', 'fwhm']
                        param_labels = ['log(n)', 'T_ex (K)', 'Velocity (km/s)', 'FWHM (km/s)']
                        
                        selected_param = st.selectbox("Seleccionar parámetro", param_names)
                        param_idx = param_names.index(selected_param)
                        
                        # Crear visualización 3D
                        fig = create_3d_plot(
                            model['embedding'], 
                            model['formulas'], 
                            model['y'][:, param_idx], 
                            selected_param, 
                            param_labels[param_idx],
                            new_embeddings,
                            new_formulas,
                            new_params[:, param_idx],
                            f"3D UMAP: {selected_param} (Training + Predictions)"
                        )
                        
                        st.plotly_chart(fig, use_container_width=True)
                        
                    elif viz_type == "Por fórmula molecular":
                        # Crear visualización 3D por fórmula
                        fig = create_3d_plot(
                            model['embedding'], 
                            model['formulas'], 
                            None, 
                            None, 
                            None,
                            new_embeddings,
                            new_formulas,
                            None,
                            "3D UMAP: Molecular Formula (Training + Predictions)"
                        )
                        
                        st.plotly_chart(fig, use_container_width=True)
                        
                    elif viz_type == "Proyecciones 2D":
                        # Crear proyecciones 2D
                        projections = [(0, 1, 'XY'), (0, 2, 'XZ'), (1, 2, 'YZ')]
                        
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            fig1 = create_2d_projection(
                                model['embedding'],
                                model['formulas'],
                                projections[0][0],
                                projections[0][1],
                                projections[0][2],
                                new_embeddings,
                                new_formulas
                            )
                            st.plotly_chart(fig1, use_container_width=True)
                            
                        with col2:
                            fig2 = create_2d_projection(
                                model['embedding'],
                                model['formulas'],
                                projections[1][0],
                                projections[1][1],
                                projections[1][2],
                                new_embeddings,
                                new_formulas
                            )
                            st.plotly_chart(fig2, use_container_width=True)
                        
                        fig3 = create_2d_projection(
                            model['embedding'],
                            model['formulas'],
                            projections[2][0],
                            projections[2][1],
                            projections[2][2],
                            new_embeddings,
                            new_formulas
                        )
                        st.plotly_chart(fig3, use_container_width=True)
                    
                    # Mostrar datos tabulares
                    st.header("Datos de Predicción")
                    
                    # Crear tabla de predicciones
                    prediction_data = []
                    for i in range(len(new_embeddings)):
                        prediction_data.append({
                            'Archivo': new_filenames[i],
                            'Fórmula': new_formulas[i],
                            'log(n)': f"{new_params[i, 0]:.2f}",
                            'T_ex (K)': f"{new_params[i, 1]:.2f}",
                            'Velocity': f"{new_params[i, 2]:.2f}",
                            'FWHM': f"{new_params[i, 3]:.2f}",
                            'UMAP X': f"{new_embeddings[i, 0]:.4f}",
                            'UMAP Y': f"{new_embeddings[i, 1]:.4f}",
                            'UMAP Z': f"{new_embeddings[i, 2]:.4f}"
                        })
                    
                    create_data_table(prediction_data, "Resultados de Predicción")
                    
                    # Mostrar vecinos KNN para cada predicción
                    st.subheader("Vecinos más Cercanos (KNN)")
                    
                    for i in range(len(new_embeddings)):
                        with st.expander(f"Vecinos para {new_formulas[i]} ({new_filenames[i]})"):
                            if i < len(knn_indices) and knn_indices[i]:
                                neighbor_data = []
                                for idx in knn_indices[i]:
                                    neighbor_data.append({
                                        'Fórmula': model['formulas'][idx],
                                        'log(n)': f"{model['y'][idx, 0]:.2f}",
                                        'T_ex (K)': f"{model['y'][idx, 1]:.2f}",
                                        'Velocity': f"{model['y'][idx, 2]:.2f}",
                                        'FWHM': f"{model['y'][idx, 3]:.2f}",
                                        'Distancia': f"{np.linalg.norm(new_embeddings[i] - model['embedding'][idx]):.4f}"
                                    })
                                
                                neighbor_df = pd.DataFrame(neighbor_data)
                                st.dataframe(neighbor_df, use_container_width=True)
                            else:
                                st.write("No se encontraron vecinos cercanos.")
                
                else:
                    st.warning("No se pudieron procesar los espectros. Verifique el formato de los archivos.")
            
        except Exception as e:
            st.sidebar.error(f"Error cargando el modelo: {str(e)}")
    
    else:
        st.info("Por favor, cargue un modelo preentrenado para comenzar.")

if __name__ == "__main__":
    main()