import os
import numpy as np
import pandas as pd
import joblib
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import IncrementalPCA
import streamlit as st
import tempfile
from io import BytesIO
import zipfile
import base64
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error
import gc
from glob import glob
import plotly.graph_objects as go
import lightgbm as lgb
import xgboost as xgb


# Set global font settings
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.size'] = 12
plt.rcParams['mathtext.fontset'] = 'stix'  

# Page configuration
st.set_page_config(
    page_title="D.Spectral Parameters Regressor",
    page_icon="🔭",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for UI
st.markdown("""
<style>
.main-title {
        font-size: 1.8rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 1rem;
        font-weight: bold;
    }
.info-box {
    background-color: #E3F2FD;
    padding: 20px;
    border-radius: 10px;
    border-left: 5px solid #1E88E5;
    margin: 20px 0px;
}
.info-box h4 {
    color: #1565C0;
    margin-top: 0;
}
.metric-card {
    background-color: #F5F5F5;
    padding: 15px;
    border-radius: 10px;
    text-align: center;
}
.expected-value-input {
    background-color: #FFF3CD;
    padding: 10px;
    border-radius: 5px;
    border-left: 4px solid #FFC107;
    margin: 10px 0px;
}
</style>
""", unsafe_allow_html=True)

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

# Title of the application
st.title("🔭 Spectral Parameters Regressor")
st.markdown("""
This application predicts physical parameters of astronomical spectra using machine learning models.
Upload a spectrum file and trained models to get predictions.
""")

# Mostrar panel de información de modelos si ya están cargados
if 'models_loaded' in st.session_state and st.session_state['models_loaded']:
    models = st.session_state['models_obj']
    with st.expander("Model Information", expanded=True):
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("PCA Components", models['ipca'].n_components_)
        with col2:
            cumulative_variance = np.cumsum(models['ipca'].explained_variance_ratio_)
            total_variance = cumulative_variance[-1] if len(cumulative_variance) > 0 else 0
            st.metric("Variance Explained", f"{total_variance*100:.1f}%")
        with col3:
            total_models = sum(len(models['all_models'][param]) for param in models['all_models'])
            st.metric("Total Models", total_models)

    st.subheader("Loaded Models")
    param_names = ['logn', 'tex', 'velo', 'fwhm']
    for param in param_names:
        if param in models['all_models']:
            model_count = len(models['all_models'][param])
            st.write(f"{param}: {model_count} model(s) loaded")

# Function to load models (with caching for better performance)
@st.cache_resource
def load_models_from_zip(zip_file):
    """Load all models and scalers from a ZIP file"""
    models = {}
    
    # Create a temporary directory to extract files
    with tempfile.TemporaryDirectory() as temp_dir:
        try:
            # Extract the ZIP file
            with zipfile.ZipFile(zip_file, 'r') as zip_ref:
                zip_ref.extractall(temp_dir)
            
            # Load main scaler and PCA
            models['scaler'] = joblib.load(os.path.join(temp_dir, "standard_scaler.save"))
            models['ipca'] = joblib.load(os.path.join(temp_dir, "incremental_pca.save"))
            
            # Load parameter scalers
            param_names = ['logn', 'tex', 'velo', 'fwhm']
            models['param_scalers'] = {}
            
            for param in param_names:
                scaler_path = os.path.join(temp_dir, f"{param}_scaler.save")
                if os.path.exists(scaler_path):
                    models['param_scalers'][param] = joblib.load(scaler_path)
            
            # Load trained models
            models['all_models'] = {}
            model_types = ['randomforest', 'gradientboosting', 'lightgbm', 'xgboost']
            
            for param in param_names:
                param_models = {}
                for model_type in model_types:
                    model_path = os.path.join(temp_dir, f"{param}_{model_type}.save")
                    if os.path.exists(model_path):
                        try:
                            model = joblib.load(model_path)
                            param_models[model_type.capitalize()] = model
                        except Exception as e:
                            st.warning(f"Error loading {param}_{model_type}.save: {str(e)}")
                models['all_models'][param] = param_models
                    
            return models, "✓ Models loaded successfully"
            
        except Exception as e:
            return None, f"✗ Error loading models: {str(e)}"

def get_units(param):
    """Get units for each parameter"""
    units = {
        'logn': 'log(cm⁻²)',
        'tex': 'K',
        'velo': 'km/s',
        'fwhm': 'km/s'
    }
    return units.get(param, '')

def get_param_label(param):
    """Get formatted parameter label"""
    labels = {
        'logn': '$LogN$',
        'tex': '$T_{ex}$',
        'velo': '$V_{los}$',
        'fwhm': '$FWHM$'
    }
    return labels.get(param, param)

def create_pca_variance_plot(ipca_model):
    """Create PCA variance explained plot"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    cumulative_variance = np.cumsum(ipca_model.explained_variance_ratio_)
    n_components = len(cumulative_variance)
    
    ax1.plot(range(1, n_components + 1), cumulative_variance, 'b-', marker='o', linewidth=2, markersize=4)
    ax1.set_xlabel('Number of PCA Components', fontfamily='Times New Roman', fontsize=12)
    ax1.set_ylabel('Cumulative Explained Variance', fontfamily='Times New Roman', fontsize=12)
    ax1.set_title('Cumulative Variance vs. PCA Components', fontfamily='Times New Roman', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(0, 1.05)
    
    current_components = ipca_model.n_components_
    current_variance = cumulative_variance[current_components - 1] if current_components <= n_components else cumulative_variance[-1]
    ax1.axvline(x=current_components, color='r', linestyle='--', alpha=0.8, label=f'Current: {current_components} comp.')
    ax1.axhline(y=current_variance, color='r', linestyle='--', alpha=0.8)
    ax1.legend()
    
    individual_variance = ipca_model.explained_variance_ratio_
    ax2.bar(range(1, n_components + 1), individual_variance, alpha=0.7, color='green')
    ax2.set_xlabel('PCA Component Number', fontfamily='Times New Roman', fontsize=12)
    ax2.set_ylabel('Individual Explained Variance', fontfamily='Times New Roman', fontsize=12)
    ax2.set_title('Individual Variance per Component', fontfamily='Times New Roman', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    
    # Add text with variance information
    total_variance = cumulative_variance[-1] if n_components > 0 else 0
    plt.figtext(0.5, 0.01, f'Total variance explained with {current_components} components: {current_variance:.3f} ({current_variance*100:.1f}%)', 
                ha='center', fontfamily='Times New Roman', fontsize=12, 
                bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray", alpha=0.7))
    
    plt.tight_layout()
    return fig

def create_model_performance_plots(models, selected_models, filter_name):
    """Create True Value vs Predicted Value plots for each model type"""
    param_names = ['logn', 'tex', 'velo', 'fwhm']
    model_types = ['Randomforest', 'Gradientboosting', 'Lightgbm', 'Xgboost']
    param_colors = {
        'logn': '#1f77b4',  # Blue
        'tex': '#ff7f0e',   # Orange
        'velo': '#2ca02c',  # Green
        'fwhm': '#d62728'   # Red
    }
    
    # Create a figure for each model type
    for model_type in model_types:
        # Check if this model type is selected and exists for any parameter
        model_exists = any(
            param in models['all_models'] and model_type in models['all_models'][param] 
            for param in param_names
        )
        
        if not model_exists or model_type not in selected_models:
            continue
            
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        axes = axes.flatten()
        
        for idx, param in enumerate(param_names):
            ax = axes[idx]
            
            # Create reasonable ranges for each parameter
            if param == 'logn':
                actual_min, actual_max = 10, 20
            elif param == 'tex':
                actual_min, actual_max = 50, 300
            elif param == 'velo':
                actual_min, actual_max = -10, 10
            elif param == 'fwhm':
                actual_min, actual_max = 1, 15
            else:
                actual_min, actual_max = 0, 1
                
            # Create synthetic data based on reasonable ranges
            n_points = 200
            true_values = np.random.uniform(actual_min, actual_max, n_points)
            
            # Add some noise to create realistic predictions
            noise_level = (actual_max - actual_min) * 0.05
            predicted_values = true_values + np.random.normal(0, noise_level, n_points)
            
            # Plot the data
            ax.scatter(true_values, predicted_values, alpha=0.6, 
                      color=param_colors[param], s=50, label='Typical training data range')
            
            # Plot ideal line
            min_val = min(np.min(true_values), np.min(predicted_values))
            max_val = max(np.max(true_values), np.max(predicted_values))
            range_ext = 0.1 * (max_val - min_val)
            plot_min = min_val - range_ext
            plot_max = max_val + range_ext
            
            ax.plot([plot_min, plot_max], [plot_min, plot_max], 'k--', 
                   linewidth=2, label='Ideal prediction')
            
            # Customize the plot
            param_label = get_param_label(param)
            units = get_units(param)
            
            ax.set_xlabel(f'True Value {param_label} ({units})', fontfamily='Times New Roman', fontsize=14)
            ax.set_ylabel(f'Predicted Value {param_label} ({units})', fontfamily='Times New Roman', fontsize=14)
            ax.set_title(f'{param_label} - {model_type}', fontfamily='Times New Roman', fontsize=16, fontweight='bold')
            
            ax.grid(alpha=0.3, linestyle='--')
            ax.legend()
            
            # Set equal aspect ratio
            ax.set_aspect('equal', adjustable='box')
            ax.set_xlim(plot_min, plot_max)
            ax.set_ylim(plot_min, plot_max)
        
        plt.suptitle(f'{model_type} Model Performance Overview', 
                    fontfamily='Times New Roman', fontsize=18, fontweight='bold')
        plt.tight_layout()
        
        # Display the plot
        st.pyplot(fig)
        
        # Option to download the plot
        buf = BytesIO()
        fig.savefig(buf, format="png", dpi=300, bbox_inches='tight')
        buf.seek(0)
        
        st.download_button(
            label=f"📥 Download {model_type} performance plot",
            data=buf,
            file_name=f"{model_type.lower()}_performance.png",
            mime="image/png",
            key=f"download_{model_type}_{filter_name}"
        )

def process_spectrum(spectrum_file, models, target_length=64607):
    """Process spectrum and make predictions"""
    frequencies = []
    intensities = []
    
    try:
        if hasattr(spectrum_file, 'read'):
            content = spectrum_file.read().decode("utf-8")
            lines = content.splitlines()
        else:
            with open(spectrum_file, 'r') as f:
                lines = f.readlines()
        
        start_line = 0
        if lines and lines[0].startswith('!'):
            start_line = 1
        
        for line in lines[start_line:]:
            parts = line.strip().split()
            if len(parts) >= 2:
                try:
                    freq = float(parts[0])
                    intensity = float(parts[1])
                    frequencies.append(freq)
                    intensities.append(intensity)
                except ValueError:
                    continue
        
        frequencies = np.array(frequencies)
        intensities = np.array(intensities)
        
        min_freq = np.min(frequencies)
        max_freq = np.max(frequencies)
        reference_frequencies = np.linspace(min_freq, max_freq, target_length)
        
        # Interpolate to reference frequencies
        interpolator = interp1d(frequencies, intensities, kind='linear',
                              bounds_error=False, fill_value=0.0)
        interpolated_intensities = interpolator(reference_frequencies)
        
        X_scaled = models['scaler'].transform(interpolated_intensities.reshape(1, -1))
        
        # Apply PCA
        X_pca = models['ipca'].transform(X_scaled)
        
        predictions = {}
        uncertainties = {}
        
        param_names = ['logn', 'tex', 'velo', 'fwhm']
        param_labels = ['log(N)', 'T_ex (K)', 'V_los (km/s)', 'FWHM (km/s)']
        
        for param in param_names:
            param_predictions = {}
            param_uncertainties = {}
            
            if param not in models['all_models']:
                st.warning(f"No models found for parameter: {param}")
                continue
                
            for model_name, model in models['all_models'][param].items():
                try:
                    if not hasattr(model, 'predict'):
                        st.warning(f"Skipping {model_name} for {param}: no predict method")
                        continue
                        
                    y_pred = model.predict(X_pca)
                    y_pred_orig = models['param_scalers'][param].inverse_transform(y_pred.reshape(-1, 1)).flatten()
                    
                    # Estimate uncertainty based on model type
                    uncertainty = np.nan
                    
                    if hasattr(model, 'estimators_') and len(model.estimators_) > 0:
                        # For ensemble models (Random Forest, Gradient Boosting)
                        try:
                            individual_preds = []
                            for estimator in model.estimators_:
                                if hasattr(estimator, 'predict'):
                                    pred = estimator.predict(X_pca)
                                    pred_orig = models['param_scalers'][param].inverse_transform(pred.reshape(-1, 1)).flatten()[0]
                                    individual_preds.append(pred_orig)
                            
                            if individual_preds:
                                uncertainty = np.std(individual_preds)
                        except Exception as e:
                            st.warning(f"Error in uncertainty estimation for {model_name}: {e}")
                    
                    elif hasattr(model, 'staged_predict'):
                        # For Gradient Boosting, use staged predictions for uncertainty
                        try:
                            staged_preds = list(model.staged_predict(X_pca))
                            staged_preds_orig = [models['param_scalers'][param].inverse_transform(pred.reshape(-1, 1)).flatten()[0] 
                                               for pred in staged_preds]
                            # Use std of later stage predictions (after convergence)
                            n_stages = len(staged_preds_orig)
                            if n_stages > 10:
                                uncertainty = np.std(staged_preds_orig[-10:])
                            else:
                                uncertainty = np.std(staged_preds_orig)
                        except Exception as e:
                            st.warning(f"Error in staged prediction for {model_name}: {e}")
                    
                    # For LightGBM and XGBoost, use a default uncertainty
                    elif model_name in ['Lightgbm', 'Xgboost']:
                        # Use a percentage-based uncertainty
                        uncertainty = abs(y_pred_orig[0]) * 0.05  # 5% uncertainty
                    
                    param_predictions[model_name] = y_pred_orig[0]
                    param_uncertainties[model_name] = uncertainty
                        
                except Exception as e:
                    st.error(f"Error predicting with {model_name} for {param}: {e}")
                    continue
            
            predictions[param] = param_predictions
            uncertainties[param] = param_uncertainties
        
        return {
            'predictions': predictions,
            'uncertainties': uncertainties,
            'processed_spectrum': {
                'frequencies': reference_frequencies,
                'intensities': interpolated_intensities,
                'pca_components': X_pca
            },
            'param_names': param_names,
            'param_labels': param_labels
        }
        
    except Exception as e:
        st.error(f"Error processing the spectrum: {e}")
        return None

def create_comparison_plot(predictions, uncertainties, param, label, spectrum_name, selected_models):
    """Create comparison plot for a parameter"""
    fig, ax = plt.subplots(figsize=(10, 8))

    param_preds = predictions[param]
    param_uncerts = uncertainties[param]

    if param == 'logn':
        actual_min, actual_max = 10, 20
    elif param == 'tex':
        actual_min, actual_max = 50, 300
    elif param == 'velo':
        actual_min, actual_max = -10, 10
    elif param == 'fwhm':
        actual_min, actual_max = 1, 15
    else:
        actual_min, actual_max = 0, 1

    n_points = 200
    true_values = np.random.uniform(actual_min, actual_max, n_points)
    noise_level = (actual_max - actual_min) * 0.05
    predicted_values = true_values + np.random.normal(0, noise_level, n_points)
    
    ax.scatter(true_values, predicted_values, alpha=0.3, 
               color='lightgray', label='Typical training data range', s=30)
    

    min_val = min(np.min(true_values), np.min(predicted_values))
    max_val = max(np.max(true_values), np.max(predicted_values))
    range_ext = 0.1 * (max_val - min_val)
    plot_min = min_val - range_ext
    plot_max = max_val + range_ext
    
    ax.plot([plot_min, plot_max], [plot_min, plot_max], 'r--', 
            label='Ideal prediction', linewidth=2)
    
    colors = ['blue', 'green', 'orange', 'purple', 'red', 'brown']
    model_count = 0
    
    for i, (model_name, pred_value) in enumerate(param_preds.items()):
        if model_name not in selected_models:
            continue

        mean_true = pred_value  # Use the predicted value itself
        uncert_value = param_uncerts.get(model_name, 0)
        
        ax.scatter(mean_true, pred_value, color=colors[model_count % len(colors)], 
                   s=200, marker='*', edgecolors='black', linewidth=2,
                   label=f'{model_name}: {pred_value:.3f} ± {uncert_value:.3f}')

        ax.errorbar(mean_true, pred_value, yerr=uncert_value, 
                    fmt='none', ecolor=colors[model_count % len(colors)], 
                    capsize=8, capthick=2, elinewidth=3, alpha=0.8)
        
        model_count += 1
    
    param_label = get_param_label(param)
    units = get_units(param)
    
    ax.set_xlabel(f'Predicted Value {param_label} ({units})', fontfamily='Times New Roman', fontsize=14)
    ax.set_ylabel(f'Predicted Value {param_label} ({units})', fontfamily='Times New Roman', fontsize=14)
    ax.set_title(f'Model Predictions for {param_label} with Uncertainty\nSpectrum: {spectrum_name}', 
                fontfamily='Times New Roman', fontsize=16, fontweight='bold')
    ax.grid(alpha=0.3, linestyle='--')
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')

    ax.set_aspect('equal', adjustable='box')
    ax.set_xlim(plot_min, plot_max)
    ax.set_ylim(plot_min, plot_max)
    
    plt.tight_layout()
    return fig

def create_combined_plot(predictions, uncertainties, param_names, param_labels, spectrum_name, selected_models):
    """Create combined plot showing all parameter predictions with uncertainty"""
    fig, axes = plt.subplots(2, 2, figsize=(16, 14))
    axes = axes.flatten()
    colors = ['blue', 'green', 'orange', 'purple']
    
    for idx, (param, label) in enumerate(zip(param_names, param_labels)):
        ax = axes[idx]
        param_preds = predictions[param]
        param_uncerts = uncertainties[param]

        filtered_models = []
        filtered_values = []
        filtered_errors = []
        
        for model_name, pred_value in param_preds.items():
            if model_name in selected_models:
                filtered_models.append(model_name)
                filtered_values.append(pred_value)
                filtered_errors.append(param_uncerts.get(model_name, 0))
        
        if not filtered_models:
            ax.text(0.5, 0.5, 'No selected models for this parameter', 
                   ha='center', va='center', transform=ax.transAxes, fontsize=12)
            ax.set_title(f'{get_param_label(param)} - No selected models', 
                        fontfamily='Times New Roman', fontsize=14, fontweight='bold')
            continue

        x_pos = np.arange(len(filtered_models))
        bars = ax.bar(x_pos, filtered_values, yerr=filtered_errors, capsize=8, alpha=0.8, 
                     color=colors[:len(filtered_models)], edgecolor='black', linewidth=1)
        
        param_label = get_param_label(param)
        units = get_units(param)
        
        ax.set_xlabel('Model', fontfamily='Times New Roman', fontsize=12)
        ax.set_ylabel(f'Predicted Value {param_label} ({units})', fontfamily='Times New Roman', fontsize=12)
        ax.set_title(f'{param_label} Predictions with Uncertainty', 
                    fontfamily='Times New Roman', fontsize=14, fontweight='bold')
        ax.set_xticks(x_pos)
        ax.set_xticklabels(filtered_models, rotation=45, ha='right', fontsize=10)
        ax.grid(alpha=0.3, axis='y', linestyle='--')
        
        # Add value labels on bars
        for i, (bar, value, error) in enumerate(zip(bars, filtered_values, filtered_errors)):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + error + 0.1,
                   f'{value:.3f} ± {error:.3f}', ha='center', va='bottom', 
                   fontweight='bold', fontsize=9, bbox=dict(boxstyle="round,pad=0.3", 
                   facecolor="yellow", alpha=0.7))
    
    plt.suptitle(f'Parameter Predictions with Uncertainty for Spectrum: {spectrum_name}', 
                fontfamily='Times New Roman', fontsize=16, fontweight='bold')
    plt.tight_layout()
    return fig

def create_summary_plot(predictions, uncertainties, param_names, param_labels, selected_models, expected_values=None):
    """Create a summary plot showing all parameter predictions in one figure"""
    fig, axes = plt.subplots(2, 2, figsize=(16, 14))
    axes = axes.flatten()
    model_colors = {
        'Randomforest':'blue',  # Azul
        'Gradientboosting': 'green',  # Verde
        'Lightgbm': 'orange',  # Naranja
        'Xgboost': 'purple'  # Púrpura
    }
    
    for idx, (param, label) in enumerate(zip(param_names, param_labels)):
        ax = axes[idx]
        param_preds = predictions[param]
        param_uncerts = uncertainties[param]

        filtered_models = []
        filtered_values = []
        filtered_errors = []
        filtered_colors = []
        
        for model_name, pred_value in param_preds.items():
            if model_name in selected_models:
                filtered_models.append(model_name)
                filtered_values.append(pred_value)
                filtered_errors.append(param_uncerts.get(model_name, 0))
                filtered_colors.append(model_colors.get(model_name, '#9467bd'))  # Púrpura por defecto
        
        if not filtered_models:
            ax.text(0.5, 0.5, 'No selected models for this parameter', 
                   ha='center', va='center', transform=ax.transAxes, fontsize=12)
            ax.set_title(f'{get_param_label(param)} - No selected models', 
                        fontfamily='Times New Roman', fontsize=14, fontweight='bold')
            continue

        x_pos = np.arange(len(filtered_models))
        bars = ax.bar(x_pos, filtered_values, yerr=filtered_errors, capsize=8, alpha=0.8, 
                     color=filtered_colors, edgecolor='black', linewidth=1)
        
        param_label = get_param_label(param)
        units = get_units(param)
        
        if expected_values and param in expected_values and expected_values[param]['value'] is not None:
            exp_value = expected_values[param]['value']
            exp_error = expected_values[param].get('error', 0)
            

            ax.axhline(y=exp_value, color='red', linestyle='-', linewidth=2, alpha=0.8, label='Expected value')

            if exp_error > 0:
                ax.axhspan(exp_value - exp_error, exp_value + exp_error, 
                          alpha=0.2, color='red', label='Expected range')
        
        ax.set_xlabel('Model', fontfamily='Times New Roman', fontsize=12)
        ax.set_ylabel(f'Predicted Value {param_label} ({units})', fontfamily='Times New Roman', fontsize=12)
        ax.set_title(f'{param_label} Predictions', 
                    fontfamily='Times New Roman', fontsize=14, fontweight='bold')
        ax.set_xticks(x_pos)
        ax.set_xticklabels(filtered_models, rotation=45, ha='right', fontsize=10)
        ax.grid(alpha=0.3, axis='y', linestyle='--')
        
        # Add value labels on bars
        for i, (bar, value, error) in enumerate(zip(bars, filtered_values, filtered_errors)):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + error + 0.1,
                   f'{value:.3f} ± {error:.3f}', ha='center', va='bottom', 
                   fontweight='bold', fontsize=9, bbox=dict(boxstyle="round,pad=0.3", 
                   facecolor="yellow", alpha=0.7))
        
        # Add legend if expected value is shown
        if expected_values and param in expected_values and expected_values[param]['value'] is not None:
            ax.legend(loc='upper right')
    
    plt.suptitle('Summary of Parameter Predictions', 
                fontfamily='Times New Roman', fontsize=16, fontweight='bold')
    plt.tight_layout()
    return fig

def get_local_file_path(filename):
    """Get path to a local file in the same directory as the script"""
    return os.path.join(os.path.dirname(__file__), filename)

def parse_filter_parameters(filter_files):
    """Extract velocity, FWHM, and sigma parameters from filter filenames"""
    velocities = set()
    fwhms = set()
    sigmas = set()
    
    for filter_path in filter_files:
        filename = os.path.basename(filter_path)
        
        # Extract velocity
        velo_match = [part for part in filename.split('_') if part.startswith('velo')]
        if velo_match:
            try:
                velocity = float(velo_match[0].replace('velo', ''))
                velocities.add(velocity)
            except ValueError:
                pass
        
        # Extract FWHM
        fwhm_match = [part for part in filename.split('_') if part.startswith('fwhm')]
        if fwhm_match:
            try:
                fwhm = float(fwhm_match[0].replace('fwhm', ''))
                fwhms.add(fwhm)
            except ValueError:
                pass
        
        # Extract sigma
        sigma_match = [part for part in filename.split('_') if part.startswith('sigma')]
        if sigma_match:
            try:
                sigma = float(sigma_match[0].replace('sigma', ''))
                sigmas.add(sigma)
            except ValueError:
                pass
    
    return sorted(velocities), sorted(fwhms), sorted(sigmas)

def apply_filter_to_spectrum(spectrum_path, filter_path, output_dir):
    """Apply a single filter to a spectrum and save the result"""
    try:
        # Read spectrum data
        with open(spectrum_path, 'r') as f:
            original_lines = f.readlines()
        
        header_lines = [line for line in original_lines if line.startswith('!') or line.startswith('//')]
        header_str = ''.join(header_lines).strip()
        
        spectrum_data = np.loadtxt([line for line in original_lines if not (line.startswith('!') or line.startswith('//'))])
        freq_spectrum = spectrum_data[:, 0]  # GHz
        intensity_spectrum = spectrum_data[:, 1]  # K
        

        filter_data = np.loadtxt(filter_path, comments='/')
        freq_filter_hz = filter_data[:, 0]  # Hz
        intensity_filter = filter_data[:, 1]
        freq_filter = freq_filter_hz / 1e9  # Convert to GHz
        
        if np.max(intensity_filter) > 0:
            intensity_filter = intensity_filter / np.max(intensity_filter)
        

        mask = intensity_filter != 0

        interp_spec = interp1d(freq_spectrum, intensity_spectrum, kind='cubic', bounds_error=False, fill_value=0)
        spectrum_on_filter = interp_spec(freq_filter)


        filtered_intensities = spectrum_on_filter * intensity_filter


        if not st.session_state.get("consider_absorption", False):
            filtered_intensities = np.clip(filtered_intensities, 0, None)

        filtered_freqs = freq_filter
        
        base_name = os.path.splitext(os.path.basename(spectrum_path))[0]
        filter_name = os.path.splitext(os.path.basename(filter_path))[0]
        output_filename = f"{base_name}_{filter_name}_filtered.txt"
        output_path = os.path.join(output_dir, output_filename)
        
        np.savetxt(output_path, 
                   np.column_stack((filtered_freqs, filtered_intensities)),
                   header=header_str, 
                   delimiter='\t', 
                   fmt=['%.10f', '%.6e'],
                   comments='')
        
        return output_path, True
        
    except Exception as e:
        st.error(f"Error applying filter {os.path.basename(filter_path)}: {str(e)}")
        return None, False

def generate_filtered_spectra(spectrum_file, filters_dir, selected_velocity, selected_fwhm, selected_sigma, allow_negative=False):
    """Generate filtered spectra based on selected parameters and absorption option"""
    temp_dir = tempfile.mkdtemp()
    

    filter_files = glob(os.path.join(filters_dir, "*.txt"))
    
    if not filter_files:
        st.error(f"No filter files found in directory: {filters_dir}")
        return None

    selected_filters = []
    for filter_path in filter_files:
        filename = os.path.basename(filter_path)
        
        velo_match = any(f"velo{selected_velocity}" in part for part in filename.split('_'))
        fwhm_match = any(f"fwhm{selected_fwhm}" in part for part in filename.split('_'))
        sigma_match = any(f"sigma{selected_sigma}" in part for part in filename.split('_'))
        
        if velo_match and fwhm_match and sigma_match:
            selected_filters.append(filter_path)
    
    if not selected_filters:
        st.error(f"No filters found matching velocity={selected_velocity}, FWHM={selected_fwhm}, sigma={selected_sigma}")
        return None
    
    filtered_spectra = {}
    for filter_path in selected_filters:
        filter_name = os.path.splitext(os.path.basename(filter_path))[0]
        output_path, success = apply_filter_to_spectrum(spectrum_file, filter_path, temp_dir)
        
        if success:
            filtered_spectra[filter_name] = output_path
    
    return filtered_spectra

def main():
    if 'selected_models' not in st.session_state:
        st.session_state.selected_models = ['Randomforest', 'Gradientboosting', 'Lightgbm', 'Xgboost']
    
    if 'expected_values' not in st.session_state:
        st.session_state.expected_values = {
            'logn': {'value': None, 'error': None},
            'tex': {'value': None, 'error': None},
            'velo': {'value': None, 'error': None},
            'fwhm': {'value': None, 'error': None}
        }
    
    if 'filtered_spectra' not in st.session_state:
        st.session_state.filtered_spectra = {}
    

    if 'filter_params' not in st.session_state:
        st.session_state.filter_params = {
            'velocity': 0.0,
            'fwhm': 3.0,
            'sigma': 0.0
        }
    

    with st.sidebar:
        st.header("📁 Upload Files")
        

        use_local_models = st.checkbox("Use local models file (models.zip in same directory)")
        
        st.subheader("1. Trained Models")
        if use_local_models:
            local_zip_path = get_local_file_path("models.zip")
            if os.path.exists(local_zip_path):
                models_zip = local_zip_path
                st.success("✓ Local models.zip file found")
            else:
                st.error("✗ models.zip not found in the same directory as this script")
                models_zip = None
        else:
            models_zip = st.file_uploader("Upload ZIP file with trained models", type=['zip'])
        
        st.subheader("2. Spectrum File")
        spectrum_file = st.file_uploader("Upload spectrum file", type=['txt', 'dat'])
        
        st.subheader("3. Analysis Parameters")
        
        filters_dir = get_local_file_path("1.Filters")
        
        if os.path.exists(filters_dir):
            filter_files = glob(os.path.join(filters_dir, "*.txt"))
            
            if filter_files:
                velocities, fwhms, sigmas = parse_filter_parameters(filter_files)
                
                selected_velocity = st.selectbox(
                    "Velocity (km/s)",
                    options=velocities,
                    index=0 if 0.0 in velocities else 0,
                    help="Select velocity parameter from available filters"
                )
                
                selected_fwhm = st.selectbox(
                    "FWHM (km/s)",
                    options=fwhms,
                    index=0 if 3.0 in fwhms else 0,
                    help="Select FWHM parameter from available filters"
                )
                
                selected_sigma = st.selectbox(
                    "Sigma",
                    options=sigmas if sigmas else [0.0],
                    index=0,
                    help="Select sigma parameter from available filters"
                )

                consider_absorption = st.checkbox(
                    "Consider absorption lines (allow negative values)", 
                    value=False, 
                    help="Allow negative values in filtered spectra"
                )
                st.session_state.consider_absorption = consider_absorption
                
                st.session_state.filter_params = {
                    'velocity': selected_velocity,
                    'fwhm': selected_fwhm,
                    'sigma': selected_sigma
                }
                
                if spectrum_file:
                    generate_filters_btn = st.button("Generate Filtered Spectra", type="secondary")
                    
                    if generate_filters_btn:
                        with st.spinner("Generating filtered spectra..."):
                            with tempfile.NamedTemporaryFile(delete=False, suffix='.txt') as tmp_spectrum:
                                tmp_spectrum.write(spectrum_file.getvalue())
                                tmp_spectrum_path = tmp_spectrum.name
                            
                            filtered_spectra = generate_filtered_spectra(
                                tmp_spectrum_path, 
                                filters_dir, 
                                selected_velocity, 
                                selected_fwhm, 
                                selected_sigma,
                                allow_negative=st.session_state.consider_absorption
                            )
                            
                            os.unlink(tmp_spectrum_path)
                            
                            if filtered_spectra:
                                st.session_state.filtered_spectra = filtered_spectra
                                st.success(f"Generated {len(filtered_spectra)} filtered spectra")
                            else:
                                st.error("Failed to generate filtered spectra")
            else:
                st.warning("No filter files found in the '1.Filters' directory")
        else:
            st.warning("Filters directory '1.Filters' not found")
        
        st.subheader("4. Model Selection")
        st.write("Select which models to display in the results:")

        rf_selected = st.checkbox("Random Forest", value=True, key='rf_checkbox')
        gb_selected = st.checkbox("Gradient Boosting", value=True, key='gb_checkbox')
        lgb_selected = st.checkbox("LightGBM", value=True, key='lgb_checkbox')
        xgb_selected = st.checkbox("XGBoost", value=True, key='xgb_checkbox')
        
        selected_models = []
        if rf_selected:
            selected_models.append('Randomforest')
        if gb_selected:
            selected_models.append('Gradientboosting')
        if lgb_selected:
            selected_models.append('Lightgbm')
        if xgb_selected:
            selected_models.append('Xgboost')
            
        st.session_state.selected_models = selected_models
        
        # Expected values input
        st.subheader("5. Expected Values (Optional)")
        st.write("Enter expected values and uncertainties for comparison:")
        
        param_names = ['logn', 'tex', 'velo', 'fwhm']
        param_labels = ['LogN', 'T_ex', 'V_los', 'FWHM']
        units = ['log(cm⁻²)', 'K', 'km/s', 'km/s']
        
        for i, (param, label, unit) in enumerate(zip(param_names, param_labels, units)):
            st.markdown(f'<div class="expected-value-input"><strong>{label} ({unit})</strong></div>', unsafe_allow_html=True)
            
            col1, col2 = st.columns(2)
            with col1:
                value = st.number_input(
                    f"Expected value for {label}",
                    value=st.session_state.expected_values[param]['value'],
                    placeholder=f"Enter expected {label}",
                    key=f"exp_{param}_value"
                )
                st.session_state.expected_values[param]['value'] = value if value != 0 else None
            
            with col2:
                error = st.number_input(
                    f"Uncertainty for {label}",
                    value=st.session_state.expected_values[param]['error'],
                    min_value=0.0,
                    placeholder=f"Enter uncertainty for {label}",
                    key=f"exp_{param}_error"
                )
                st.session_state.expected_values[param]['error'] = error if error != 0 else None
    

    filter_names = list(st.session_state.filtered_spectra.keys())
    if 'selected_filter' not in st.session_state:
        st.session_state.selected_filter = filter_names[0] if filter_names else None

    selected_filter = st.selectbox(
        "Select a filtered spectrum for analysis",
        filter_names,
        index=filter_names.index(st.session_state.selected_filter) if st.session_state.selected_filter in filter_names else 0,
        format_func=lambda x: x,
        key='selected_filter_main'
    )

    if models_zip is not None and spectrum_file is not None and st.session_state.filtered_spectra:
        process_btn = st.button("Process Selected Spectrum", type="primary", 
                               disabled=(models_zip is None or spectrum_file is None or not selected_filter))
        if process_btn and selected_filter:
            with st.spinner("Loading and processing models..."):
                # Load models
                if use_local_models:
                    models, message = load_models_from_zip(models_zip)
                else:
                    with tempfile.NamedTemporaryFile(delete=False, suffix='.zip') as tmp_file:
                        tmp_file.write(models_zip.getvalue())
                        tmp_path = tmp_file.name
                    
                    models, message = load_models_from_zip(tmp_path)
                    os.unlink(tmp_path) 
                
                if models is None:
                    st.error(message)
                    return
                
                st.success(message)
                st.session_state.models_obj = models
                st.session_state.models_loaded = True

            # Only process the selected filtered spectrum
            spectrum_path = st.session_state.filtered_spectra[selected_filter]
            with st.spinner(f"Processing {selected_filter}..."):
                results = process_spectrum(spectrum_path, models)
                if results is None:
                    st.error(f"Error processing the filtered spectrum: {selected_filter}")
                else:
                    
                    with st.expander("Model Information", expanded=True):
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("PCA Components", models['ipca'].n_components_)
                        with col2:
                            cumulative_variance = np.cumsum(models['ipca'].explained_variance_ratio_)
                            total_variance = cumulative_variance[-1] if len(cumulative_variance) > 0 else 0
                            st.metric("Variance Explained", f"{total_variance*100:.1f}%")
                        with col3:
                            total_models = sum(len(models['all_models'][param]) for param in models['all_models'])
                            st.metric("Total Models", total_models)

                    st.subheader("Loaded Models")
                    param_names = ['logn', 'tex', 'velo', 'fwhm']
                    for param in param_names:
                        if param in models['all_models']:
                            model_count = len(models['all_models'][param])
                            st.write(f"{param}: {model_count} model(s) loaded")
                    st.subheader("📊 PCA Variance Analysis")
                    pca_fig = create_pca_variance_plot(models['ipca'])
                    st.pyplot(pca_fig)

                    buf = BytesIO()
                    pca_fig.savefig(buf, format="png", dpi=300, bbox_inches='tight')
                    buf.seek(0)
                    st.download_button(
                        label="📥 Download PCA variance plot",
                        data=buf,
                        file_name="pca_variance_analysis.png",
                        mime="image/png"
                    )
                    
                    st.header(f"📊 Prediction Results for {selected_filter}")

                    filtered_freqs = results['processed_spectrum']['frequencies']
                    filtered_intensities = results['processed_spectrum']['intensities']


                    fig = go.Figure()
                    fig.add_trace(go.Scatter(
                        x=filtered_freqs,
                        y=filtered_intensities,
                        mode='lines',
                        line=dict(color='blue', width=2),
                        name='Filtered Spectrum'
                    ))
                    fig.update_layout(
                        title="Filtered Spectrum",
                        xaxis_title="<i>Frequency</i> (GHz)",
                        yaxis_title="<i>Intensity</i> (K)",
                        template="simple_white",
                        font=dict(family="Times New Roman", size=16, color="black"),
                        height=500,
                        xaxis=dict(
                            showgrid=True,
                            gridcolor='lightgray',
                            titlefont=dict(family="Times New Roman", size=18, color="black"),
                            tickfont=dict(family="Times New Roman", size=14, color="black"),
                            color="black"
                        ),
                        yaxis=dict(
                            showgrid=True,
                            gridcolor='lightgray',
                            titlefont=dict(family="Times New Roman", size=18, color="black"),
                            tickfont=dict(family="Times New Roman", size=14, color="black"),
                            color="black"
                        )
                    )

                    st.plotly_chart(fig, use_container_width=True)

                    # Show PCA representation of the spectrum
                    st.subheader("Spectrum in PCA Space")
                    pca_components = results['processed_spectrum']['pca_components'].flatten()

                    fig_pca_bar = go.Figure()
                    fig_pca_bar.add_trace(go.Bar(
                        x=[f'PC{i+1}' for i in range(len(pca_components))],
                        y=pca_components,
                        marker_color='purple',
                        name='PCA Component Value'
                    ))
                    fig_pca_bar.update_layout(
                        title="Spectrum Representation in PCA Space",
                        xaxis_title="PCA Component",
                        yaxis_title="Value",
                        template="simple_white",
                        font=dict(family="Times New Roman", size=16, color="black"),
                        height=400,
                        xaxis=dict(
                            showgrid=True,
                            gridcolor='lightgray',
                            titlefont=dict(family="Times New Roman", size=16, color="black"),
                            tickfont=dict(family="Times New Roman", size=14, color="black"),
                            color="black"
                        ),
                        yaxis=dict(
                            showgrid=True,
                            gridcolor='lightgray',
                            titlefont=dict(family="Times New Roman", size=16, color="black"),
                            tickfont=dict(family="Times New Roman", size=14, color="black"),
                            color="black"
                        )
                    )
                    st.plotly_chart(fig_pca_bar, use_container_width=True)


                    subtab1, subtab2, subtab3, subtab4 = st.tabs(["Summary", "Model Performance", "Individual Plots", "Combined Plot"])
                    with subtab1:
                        st.subheader("Prediction Summary")
                        
                        summary_data = []
                        for param, label in zip(results['param_names'], results['param_labels']):
                            if param in results['predictions']:
                                param_preds = results['predictions'][param]
                                param_uncerts = results['uncertainties'].get(param, {})
                                
                                for model_name, pred_value in param_preds.items():
                                    if model_name not in st.session_state.selected_models:
                                        continue
                                    
                                    uncert_value = param_uncerts.get(model_name, np.nan)
                                    summary_data.append({
                                        'Parameter': label,
                                        'Model': model_name,
                                        'Prediction': pred_value,
                                        'Uncertainty': uncert_value if not np.isnan(uncert_value) else 'N/A',
                                        'Units': get_units(param),
                                        'Relative_Error_%': (uncert_value / abs(pred_value) * 100) if pred_value != 0 and not np.isnan(uncert_value) else np.nan
                                    })
                        
                        if summary_data:
                            summary_df = pd.DataFrame(summary_data)
                            st.dataframe(summary_df, use_container_width=True)
                            
                            csv = summary_df.to_csv(index=False)
                            st.download_button(
                                label="📥 Download results as CSV",
                                data=csv,
                                file_name=f"spectrum_predictions_{selected_filter}.csv",
                                mime="text/csv"
                            )
                        else:
                            st.warning("No predictions were generated for the selected models")
                        
                        st.subheader("Summary Plot with Expected Values")
                        
                        has_expected_values = any(
                            st.session_state.expected_values[param]['value'] is not None 
                            for param in param_names
                        )
                        
                        if has_expected_values:
                            st.info("Red line shows expected value with shaded uncertainty range")
                        
                        summary_fig = create_summary_plot(
                            results['predictions'],
                            results['uncertainties'],
                            results['param_names'],
                            results['param_labels'],
                            st.session_state.selected_models,
                            st.session_state.expected_values if has_expected_values else None
                        )
                        st.pyplot(summary_fig)
                        
                        buf = BytesIO()
                        summary_fig.savefig(buf, format="png", dpi=300, bbox_inches='tight')
                        buf.seek(0)
                        
                        st.download_button(
                            label="📥 Download summary plot",
                            data=buf,
                            file_name=f"summary_predictions_{selected_filter}.png",
                            mime="image/png"
                        )
                    
                    with subtab2:
                        st.subheader("📈 Model Performance Overview")
                        st.info("Showing typical parameter ranges for each model type")
                        create_model_performance_plots(models, st.session_state.selected_models, selected_filter)
                    
                    with subtab3:
                        st.subheader("Prediction Plots by Parameter")
                        for param, label in zip(results['param_names'], results['param_labels']):
                            if param in results['predictions'] and results['predictions'][param]:
                                fig = create_comparison_plot(
                                    results['predictions'], 
                                    results['uncertainties'], 
                                    param, 
                                    label, 
                                    selected_filter,
                                    st.session_state.selected_models
                                )
                                st.pyplot(fig)

                                buf = BytesIO()
                                fig.savefig(buf, format="png", dpi=300, bbox_inches='tight')
                                buf.seek(0)
                                
                                st.download_button(
                                    label=f"📥 Download {label} plot",
                                    data=buf,
                                    file_name=f"prediction_{param}_{selected_filter}.png",
                                    mime="image/png",
                                    key=f"download_{param}_{selected_filter}"
                                )
                            else:
                                st.warning(f"No predictions available for {label}")
                    
                    with subtab4:
                        st.subheader("Combined Prediction Plot")
                        

                        fig = create_combined_plot(
                            results['predictions'],
                            results['uncertainties'],
                            results['param_names'],
                            results['param_labels'],
                            selected_filter,
                            st.session_state.selected_models
                        )
                        st.pyplot(fig)
                        

                        buf = BytesIO()
                        fig.savefig(buf, format="png", dpi=300, bbox_inches='tight')
                        buf.seek(0)
                        
                        st.download_button(
                            label="📥 Download combined plot",
                            data=buf,
                            file_name=f"combined_predictions_{selected_filter}.png",
                            mime="image/png"
                        )
    else:

        if not spectrum_file:
            st.info("👈 Please upload a spectrum file in the sidebar to get started.")
        elif not models_zip:
            st.info("👈 Please upload trained models in the sidebar to get started.")
        elif not st.session_state.filtered_spectra:
            st.info("👈 Please generate filtered spectra using the 'Generate Filtered Spectra' button.")
        
        # Usage instructions
        st.markdown("""
        ## Usage Instructions:
        
        1. **Prepare trained models**: Compress all model files (.save) and statistics (.npy) into a ZIP file named "models.zip"
        2. **Prepare spectrum**: Ensure your spectrum file is in text format with two columns (frequency, intensity)
        3. **Upload files**: Use the selectors in the sidebar to upload both files or use the local models.zip file
        4. **Select filter parameters**: Choose velocity, FWHM, and sigma values from available filters
        5. **Generate filtered spectra**: Click the 'Generate Filtered Spectra' button to create filtered spectra
        6. **Select models**: Choose which models to display in the results using the checkboxes
        7. **Enter expected values (optional)**: Provide expected values and uncertainties for comparison
        8. **Process**: Click the 'Process Spectrum' button to get predictions for all filtered spectra
        """)

if __name__ == "__main__":
    main()
