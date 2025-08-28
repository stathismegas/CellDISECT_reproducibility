import scanpy as sc
import scipy
import pandas as pd
import numpy as np
from celldisect import CellDISECT
import gc
import torch
import random
from scipy.stats import pearsonr, spearmanr
from scipy.stats import wasserstein_distance
from scipy.spatial.distance import cdist
from sklearn.metrics.pairwise import rbf_kernel
from typing import Dict, List, Tuple, Optional, Callable, Any
from dataclasses import dataclass
from pathlib import Path
import warnings
from tqdm import tqdm
import os



@dataclass
class BenchmarkConfig:
    """Configuration for benchmark parameters and paths"""
    # Data paths
    data_path: str = '/lustre/scratch126/cellgen/lotfollahi/aa34/celldisect/datasets/eraslan_preprocessed1212_split_deg.h5ad'
    models_path: str = '/lustre/scratch126/cellgen/lotfollahi/aa34/mamad-works/models_celldisect/celldisect_split_4'
    results_path: str = '/lustre/scratch126/cellgen/lotfollahi/aa34/mamad-works/models_celldisect/cf_benchmarks'
    
    # Experimental parameters
    cats: List[str] = None
    cov_names: List[str] = None
    cov_values: List[str] = None
    cov_values_cf: List[str] = None
    cell_type_key: str = 'Broad cell type'
    sex_key: str = 'sex'
    cell_type_of_interest = ['Immune (DC/macrophage)', 'Immune (alveolar macrophage)']
    sex_of_interst: str = 'male'
    split_name: str = 'split_4'
    cell_type_included: bool = False
    n_samples_from_source_max: int = 500
    random_seed: int = 42
    
    # DEG analysis parameters
    deg_subsets: List[int] = None  # e.g., [20, None] for top 20 and all genes
    
    def __post_init__(self):
        # Set defaults for mutable fields
        if self.cats is None:
            self.cats = ['tissue', 'Sample ID', 'Age_bin']
        if self.cov_names is None:
            self.cov_names = ['tissue']
        if self.cov_values is None:
            self.cov_values = ['anterior wall of left ventricle']
        if self.cov_values_cf is None:
            self.cov_values_cf = ['lingula of left lung']
        if self.deg_subsets is None:
            self.deg_subsets = [20, None]



class MetricsCalculator:
    """Unified metrics calculation system"""
    
    def __init__(self):
        self.metrics = {
            'emd': self._calculate_emd,
            'mmd': self._calculate_mmd,
            'e_distance': self._calculate_e_distance,
            'pearson': self._calculate_pearson,
            'pearson_var': self._calculate_pearson_var,
            'spearman': self._calculate_spearman,
            'spearman_var': self._calculate_spearman_var,
            'delta_pearson': self._calculate_delta_pearson,
            'delta_pearson_var': self._calculate_delta_pearson_var,
            'delta_spearman': self._calculate_delta_spearman,
            'delta_spearman_var': self._calculate_delta_spearman_var,
        }
    
    def add_metric(self, name: str, func: Callable):
        """Add a new metric function"""
        self.metrics[name] = func
    
    def calculate_all_metrics(self, x_true: np.ndarray, x_ctrl: np.ndarray, 
                            model_predictions: Dict[str, np.ndarray], 
                            deg_indices: np.ndarray) -> Dict[str, Dict[str, float]]:
        """Calculate all metrics for given predictions"""
        # Subset to DEG indices
        data = {
            'x_true': x_true[:, deg_indices],
            'x_ctrl': x_ctrl[:, deg_indices],
        }
        
        # Add model predictions
        for model_name, pred in model_predictions.items():
            data[f'x_{model_name}'] = pred[:, deg_indices]
        
        print(f"Input dimensions: x_true: {x_true.shape}")
        
        results = {}
        for metric_name, metric_func in self.metrics.items():
            print(f"Calculating {metric_name}...")
            results[metric_name] = metric_func(**data)
        
        return results
    
    def _calculate_emd(self, x_true, x_ctrl, **kwargs):
        """Calculate Earth Mover's Distance (Wasserstein distance)"""
        methods = {'Control': x_ctrl}
        
        # Add model predictions
        for key, value in kwargs.items():
            if key.startswith('x_') and key != 'x_true' and key != 'x_ctrl':
                model_name = key[2:]  # Remove 'x_' prefix
                methods[model_name] = value
        
        results = {}
        for method_name, method_data in methods.items():
            wd_values = []
            for i in range(x_true.shape[1]):
                wd = wasserstein_distance(
                    torch.tensor(x_true[:, i]), 
                    torch.tensor(method_data[:, i])
                )
                wd_values.append(wd)
            results[method_name] = np.mean(wd_values)
        
        return results
    
    def _calculate_mmd(self, x_true, x_ctrl, **kwargs):
        """Calculate MMD"""
        results = {'Control': self._calculate_scalar_mmd(x_true, x_ctrl)}
        
        # Add model predictions
        for key, value in kwargs.items():
            if key.startswith('x_') and key != 'x_true' and key != 'x_ctrl':
                model_name = key[2:]  # Remove 'x_' prefix
                results[model_name] = self._calculate_scalar_mmd(x_true, value)
        
        return results
    
    def _calculate_e_distance(self, x_true, x_ctrl, **kwargs):
        """Calculate energy distance"""
        results = {'Control': self._calculate_scalar_e_distance(x_true, x_ctrl)}
        
        # Add model predictions
        for key, value in kwargs.items():
            if key.startswith('x_') and key != 'x_true' and key != 'x_ctrl':
                model_name = key[2:]  # Remove 'x_' prefix
                results[model_name] = self._calculate_scalar_e_distance(x_true, value)
        
        return results

    def _calculate_pearson(self, x_true, x_ctrl, **kwargs):
        """Calculate Pearson correlation for means"""
        results = {'Control': pearsonr(x_true.mean(0), x_ctrl.mean(0))[0]}
        
        # Add model predictions
        for key, value in kwargs.items():
            if key.startswith('x_') and key != 'x_true' and key != 'x_ctrl':
                model_name = key[2:]  # Remove 'x_' prefix
                results[model_name] = pearsonr(x_true.mean(0), value.mean(0))[0]
        
        return results
    
    def _calculate_pearson_var(self, x_true, x_ctrl, **kwargs):
        """Calculate Pearson correlation for variances"""
        results = {'Control_var': pearsonr(x_true.var(0), x_ctrl.var(0))[0]}
        
        # Add model predictions
        for key, value in kwargs.items():
            if key.startswith('x_') and key != 'x_true' and key != 'x_ctrl':
                model_name = key[2:]  # Remove 'x_' prefix
                results[f'{model_name}_var'] = pearsonr(x_true.var(0), value.var(0))[0]
        
        return results
    
    def _calculate_spearman(self, x_true, x_ctrl, **kwargs):
        """Calculate Spearman correlation using pairwise approach"""
        results = {}
        
        # Calculate pairwise correlations for control
        control_stats = self._calculate_pairwise_spearman_stats(x_true, x_ctrl)
        results['Control_min'] = control_stats['min']
        results['Control_mean'] = control_stats['mean'] 
        results['Control_median'] = control_stats['median']
        results['Control_max'] = control_stats['max']
        
        # Calculate pairwise correlations for each model
        for key, value in kwargs.items():
            if key.startswith('x_') and key != 'x_true' and key != 'x_ctrl':
                model_name = key[2:]  # Remove 'x_' prefix
                stats = self._calculate_pairwise_spearman_stats(x_true, value)
                results[f'{model_name}_min'] = stats['min']
                results[f'{model_name}_mean'] = stats['mean'] 
                results[f'{model_name}_median'] = stats['median']
                results[f'{model_name}_max'] = stats['max']
        
        return results
    
    def _calculate_pairwise_spearman_stats(self, x_true, x_pred):
        """Calculate pairwise Spearman correlations between all sample pairs and return statistics"""
        correlations = []
        
        for i in range(x_true.shape[0]):
            for j in range(x_pred.shape[0]):
                # Calculate Spearman correlation between sample i of x_true and sample j of x_pred
                corr, _ = spearmanr(x_true[i, :], x_pred[j, :])
                if not np.isnan(corr):  # Skip NaN correlations
                    correlations.append(corr)
        
        correlations = np.array(correlations)
        
        if len(correlations) == 0:
            # Handle case where all correlations are NaN
            return {'min': np.nan, 'mean': np.nan, 'median': np.nan, 'max': np.nan}
        
        return {
            'min': np.min(correlations),
            'mean': np.mean(correlations),
            'median': np.median(correlations),
            'max': np.max(correlations)
        }
    
    def _calculate_delta_pearson(self, x_true, x_ctrl, **kwargs):
        """Calculate Pearson correlation for delta means (subtract control)"""
        results = {}
        
        # Add model predictions
        for key, value in kwargs.items():
            if key.startswith('x_') and key != 'x_true' and key != 'x_ctrl':
                model_name = key[2:]  # Remove 'x_' prefix
                results[model_name] = pearsonr(x_true.mean(0) - x_ctrl.mean(0), value.mean(0) - x_ctrl.mean(0))[0]
        
        return results
    
    def _calculate_delta_pearson_var(self, x_true, x_ctrl, **kwargs):
        """Calculate Pearson correlation for delta variances (subtract control)"""
        results = {}
        
        # Add model predictions
        for key, value in kwargs.items():
            if key.startswith('x_') and key != 'x_true' and key != 'x_ctrl':
                model_name = key[2:]  # Remove 'x_' prefix
                results[f'{model_name}_var'] = pearsonr(x_true.var(0) - x_ctrl.var(0), value.var(0) - x_ctrl.var(0))[0]
        
        return results
    
    def _calculate_delta_spearman(self, x_true, x_ctrl, **kwargs):
        """Calculate Spearman correlation for delta values using pairwise approach"""
        results = {}
        
        # Subtract control mean from each sample (delta approach)
        x_ctrl_mean = x_ctrl.mean(0)
        x_true_delta = x_true - x_ctrl_mean
        
        # Calculate pairwise correlations for each model
        for key, value in kwargs.items():
            if key.startswith('x_') and key != 'x_true' and key != 'x_ctrl':
                model_name = key[2:]  # Remove 'x_' prefix
                model_delta = value - x_ctrl_mean
                stats = self._calculate_pairwise_spearman_stats(x_true_delta, model_delta)
                results[f'{model_name}_min'] = stats['min']
                results[f'{model_name}_mean'] = stats['mean']
                results[f'{model_name}_median'] = stats['median']
                results[f'{model_name}_max'] = stats['max']
        
        return results
    
    def _calculate_spearman_var(self, x_true, x_ctrl, **kwargs):
        """Calculate Spearman correlation for variances"""
        results = {'Control_var': spearmanr(x_true.var(0), x_ctrl.var(0))[0]}
        
        # Add model predictions
        for key, value in kwargs.items():
            if key.startswith('x_') and key != 'x_true' and key != 'x_ctrl':
                model_name = key[2:]  # Remove 'x_' prefix
                results[f'{model_name}_var'] = spearmanr(x_true.var(0), value.var(0))[0]
        
        return results
    
    def _calculate_delta_spearman_var(self, x_true, x_ctrl, **kwargs):
        """Calculate Spearman correlation for delta variances (subtract control)"""
        results = {}
        
        # Add model predictions
        for key, value in kwargs.items():
            if key.startswith('x_') and key != 'x_true' and key != 'x_ctrl':
                model_name = key[2:]  # Remove 'x_' prefix
                results[f'{model_name}_var'] = spearmanr(x_true.var(0) - x_ctrl.var(0), value.var(0) - x_ctrl.var(0))[0]
        
        return results


    def _calculate_scalar_mmd(self, x, y, gammas=None) -> float:
        """Compute the Mean Maximum Discrepancy (MMD) across different length scales
        Code adapted from CellFlow: https://github.com/theislab/CellFlow/blob/main/src/cellflow/metrics/_metrics.py

        Parameters
        ----------
            x
                An array of shape [num_samples, num_features].
            y
                An array of shape [num_samples, num_features].
            gammas
                A sequence of values for the paramater gamma of the rbf kernel.

        Returns
        -------
            A scalar denoting the average MMD over all gammas.
        """
        if gammas is None:
            gammas = [2, 1, 0.5, 0.1, 0.01, 0.005]
        mmds = [self._maximum_mean_discrepancy(x, y, gamma=gamma) for gamma in gammas]  # type: ignore[union-attr]
        return np.nanmean(np.array(mmds))
    
    def _maximum_mean_discrepancy(self, x, y, gamma: float = 1.0) -> float:
        """Compute the Maximum Mean Discrepancy (MMD) between two distributions x and y.
        Code adapted from CellFlow: https://github.com/theislab/CellFlow/blob/main/src/cellflow/metrics/_metrics.py

        Parameters
        ----------
            x
                An array of shape [num_samples, num_features].
            y
                An array of shape [num_samples, num_features].
            gamma
                Parameter for the rbf kernel.
            exact
                Use exact or fast rbf kernel.

        Returns
        -------
            A scalar denoting the squared maximum mean discrepancy loss.
        """
        kernel = rbf_kernel
        xx = kernel(x, x, gamma)
        xy = kernel(x, y, gamma)
        yy = kernel(y, y, gamma)
        return xx.mean() + yy.mean() - 2 * xy.mean()

    def _calculate_scalar_e_distance(self, x, y) -> float:
        """Compute the energy distance between x and y as in :cite:`Peidli2024`.
        Code adapted from CellFlow: https://github.com/theislab/CellFlow/blob/main/src/cellflow/metrics/_metrics.py

        Parameters
        ----------
            x
                An array of shape [num_samples, num_features].
            y
                An array of shape [num_samples, num_features].

        Returns
        -------
            A scalar denoting the energy distance value.
        """
        sigma_X = self._pairwise_squeuclidean(x, x).mean()
        sigma_Y = self._pairwise_squeuclidean(y, y).mean()
        delta = self._pairwise_squeuclidean(x, y).mean()
        return 2 * delta - sigma_X - sigma_Y


    def _pairwise_squeuclidean(self, x, y):
        return cdist(x, y, metric='sqeuclidean')



class ModelPredictor:
    """Handles model loading and predictions for all methods"""
    
    def __init__(self, config: BenchmarkConfig):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    def get_model_folders(self):
        """Get list of model folders to compare"""
        model_folders = [
            "pretrainAE_0_maxEpochs_1000_split_split_4_reconW_20_cfWeight_0.8_beta_0.003_clf_0.05_adv_0.014_advp_5_n_cf_1_lr_0.003_wd_5e-05_new_cf_True_dropout_0.1_n_hidden_128_n_latent_32_n_layers_1batch_size_256_NoCT",
            "pretrainAE_0_maxEpochs_1000_split_split_4_reconW_20_cfWeight_0.8_beta_0.003_clf_0.05_adv_0.014_advp_5_n_cf_1_lr_0.003_wd_5e-05_new_cf_True_dropout_0.1_n_hidden_128_n_latent_32_n_layers_2batch_size_256_NoCT",
            "pretrainAE_0_maxEpochs_1000_split_split_4_reconW_20_cfWeight_0.8_beta_0.003_clf_0.05_adv_0.014_advp_5_n_cf_1_lr_0.003_wd_5e-05_new_cf_True_dropout_0.1_n_hidden_128_n_latent_32_n_layers_4batch_size_256_NoCT",
            "pretrainAE_0_maxEpochs_1000_split_split_4_reconW_20_cfWeight_0.8_beta_0.003_clf_0.05_adv_0.014_advp_5_n_cf_1_lr_0.003_wd_5e-05_new_cf_True_dropout_0.1_n_hidden_128_n_latent_32_n_layers_6batch_size_256_NoCT",
            "pretrainAE_0_maxEpochs_1000_split_split_4_reconW_20_cfWeight_0.8_beta_0.003_clf_0.05_adv_0.014_advp_5_n_cf_1_lr_0.006_wd_5e-05_new_cf_True_dropout_0.1_n_hidden_128_n_latent_32_n_layers_2batch_size_256_NoCT",
            "pretrainAE_0_maxEpochs_1000_split_split_4_reconW_20_cfWeight_0.8_beta_0.003_clf_0.05_adv_0.014_advp_5_n_cf_1_lr_0.01_wd_5e-05_new_cf_True_dropout_0.1_n_hidden_128_n_latent_32_n_layers_2batch_size_256_NoCT",
            "pretrainAE_0_maxEpochs_1000_split_split_4_reconW_20_cfWeight_0.8_beta_0.003_clf_0.05_adv_0.014_advp_5_n_cf_1_lr_0.03_wd_5e-05_new_cf_True_dropout_0.1_n_hidden_128_n_latent_32_n_layers_2batch_size_256_NoCT",
            "pretrainAE_0_maxEpochs_1000_split_split_4_reconW_20_cfWeight_0.8_beta_0.003_clf_0.05_adv_0.014_advp_5_n_cf_1_lr_0.06_wd_5e-05_new_cf_True_dropout_0.1_n_hidden_128_n_latent_32_n_layers_2batch_size_256_NoCT",
            "pretrainAE_0_maxEpochs_1000_split_split_4_reconW_20_cfWeight_0.8_beta_0.003_clf_0.05_adv_0.014_advp_5_n_cf_1_lr_0.1_wd_5e-05_new_cf_True_dropout_0.1_n_hidden_128_n_latent_32_n_layers_2batch_size_256_NoCT"
        ]
        return model_folders
    
    def load_models(self, adata):
        """Load all CellDISECT models"""
        model_folders = self.get_model_folders()
        models = {}
        
        for folder in model_folders:
            model_path = os.path.join(self.config.models_path, folder)
            if os.path.exists(model_path):
                print(f"Loading model from {folder}...")
                try:
                    model = CellDISECT.load(model_path, adata=adata)
                    # Create a shorter name for the model based on key parameters
                    model_name = self._extract_model_name(folder)
                    models[model_name] = model
                except Exception as e:
                    print(f"Failed to load model from {folder}: {e}")
            else:
                print(f"Model path does not exist: {model_path}")
        
        return models
    
    def _extract_model_name(self, folder_name):
        """Extract a meaningful name from the folder name"""
        # First check for different learning rates (these models have n_layers_2 but different lr)
        if "lr_0.006" in folder_name:
            return "CellDISECT_lr0.006"
        elif "lr_0.01" in folder_name:
            return "CellDISECT_lr0.01"
        elif "lr_0.03" in folder_name:
            return "CellDISECT_lr0.03"
        elif "lr_0.06" in folder_name:
            return "CellDISECT_lr0.06"
        elif "lr_0.1" in folder_name:
            return "CellDISECT_lr0.1"
        # Then check for different layer counts
        elif "n_layers_1" in folder_name:
            return "CellDISECT_1layer"
        elif "n_layers_2" in folder_name:
            return "CellDISECT_2layer"
        elif "n_layers_4" in folder_name:
            return "CellDISECT_4layer"
        elif "n_layers_6" in folder_name:
            return "CellDISECT_6layer"
        else:
            return "CellDISECT_default"
    
    def predict_celldisect(self, model, adata, cell_type, sex: str, n_samples: Optional[int] = None):
        """Generate CellDISECT predictions"""
        x_ctrl, x_true, x_pred = model.predict_counterfactuals(
            adata[
                (adata.obs[self.config.cell_type_key].isin(cell_type)) &
                (adata.obs[self.config.sex_key] == sex)
                ].copy(),
            cov_names=self.config.cov_names,
            cov_values=self.config.cov_values,
            cov_values_cf=self.config.cov_values_cf,
            cats=self.config.cats,
            n_samples_from_source=n_samples,
            seed=self.config.random_seed,
        )
        return np.log1p(x_ctrl), np.log1p(x_true), np.log1p(x_pred)


class BenchmarkRunner:
    """Main benchmark runner that orchestrates the entire process"""
    
    def __init__(self, config: BenchmarkConfig):
        self.config = config
        self.metrics_calc = MetricsCalculator()
        self.predictor = ModelPredictor(config)
        
        # Create output directory
        Path(self.config.results_path).mkdir(parents=True, exist_ok=True)
    
    def load_data(self):
        """Load and prepare data"""
        print("Loading data...")
        adata = sc.read_h5ad(self.config.data_path)
        
        # Filter cells with non-zero counts
        adata = adata[adata.layers['counts'].sum(1) != 0].copy()

        # If sparese, convert to dense
        if scipy.sparse.issparse(adata.X):
            adata.X = adata.X.todense()
        if scipy.sparse.issparse(adata.layers['counts']):
            adata.layers['counts'] = adata.layers['counts'].todense()
                
        cell_type_included = (
            self.config.cell_type_included  # Set to True if you have provided a cell type annotation in the cats list
        )
        if not cell_type_included:
            adata.obs["_cluster"] = (
                "0"  # Dummy obs for inference (not-training) time, to avoid computing neighbors and clusters again in setup_anndata | AVOID ADDING BEFORE TRAINING
            )

        gc.collect()
        
        return adata
    
    def get_deg_indices(self, adata, n_top_deg: Optional[int]):
        """Get indices for differentially expressed genes"""
        deg_list = adata.uns[f"rank_genes_groups_{self.config.split_name}"]['Immune (alveolar macrophage)_lingula of left lung']
        
        if n_top_deg is not None:
            return np.where(np.isin(adata.var_names, deg_list[:n_top_deg]))[0]
        else:
            return np.arange(adata.n_vars)
    
    def run_benchmark_for_cell_type(self, cell_type, sex: str, adata):
        """Run complete benchmark for a single cell type"""
        print(f"Processing {cell_type}...")
        gc.collect()
        
        # Calculate number of samples
        n_samples = min(
            self.config.n_samples_from_source_max,
            len(adata[(adata.obs[self.config.cell_type_key].isin(cell_type)) & 
                      (adata.obs[self.config.sex_key] == sex) &
                      (adata.obs[self.config.cov_names[0]] == self.config.cov_values[0])])
        )
        
        # Load models
        print("Loading models...")
        models = self.predictor.load_models(adata)
        
        if not models:
            print("No models loaded successfully. Exiting.")
            return
        
        # Generate predictions for all models
        model_predictions = {}
        x_ctrl = None
        x_true = None
        
        print("Generating predictions for all models...")
        for model_name, model in models.items():
            print(f"Generating predictions for {model_name}...")
            try:
                ctrl, true, pred = self.predictor.predict_celldisect(
                    model, adata, cell_type, sex, n_samples
                )
                model_predictions[model_name] = pred
                
                # Store control and true values from the first model (they should be the same for all)
                if x_ctrl is None:
                    x_ctrl = ctrl
                    x_true = true
                    
            except Exception as e:
                print(f"Failed to generate predictions for {model_name}: {e}")
        
        if not model_predictions:
            print("No predictions generated successfully. Exiting.")
            return

        # Calculate metrics for each DEG subset
        all_results = {}
        for n_top_deg in self.config.deg_subsets:
            subset_name = str(n_top_deg) if n_top_deg is not None else 'all'
            deg_indices = self.get_deg_indices(adata, n_top_deg)
            
            print(f"Calculating metrics for {subset_name} genes...")
            metrics_results = self.metrics_calc.calculate_all_metrics(
                x_true, x_ctrl, model_predictions, deg_indices
            )
            
            all_results[subset_name] = metrics_results
        
        # Save results
        self.save_results(all_results)
        gc.collect()
    
    def save_results(self, results: Dict[str, Dict[str, Dict[str, float]]]):
        """Save results to CSV files"""
        # Group metrics by type for backwards compatibility
        metric_groups = {
            'emd': ['emd'],
            'mmd': ['mmd'],
            'e_distance': ['e_distance'],
            'pearson': ['pearson', 'pearson_var'],
            'delta_pearson': ['delta_pearson', 'delta_pearson_var'],
            'spearman': ['spearman', 'spearman_var'],
            'delta_spearman': ['delta_spearman', 'delta_spearman_var']
        }
        
        for group_name, metric_names in metric_groups.items():
            # Collect data for this group
            group_data = {}
            for subset_name, subset_results in results.items():
                group_data[subset_name] = {}
                for metric_name in metric_names:
                    if metric_name in subset_results:
                        group_data[subset_name].update(subset_results[metric_name])
            
            # Save to CSV
            if group_data:
                df = pd.DataFrame.from_dict(group_data).T
                filename = f"{self.config.results_path}/eraslan_noCT_{self.config.split_name}_{group_name}.csv"
                df.to_csv(filename)
    
    def run_full_benchmark(self):
        """Run the complete benchmark for all cell types"""
        # Load data
        adata = self.load_data()
    
        self.run_benchmark_for_cell_type(
            self.config.cell_type_of_interest, self.config.sex_of_interst, adata
        )
        
        print("Benchmark completed!")


def main():
    """Main entry point"""
    # Initialize configuration (can be customized for different experiments)
    config = BenchmarkConfig()
    
    # Run benchmark
    runner = BenchmarkRunner(config)
    runner.run_full_benchmark()


if __name__ == "__main__":
    main()
