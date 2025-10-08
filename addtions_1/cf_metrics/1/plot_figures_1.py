import os
import pandas as pd
from typing import Dict, List, Optional, Callable
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os
import numpy as np
import gc

def load_metrics_dual_directory(
    csv_path_old: str,
    csv_path_new: str,
    dataset_name: str,
    file_pattern_config: Dict,
    split_name_mapping: Optional[Dict] = None,
    ignore_var_metrics: bool = True,
    filter_conditions: Optional[Dict] = None,
    spearman_stat: str = 'mean'  # NEW: Choose which spearman statistic to use
):
    """
    Generalized function to load metrics from two different directories:
    - Directory 1 (old): Pearson, Delta Pearson, EMD
    - Directory 2 (new): Spearman, Delta Spearman, MMD, Energy Distance
    
    Parameters:
    -----------
    csv_path_old : str
        Path to directory containing old metrics CSV files
    csv_path_new : str
        Path to directory containing new metrics CSV files
    dataset_name : str  
        Name of the dataset ('Kang', 'Haber', 'Eraslan')
    file_pattern_config : dict
        Configuration for parsing file names and extracting identifiers
    split_name_mapping : dict, optional
        Mapping for split names (e.g., {'1': 'ImmuneCT Lung: Female2Male'})
    ignore_var_metrics : bool
        Whether to ignore _var metrics
    filter_conditions : dict, optional
        Additional filtering conditions (e.g., {'split_type': 'allOut'})
    spearman_stat : str
        Which Spearman statistic to use: 'min', 'mean', 'median', or 'max'
        
    Returns:
    --------
    pd.DataFrame
        Combined and melted DataFrame
    """
    
    # Validate spearman_stat parameter
    valid_stats = ['min', 'mean', 'median', 'max']
    if spearman_stat not in valid_stats:
        raise ValueError(f"spearman_stat must be one of {valid_stats}, got {spearman_stat}")
    
    # Define metric mappings and their corresponding directories
    old_metrics = {
        'emd.csv': 'EMD',
        'pearson.csv': 'Pearson',
        'delta_pearson.csv': 'Delta Pearson'
    }
    
    new_metrics = {
        'mmd.csv': 'MMD',
        'e_distance.csv': 'Energy Distance',
        'spearman.csv': 'Spearman', 
        'delta_spearman.csv': 'Delta Spearman'
    }
    
    all_results = []
    
    # Process old metrics from first directory
    print(f"Processing old metrics from: {csv_path_old}")
    all_results.extend(_process_directory(csv_path_old, old_metrics, file_pattern_config, 
                                        split_name_mapping, ignore_var_metrics, 
                                        filter_conditions, dataset_name, is_old_dir=True,
                                        spearman_stat=spearman_stat))
    
    # Process new metrics from second directory
    print(f"Processing new metrics from: {csv_path_new}")
    all_results.extend(_process_directory(csv_path_new, new_metrics, file_pattern_config, 
                                        split_name_mapping, ignore_var_metrics, 
                                        filter_conditions, dataset_name, is_old_dir=False,
                                        spearman_stat=spearman_stat))
    
    if not all_results:
        print("Warning: No results found in either directory")
        return pd.DataFrame()
    
    # Combine all results
    combined_df = pd.concat(all_results, ignore_index=True)
    print(f"Combined {len(all_results)} files into dataframe with shape: {combined_df.shape}")
    
    # Apply categorical ordering if specified
    if 'categorical_order' in file_pattern_config:
        for col, order in file_pattern_config['categorical_order'].items():
            if col in combined_df.columns:
                combined_df[col] = pd.Categorical(combined_df[col], categories=order, ordered=True)
                combined_df = combined_df.sort_values(col)
    
    # Determine value columns and rename if needed
    value_columns = file_pattern_config.get('value_columns', ['CellDISECT_1layer', 'CellDISECT_2layer', 'CellDISECT_4layer', 
                   'CellDISECT_6layer', 'CellDISECT_lr0.006', 'CellDISECT_lr0.01', 
                   'CellDISECT_lr0.03', 'CellDISECT_lr0.06', 'CellDISECT_lr0.1', 'Control'])
    
    # Handle column renaming (e.g., Dis2P -> CellDISECT) - already done in _process_directory
    # Update value_columns after potential renaming in config
    if 'column_mapping' in file_pattern_config:
        value_columns = [file_pattern_config['column_mapping'].get(col, col) for col in value_columns]
    
    # Only include columns that actually exist
    existing_value_cols = [col for col in value_columns if col in combined_df.columns]
    id_vars = file_pattern_config['id_vars'] + ['Genes', 'Metric']
    
    print(f"Value columns found: {existing_value_cols}")
    print(f"ID vars: {id_vars}")
    
    # Melt to long format
    melted_df = combined_df.melt(
        id_vars=id_vars,
        value_vars=existing_value_cols,
        var_name='method',
        value_name='value'
    )
    
    melted_df['dataset'] = dataset_name
    melted_df['spearman_stat_used'] = spearman_stat  # Track which statistic was used
    print(f"Final melted dataframe shape: {melted_df.shape}")
    print(f"Metrics found: {sorted(melted_df['Metric'].unique())}")
    print(f"Using Spearman statistic: {spearman_stat}")
    
    return melted_df

def _process_directory(csv_path, metric_mappings, file_pattern_config, split_name_mapping, 
                      ignore_var_metrics, filter_conditions, dataset_name, is_old_dir=False,
                      spearman_stat='mean'):
    """
    Helper function to process files in a single directory
    
    Parameters:
    -----------
    csv_path : str
        Path to the directory
    metric_mappings : dict
        Dictionary mapping file suffixes to metric names
    is_old_dir : bool
        Whether this is the old directory (for Dis2P -> CellDISECT replacement)
    spearman_stat : str
        Which Spearman statistic to extract ('min', 'mean', 'median', 'max')
    
    Returns:
    --------
    list
        List of processed DataFrames
    """
    results = []
    
    if not os.path.exists(csv_path):
        print(f"Warning: Directory {csv_path} does not exist")
        return results
    
    files_processed = 0
    for file in os.listdir(csv_path):
        # Apply filter conditions if specified
        if filter_conditions:
            skip_file = False
            for condition_key, condition_value in filter_conditions.items():
                if condition_key == 'split_type' and condition_value not in file:
                    skip_file = True
                    break
                elif condition_key == 'contains' and condition_value not in file:
                    skip_file = True
                    break
                elif condition_key == 'not_contains' and condition_value in file:
                    skip_file = True
                    break
            if skip_file:
                continue
        
        # Check if file matches any metric pattern for this directory
        for suffix, metric_name in metric_mappings.items():
            if file.endswith(suffix):
                # Skip var metrics if requested
                if ignore_var_metrics and '_var' in file:
                    continue
                    
                # Skip delta files for regular pearson/spearman
                if suffix in ['pearson.csv', 'spearman.csv'] and 'delta' in file:
                    continue
                
                try:
                    # Extract identifiers using the provided extraction function
                    identifiers = file_pattern_config['extract_identifiers'](file, metric_name)
                    
                    if identifiers is None:
                        continue
                    
                    # Load the CSV file
                    df = pd.read_csv(os.path.join(csv_path, file), index_col=0)
                    
                    # Handle new Spearman format with multiple statistics
                    if metric_name in ['Spearman', 'Delta Spearman']:
                        df = _process_spearman_columns(df, spearman_stat, metric_name)
                    
                    # Handle Dis2P -> CellDISECT replacement (especially important for old directory)
                    if 'Dis2P' in df.columns:
                        df = df.rename(columns={'Dis2P': 'CellDISECT'})
                        print(f"Renamed 'Dis2P' to 'CellDISECT' in {file}")
                    
                    # Apply any additional column mapping from config
                    if 'column_mapping' in file_pattern_config:
                        df = df.rename(columns=file_pattern_config['column_mapping'])
                    
                    # Add metadata columns
                    for key, value in identifiers.items():
                        # Apply split name mapping if provided
                        if key in ['Split', 'Cell Type'] and split_name_mapping and value in split_name_mapping:
                            value = split_name_mapping[value]
                        df[key] = value
                    
                    df['Genes'] = df.index
                    df['Metric'] = metric_name
                    
                    results.append(df)
                    files_processed += 1
                    print(f"Processed {file} -> {metric_name} (shape: {df.shape})")
                    
                except Exception as e:
                    print(f"Error processing {file}: {e}")
                    continue
                    
                break
    
    print(f"Processed {files_processed} files from {csv_path}")
    return results

def _process_spearman_columns(df, spearman_stat, metric_name):
    """
    Process Spearman CSV files that have multiple statistics per method
    Extract the chosen statistic and rename columns back to method names
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame with columns like CellDISECT_min, CellDISECT_mean, etc.
    spearman_stat : str
        Which statistic to extract ('min', 'mean', 'median', 'max')
    metric_name : str
        Name of the metric for logging
        
    Returns:
    --------
    pd.DataFrame
        DataFrame with columns renamed to method names (CellDISECT, Biolord, etc.)
    """
    # Define the methods that should have the new format
    methods = ['CellDISECT_1layer', 'CellDISECT_2layer', 'CellDISECT_4layer', 'CellDISECT_6layer', 'CellDISECT_lr0.006', 'CellDISECT_lr0.01', 'CellDISECT_lr0.03', 'CellDISECT_lr0.06', 'CellDISECT_lr0.1']
    if metric_name == 'Spearman':  # Regular spearman includes Control
        methods.append('Control')
    
    # Create new column mapping
    new_columns = {}
    columns_found = []
    
    for method in methods:
        stat_col = f"{method}_{spearman_stat}"
        if stat_col in df.columns:
            new_columns[stat_col] = method
            columns_found.append(stat_col)
        else:
            print(f"Warning: Expected column {stat_col} not found in {metric_name}")
    
    # Check if we found the expected new format
    if columns_found:
        print(f"Found new Spearman format in {metric_name}, extracting '{spearman_stat}' statistic")
        print(f"Columns found: {columns_found}")
        
        # Select only the desired statistic columns and other metadata columns
        other_cols = [col for col in df.columns if not any(col.startswith(f"{m}_") for m in methods)]
        cols_to_keep = columns_found + other_cols
        
        df = df[cols_to_keep].copy()
        df = df.rename(columns=new_columns)
    else:
        print(f"No new Spearman format detected in {metric_name}, using original columns")
    
    return df

def create_eraslan_config():
    def extract_eraslan_identifiers(filename, metric_name):
        # Different patterns for different metrics
        if metric_name == 'Delta Pearson' or metric_name == 'Delta Spearman' or metric_name == 'Energy Distance':
            # Format: ..._split_delta_pearson.csv
            parts = filename.split('_')
            split_idx = -3  # Third from end
        else:
            # Format: ..._split_metric.csv  
            parts = filename.split('_')
            split_idx = -2  # Second from end
            
        if len(parts) > abs(split_idx):
            return {'Split': parts[split_idx]}
        return None
    
    return {
        'extract_identifiers': extract_eraslan_identifiers,
        'id_vars': ['Split'],
        'value_columns': ['CellDISECT_1layer', 'CellDISECT_2layer', 'CellDISECT_4layer', 
                   'CellDISECT_6layer', 'CellDISECT_lr0.006', 'CellDISECT_lr0.01', 
                   'CellDISECT_lr0.03', 'CellDISECT_lr0.06', 'CellDISECT_lr0.1', 'Control'],  # Already using CellDISECT
        'categorical_order': {
            'Split': [
                'ImmuneCT Lung: Female2Male',
                'Epithelial luminal: Female2Male & Breast2Prostate', 
                'Immune Male: Ventricle2Lung'
            ]
        }
    }

# 3. Eraslan dataset
csv_path_eraslan_old = 'cf_metrics/old'
csv_path_eraslan_new = 'cf_metrics/new'
eraslan_config = create_eraslan_config()
split_names_map = {
    '1': 'ImmuneCT Lung: Female2Male',
    '2': 'Epithelial luminal: Female2Male & Breast2Prostate',
    '4': 'Immune Male: Ventricle2Lung',
}
eraslan_filter = {'contains': 'noCT'}
eraslan = load_metrics_dual_directory(csv_path_eraslan_old, csv_path_eraslan_new, 'Eraslan', 
                                     eraslan_config, split_name_mapping=split_names_map, 
                                     filter_conditions=eraslan_filter, spearman_stat='mean')


import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

default_palette = sns.color_palette()  # This gives you the default colors

custom_palette = {
        'Control': default_palette[-1],  # Light Blue
        'CellDISECT_1layer': default_palette[0],     # Blue
        'CellDISECT_2layer': default_palette[1],     # Orange
        'CellDISECT_4layer': default_palette[2],     # Green
        'CellDISECT_6layer': default_palette[3],     # Red
        'CellDISECT_lr0.006': default_palette[4],    # Purple
        'CellDISECT_lr0.01': default_palette[5],     # Brown
        'CellDISECT_lr0.03': default_palette[6],     # Pink
        'CellDISECT_lr0.06': default_palette[7],     # Gray
        'CellDISECT_lr0.1': default_palette[8],      # Olive
    }
    
# Define order for models
model_order = ['Control', 'CellDISECT_1layer', 'CellDISECT_2layer', 'CellDISECT_4layer', 
               'CellDISECT_6layer', 'CellDISECT_lr0.006', 'CellDISECT_lr0.01', 
               'CellDISECT_lr0.03', 'CellDISECT_lr0.06', 'CellDISECT_lr0.1']
pal = [custom_palette[i] for i in model_order]

def plot_aggregation(
    df,
    genes,
    save_name,
    hue_order=model_order,
    pal_to_use=pal[1:],
    spearman_stat_display=None,  # NEW: Optional display of which Spearman stat is being shown
):
    """
    Plot aggregated metrics with support for new Spearman statistics
    Energy Distance is plotted separately due to different scale
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame from load_metrics_dual_directory
    genes : str
        Gene subset to plot
    save_name : str
        Name for saved file
    hue_order : list
        Order of methods in plot
    pal_to_use : list
        Colors to use
    spearman_stat_display : str, optional
        Which Spearman statistic is being displayed (for labeling)
    """
    
    # Check if dataframe contains spearman_stat_used info
    spearman_stat_used = None
    if 'spearman_stat_used' in df.columns:
        spearman_stat_used = df['spearman_stat_used'].iloc[0]
    elif spearman_stat_display:
        spearman_stat_used = spearman_stat_display
    
    # Define metric display order and direction (↓ = lower is better, ↑ = higher is better)
    metric_order = ['Pearson', 'Delta Pearson', 'Spearman', 'Delta Spearman', 'EMD', 'MMD', 'Energy Distance']
    metric_direction = {
        'Pearson': '↑',
        'Delta Pearson': '↑',
        'Spearman': '↑',
        'Delta Spearman': '↑',
        'EMD': '↓',
        'MMD': '↓',
        'Energy Distance': '↓'
    }
    
    # Generate display labels with Spearman statistic info if available
    metric_labels = {}
    for metric in metric_order:
        direction = metric_direction[metric]
        if metric in ['Spearman', 'Delta Spearman'] and spearman_stat_used:
            label = f"{metric} ({spearman_stat_used}) {direction}"
        else:
            label = f"{metric} {direction}"
        metric_labels[metric] = label

    data = df[df['Genes'] == genes].copy()
    
    # Separate Energy Distance from other metrics
    energy_data = data[data['Metric'] == 'Energy Distance'].copy()
    other_data = data[data['Metric'] != 'Energy Distance'].copy()
    
    # Set categorical order for other metrics (excluding Energy Distance)
    other_metrics_order = [m for m in metric_order if m != 'Energy Distance']
    other_data['Metric'] = pd.Categorical(other_data['Metric'], categories=other_metrics_order, ordered=True)
    
    sns.set_theme(style="ticks", font_scale=1.8)
    
    # Create figure with two subplots vertically stacked
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 12), gridspec_kw={'height_ratios': [6, 1]})
    
    # Plot other metrics (top subplot)
    sns.barplot(
        data=other_data, 
        y='Metric', x='value', hue='method',
        hue_order=hue_order,
        palette=pal_to_use,
        orient='h',
        errorbar='se',
        ax=ax1
    )
    
    # Plot Energy Distance (bottom subplot)
    sns.barplot(
        data=energy_data, 
        y='Metric', x='value', hue='method',
        hue_order=hue_order,
        palette=pal_to_use,
        orient='h',
        errorbar='se',
        ax=ax2
    )
    
    # Customize top subplot
    ax1.set_ylabel(None)
    ax1.set_xlabel('')  # Remove x-label from top plot
    
    # Update y-axis labels for top subplot
    other_metric_labels = [metric_labels[m] for m in other_metrics_order]
    ax1.set_yticklabels(other_metric_labels)
    
    # Add value labels to top subplot
    for container in ax1.containers:
        labels = ax1.bar_label(container, fmt='%.2f', fontsize=11, padding=5, label_type='edge')
        for label in labels:
            label.set_y(label.get_position()[1] + 6.1)
    
    # Customize bottom subplot
    ax2.set_ylabel(None)
    ax2.set_xlabel('Value', fontweight='bold')
    ax2.set_yticklabels([metric_labels['Energy Distance']])
    
    # Add value labels to bottom subplot
    for container in ax2.containers:
        labels = ax2.bar_label(container, fmt='%.1f', fontsize=11, padding=5, label_type='edge')
        for label in labels:
            label.set_y(label.get_position()[1] + 6.1)
    
    # Handle legends - only show on top subplot
    ax1_legend = ax1.legend()
    new_labels = model_order
    for t, l in zip(ax1_legend.texts, new_labels):
        t.set_text(l)
    
    # Position legend outside the plot area
    ax1.legend(loc="upper left", bbox_to_anchor=(1.02, 1.0))
    
    # Remove legend from bottom subplot
    ax2.legend().remove()
    
    # Apply despine to both subplots
    sns.despine(ax=ax1)
    sns.despine(ax=ax2)
    
    # Adjust tick parameters
    ax1.tick_params(axis='x', which='major', pad=10)
    ax1.tick_params(axis='y', which='major', pad=5)
    ax2.tick_params(axis='x', which='major', pad=10)
    ax2.tick_params(axis='y', which='major', pad=5)
    
    # # Add title with Spearman statistic info if available
    # if spearman_stat_used:
    #     fig.suptitle(f'Spearman correlations using {spearman_stat_used} statistic', 
    #                 fontsize=16, y=0.95)
    
    # Adjust layout
    plt.tight_layout()
    
    # Save figures
    plt.savefig(f'cf_metrics/supp_plots_rev1/{save_name}.svg', bbox_inches='tight', dpi=300) 
    plt.savefig(f'cf_metrics/supp_plots_rev1/{save_name}.png', bbox_inches='tight', dpi=300) 
    plt.show()
    plt.clf()



metric_order = ['Pearson', 'Delta Pearson', 'Spearman', 'Delta Spearman', 'EMD', 'MMD', 'Energy Distance']

def plot_aggregation_lr_models(
    df,
    genes,
    save_name,
    spearman_stat_display=None,
):
    """
    Plot aggregated metrics for learning rate varying models only
    """
    # Filter for learning rate models
    lr_models = ['Control', 'CellDISECT_lr0.006', 'CellDISECT_lr0.01', 'CellDISECT_lr0.03', 'CellDISECT_lr0.06', 'CellDISECT_lr0.1']
    lr_pal = [custom_palette[model] for model in lr_models]
    
    # Check if dataframe contains spearman_stat_used info
    spearman_stat_used = None
    if 'spearman_stat_used' in df.columns:
        spearman_stat_used = df['spearman_stat_used'].iloc[0]
    elif spearman_stat_display:
        spearman_stat_used = spearman_stat_display
    
    # Define metric display order and direction (↓ = lower is better, ↑ = higher is better)
    metric_order = ['Pearson', 'Delta Pearson', 'Spearman', 'Delta Spearman', 'EMD', 'MMD', 'Energy Distance']
    metric_direction = {
        'Pearson': '↑',
        'Delta Pearson': '↑',
        'Spearman': '↑',
        'Delta Spearman': '↑',
        'EMD': '↓',
        'MMD': '↓',
        'Energy Distance': '↓'
    }
    
    # Generate display labels with Spearman statistic info if available
    metric_labels = {}
    for metric in metric_order:
        direction = metric_direction[metric]
        if metric in ['Spearman', 'Delta Spearman'] and spearman_stat_used:
            label = f"{metric} ({spearman_stat_used}) {direction}"
        else:
            label = f"{metric} {direction}"
        metric_labels[metric] = label

    data = df[df['Genes'] == genes].copy()
    data = data[data['method'].isin(lr_models)].copy()
    
    # Separate Energy Distance from other metrics
    energy_data = data[data['Metric'] == 'Energy Distance'].copy()
    other_data = data[data['Metric'] != 'Energy Distance'].copy()
    
    # Set categorical order for other metrics (excluding Energy Distance)
    other_metrics_order = [m for m in metric_order if m != 'Energy Distance']
    other_data['Metric'] = pd.Categorical(other_data['Metric'], categories=other_metrics_order, ordered=True)
    
    sns.set_theme(style="ticks", font_scale=1.8)
    
    # Create figure with two subplots vertically stacked
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 12), gridspec_kw={'height_ratios': [6, 1]})
    
    # Plot other metrics (top subplot)
    sns.barplot(
        data=other_data, 
        y='Metric', x='value', hue='method',
        hue_order=lr_models,
        palette=lr_pal,
        orient='h',
        errorbar='se',
        ax=ax1
    )
    
    # Plot Energy Distance (bottom subplot)
    sns.barplot(
        data=energy_data, 
        y='Metric', x='value', hue='method',
        hue_order=lr_models,
        palette=lr_pal,
        orient='h',
        errorbar='se',
        ax=ax2
    )
    
    # Customize top subplot
    ax1.set_ylabel(None)
    ax1.set_xlabel('')  # Remove x-label from top plot
    
    # Update y-axis labels for top subplot
    other_metric_labels = [metric_labels[m] for m in other_metrics_order]
    ax1.set_yticklabels(other_metric_labels)
    
    # Add value labels to top subplot
    for container in ax1.containers:
        labels = ax1.bar_label(container, fmt='%.2f', fontsize=11, padding=5, label_type='edge')
        for label in labels:
            label.set_y(label.get_position()[1] + 6.1)
    
    # Customize bottom subplot
    ax2.set_ylabel(None)
    ax2.set_xlabel('Value', fontweight='bold')
    ax2.set_yticklabels([metric_labels['Energy Distance']])
    
    # Add value labels to bottom subplot
    for container in ax2.containers:
        labels = ax2.bar_label(container, fmt='%.1f', fontsize=11, padding=5, label_type='edge')
        for label in labels:
            label.set_y(label.get_position()[1] + 6.1)
    
    # Handle legends - only show on top subplot
    ax1_legend = ax1.legend()
    new_labels = lr_models
    for t, l in zip(ax1_legend.texts, new_labels):
        t.set_text(l)
    
    # Position legend outside the plot area
    ax1.legend(loc="upper left", bbox_to_anchor=(1.02, 1.0))
    
    # Remove legend from bottom subplot
    ax2.legend().remove()
    
    # Apply despine to both subplots
    sns.despine(ax=ax1)
    sns.despine(ax=ax2)
    
    # Adjust tick parameters
    ax1.tick_params(axis='x', which='major', pad=10)
    ax1.tick_params(axis='y', which='major', pad=5)
    ax2.tick_params(axis='x', which='major', pad=10)
    ax2.tick_params(axis='y', which='major', pad=5)
    
    # Add title based on genes
    gene_title = "Top 20 Genes" if genes == '20' else "All Genes"
    fig.suptitle(f'Learning Rate - {gene_title}', fontsize=16, y=0.95)
    
    # Adjust layout
    plt.tight_layout()
    
    # Save figures
    plt.savefig(f'cf_metrics/supp_plots_rev1/{save_name}_lr.svg', bbox_inches='tight', dpi=300) 
    plt.savefig(f'cf_metrics/supp_plots_rev1/{save_name}_lr.png', bbox_inches='tight', dpi=300) 
    plt.show()
    plt.clf()

def plot_aggregation_nlayers_models(
    df,
    genes,
    save_name,
    spearman_stat_display=None,
):
    """
    Plot aggregated metrics for n_layers varying models only
    """
    # Filter for n_layers models
    nlayers_models = ['Control', 'CellDISECT_1layer', 'CellDISECT_2layer', 'CellDISECT_4layer', 'CellDISECT_6layer']
    nlayers_pal = [custom_palette[model] for model in nlayers_models]
    
    # Check if dataframe contains spearman_stat_used info
    spearman_stat_used = None
    if 'spearman_stat_used' in df.columns:
        spearman_stat_used = df['spearman_stat_used'].iloc[0]
    elif spearman_stat_display:
        spearman_stat_used = spearman_stat_display
    
    # Define metric display order and direction (↓ = lower is better, ↑ = higher is better)
    metric_order = ['Pearson', 'Delta Pearson', 'Spearman', 'Delta Spearman', 'EMD', 'MMD', 'Energy Distance']
    metric_direction = {
        'Pearson': '↑',
        'Delta Pearson': '↑',
        'Spearman': '↑',
        'Delta Spearman': '↑',
        'EMD': '↓',
        'MMD': '↓',
        'Energy Distance': '↓'
    }
    
    # Generate display labels with Spearman statistic info if available
    metric_labels = {}
    for metric in metric_order:
        direction = metric_direction[metric]
        if metric in ['Spearman', 'Delta Spearman'] and spearman_stat_used:
            label = f"{metric} ({spearman_stat_used}) {direction}"
        else:
            label = f"{metric} {direction}"
        metric_labels[metric] = label

    data = df[df['Genes'] == genes].copy()
    data = data[data['method'].isin(nlayers_models)].copy()
    
    # Separate Energy Distance from other metrics
    energy_data = data[data['Metric'] == 'Energy Distance'].copy()
    other_data = data[data['Metric'] != 'Energy Distance'].copy()
    
    # Set categorical order for other metrics (excluding Energy Distance)
    other_metrics_order = [m for m in metric_order if m != 'Energy Distance']
    other_data['Metric'] = pd.Categorical(other_data['Metric'], categories=other_metrics_order, ordered=True)
    
    sns.set_theme(style="ticks", font_scale=1.8)
    
    # Create figure with two subplots vertically stacked
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 12), gridspec_kw={'height_ratios': [6, 1]})
    
    # Plot other metrics (top subplot)
    sns.barplot(
        data=other_data, 
        y='Metric', x='value', hue='method',
        hue_order=nlayers_models,
        palette=nlayers_pal,
        orient='h',
        errorbar='se',
        ax=ax1
    )
    
    # Plot Energy Distance (bottom subplot)
    sns.barplot(
        data=energy_data, 
        y='Metric', x='value', hue='method',
        hue_order=nlayers_models,
        palette=nlayers_pal,
        orient='h',
        errorbar='se',
        ax=ax2
    )
    
    # Customize top subplot
    ax1.set_ylabel(None)
    ax1.set_xlabel('')  # Remove x-label from top plot
    
    # Update y-axis labels for top subplot
    other_metric_labels = [metric_labels[m] for m in other_metrics_order]
    ax1.set_yticklabels(other_metric_labels)
    
    # Add value labels to top subplot
    for container in ax1.containers:
        labels = ax1.bar_label(container, fmt='%.2f', fontsize=11, padding=5, label_type='edge')
        for label in labels:
            label.set_y(label.get_position()[1] + 6.1)
    
    # Customize bottom subplot
    ax2.set_ylabel(None)
    ax2.set_xlabel('Value', fontweight='bold')
    ax2.set_yticklabels([metric_labels['Energy Distance']])
    
    # Add value labels to bottom subplot
    for container in ax2.containers:
        labels = ax2.bar_label(container, fmt='%.1f', fontsize=11, padding=5, label_type='edge')
        for label in labels:
            label.set_y(label.get_position()[1] + 6.1)
    
    # Handle legends - only show on top subplot
    ax1_legend = ax1.legend()
    new_labels = nlayers_models
    for t, l in zip(ax1_legend.texts, new_labels):
        t.set_text(l)
    
    # Position legend outside the plot area
    ax1.legend(loc="upper left", bbox_to_anchor=(1.02, 1.0))
    
    # Remove legend from bottom subplot
    ax2.legend().remove()
    
    # Apply despine to both subplots
    sns.despine(ax=ax1)
    sns.despine(ax=ax2)
    
    # Adjust tick parameters
    ax1.tick_params(axis='x', which='major', pad=10)
    ax1.tick_params(axis='y', which='major', pad=5)
    ax2.tick_params(axis='x', which='major', pad=10)
    ax2.tick_params(axis='y', which='major', pad=5)
    
    # Add title based on genes
    gene_title = "Top 20 Genes" if genes == '20' else "All Genes"
    fig.suptitle(f'N-Layers - {gene_title}', fontsize=16, y=0.95)
    
    # Adjust layout
    plt.tight_layout()
    
    # Save figures
    plt.savefig(f'cf_metrics/supp_plots_rev1/{save_name}_nlayers.svg', bbox_inches='tight', dpi=300) 
    plt.savefig(f'cf_metrics/supp_plots_rev1/{save_name}_nlayers.png', bbox_inches='tight', dpi=300) 
    plt.show()
    plt.clf()