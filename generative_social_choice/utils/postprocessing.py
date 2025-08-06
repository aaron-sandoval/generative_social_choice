"""
This module contains functions for postprocessing results, including plotting and metrics.
"""

import matplotlib.pyplot as plt
import matplotlib
import json
import pandas as pd
import numpy as np
import scipy
from generative_social_choice.utils.helper_functions import get_base_dir_path
from generative_social_choice.slates.voting_utils import gini
from generative_social_choice.ratings.utility_matrix import extract_voter_utilities_from_info_csv

LIKERT_SCORES: dict[int, str] = {
    0: "Not at all",
    1: "Poorly",
    2: "Somewhat",
    3: "Mostly",
    4: "Perfectly",
}

LIKERT_SCORES_INVERSE: dict[str, int] = {v: k for k, v in LIKERT_SCORES.items()}

def calculate_proportion_confidence_intervals(counts: np.ndarray, total: int) -> np.ndarray:
    """
    Calculate confidence intervals for proportions, including exact calculation for small counts where the normal approximation is not valid.
    """
    proportions = counts / total
    nqp = counts * proportions
    standard_errors = np.sqrt(proportions * (1 - proportions) / total)
    
    # Initialize bounds with normal approximation
    lower_bounds = np.maximum(0, proportions - standard_errors)
    upper_bounds = np.minimum(1, proportions + standard_errors)
    
    # Identify indices where exact calculation is needed
    exact_indices = nqp < 5
    
    # Exact binomial calculation for nqp < 5
    if np.any(exact_indices):
        exact_lower = scipy.stats.binom.ppf(0.025, total, proportions[exact_indices]) / total
        exact_upper = scipy.stats.binom.ppf(0.975, total, proportions[exact_indices]) / total
        lower_bounds[exact_indices] = exact_lower
        upper_bounds[exact_indices] = exact_upper
    
    return np.vstack((lower_bounds, proportions, upper_bounds)).T


def scalar_utility_metrics(utilities: pd.DataFrame) -> pd.DataFrame:
    """Calculate a set of scalar metrics from a DataFrame of utilities for different scenarios.
    
    Args:
        utilities: A DataFrame of utilities for different scenarios.
        Columns are scenarios, rows are voters.

    Returns:
        A DataFrame of scalar metrics.
    """
    scalar_metrics = pd.DataFrame(index=utilities.columns, columns=["Avg_Utility", "Min_Utility", r"25th_Pctile_Utility", "Gini"])

    scalar_metrics.Avg_Utility = utilities.mean(0).T
    scalar_metrics.Min_Utility = utilities.min(0).T
    scalar_metrics["25th_Pctile_Utility"] = utilities.quantile(0.25, axis=0).T
    # Calculate Gini coefficient using scipy's implementation

    scalar_metrics.Gini = utilities.apply(gini)

    return scalar_metrics


def bootstrap_df_rows(
    data: pd.DataFrame,
    confidence_level: float = 0.95,
    n_bootstrap: int = 400
) -> pd.DataFrame:
    """
    Calculate bootstrap confidence intervals for the mean across rows grouped by MultiIndex.
    
    All levels of the MultiIndex except the last are treated as grouping indices,
    where rows with the same grouping indices belong to a group. The last level is
    a sample ID. If there is no MultiIndex, then the DataFrame is treated as having
    1 group. The columns represent metrics and should be unchanged in the output.
    
    Args:
        data: DataFrame with potentially MultiIndex rows. All levels except the last
              are grouping indices, the last level is sample ID.
        confidence_level: Width of the confidence interval for the mean (default: 0.95). 
        n_bootstrap: Number of bootstrap samples to generate (default: 400).
    
    Returns:
        DataFrame with same columns as input. Rows have MultiIndex with statistics
        as the innermost level: 'Mean', f'{confidence_level}% lower bound', 
        f'{confidence_level}% upper bound'.
    """
    # Calculate confidence interval bounds
    alpha = 1 - confidence_level
    lower_percentile = (alpha / 2) * 100
    upper_percentile = (1 - alpha / 2) * 100
    
    # Create statistic labels
    confidence_pct = int(confidence_level * 100)
    statistics = [
        f'{confidence_pct}% lower bound',
        'Mean',
        f'{confidence_pct}% upper bound'
    ]
    
    # Handle MultiIndex vs simple index
    if isinstance(data.index, pd.MultiIndex):
        # Get grouping levels (all except last)
        n_levels = data.index.nlevels
        grouping_levels = list(range(n_levels - 1))
        
        # Group by all levels except the last
        groups = data.groupby(level=grouping_levels)
        group_names = list(groups.groups.keys())
    else:
        # Treat as single group
        groups = [('all', data)]
        group_names = ['all']
    
    # Process each group
    results_list = []
    row_indices = []
    
    for group_name, group_data in groups:
        # Convert to numpy array for bootstrapping
        group_array = group_data.values
        n_samples, n_metrics = group_array.shape
        
        # Generate bootstrap samples
        bootstrap_means = []
        for _ in range(n_bootstrap):
            # Sample with replacement from the rows
            bootstrap_indices = np.random.choice(n_samples, size=n_samples, replace=True)
            bootstrap_sample = group_array[bootstrap_indices]
            bootstrap_mean = np.mean(bootstrap_sample, axis=0)
            bootstrap_means.append(bootstrap_mean)
        
        bootstrap_means = np.array(bootstrap_means)
        
        # Calculate statistics
        sample_mean = np.mean(group_array, axis=0)
        lower_bound = np.percentile(bootstrap_means, lower_percentile, axis=0)
        upper_bound = np.percentile(bootstrap_means, upper_percentile, axis=0)
        
        # Add results for this group - one row per statistic
        results_list.append(lower_bound)
        results_list.append(sample_mean)
        results_list.append(upper_bound)
        
        # Create row indices
        if isinstance(group_name, tuple):
            # Multi-level grouping
            for stat in statistics:
                row_indices.append(group_name + (stat,))
        else:
            # Single-level grouping
            for stat in statistics:
                row_indices.append((group_name, stat))
    
    # Create the result DataFrame
    if len(group_names) == 1 and group_names[0] == 'all':
        # Single group case - create simple MultiIndex
        row_index = pd.MultiIndex.from_tuples(
            [(group_names[0], stat) for stat in statistics],
            names=['group', 'statistic']
        )
    else:
        # Multiple groups case - preserve original grouping structure
        if isinstance(data.index, pd.MultiIndex):
            # Get original level names (except the last one)
            original_names = list(data.index.names[:-1]) + ['statistic']
        else:
            original_names = ['group', 'statistic']
        
        row_index = pd.MultiIndex.from_tuples(row_indices, names=original_names)
    
    result_df = pd.DataFrame(
        results_list,
        index=row_index,
        columns=data.columns
    )
    
    return result_df


def plot_likert_category_bar_chart(assignments: pd.DataFrame) -> plt.figure:
    """
    Bar chart of the distribution of likert scores with error bars.

    Utilities are rounded to the nearest integer and mapped to the likert scale.
    Bars are sequenced from lowest to highest likert score.
    The y axis is the percentage of the sample that received each score.
    Error bars represent the standard error of the proportion.
    """
    # Round utilities and map to likert scores
    assignments["utility"] = assignments["utility"].round().astype(int)
    assignments["likert_score"] = assignments["utility"].map(LIKERT_SCORES)
    
    # Calculate counts
    total = len(assignments)
    counts = assignments["likert_score"].value_counts().sort_index()

    confidence_intervals = calculate_proportion_confidence_intervals(counts.values, total)
    
    # Create DataFrame for plotting
    plot_data = pd.DataFrame({
        "likert_score": counts.index,
        "proportion": confidence_intervals[:, 1],
        "lower_bound": confidence_intervals[:, 0],
        "upper_bound": confidence_intervals[:, 2]
    }).sort_values("likert_score")

    # Sort the plot data by likert score using the reverse mapping
    plot_data = plot_data.sort_values(
        "likert_score", 
        key=lambda x: x.map(LIKERT_SCORES_INVERSE)
    )
    
    # Convert proportions to percentages for plotting
    plot_data["percentage"] = plot_data["proportion"] * 100
    plot_data["lower_bound_percentage"] = plot_data["lower_bound"] * 100
    plot_data["upper_bound_percentage"] = plot_data["upper_bound"] * 100
    
    yerr = np.array([
        plot_data["percentage"] - plot_data["lower_bound_percentage"],
        plot_data["upper_bound_percentage"] - plot_data["percentage"]
    ])
    
    # Create figure and axis objects explicitly
    fig, ax = plt.subplots()
    
    # Create bar plot
    bars = ax.bar(
        x=range(len(plot_data)),
        height=plot_data["percentage"],
        yerr=yerr,
        capsize=5
    )
    
    # Customize x-axis
    ax.set_xticks(range(len(plot_data)))
    ax.set_xticklabels(
        plot_data["likert_score"],
        # rotation=45 if len(plot_data) > 4 else 0
    )
    
    # Customize y-axis
    ax.set_ylabel("Percentage of Population (%)")
    
    return fig


def plot_likert_category_clustered_bar_chart(
    utilities: pd.DataFrame,
    labels: dict[str, str] | None = None,
) -> plt.Figure:
    """
    Clustered bar chart comparing the distribution of likert scores across multiple
    datasets with error bars.

    Args:
        utilities: DataFrame where each column contains utilities for a different
            dataset/condition
        labels: Optional dictionary mapping column names to display labels. If not
            provided, uses the column names directly.

    Returns:
        matplotlib Figure object containing the plot
    """
    # Process each dataset/column
    plot_data_dict = {}
    for column in utilities.columns:
        # Round utilities and map to likert scores
        df = pd.DataFrame({"utility": utilities[column]}).copy()
        df["utility"] = df["utility"].round().astype(int)
        df["likert_score"] = df["utility"].map(LIKERT_SCORES)
        
        # Calculate counts and proportions
        total = len(df)
        counts = df["likert_score"].value_counts().sort_index()
        
        # Calculate confidence intervals
        confidence_intervals = calculate_proportion_confidence_intervals(
            counts.values, 
            total
        )
        
        # Create DataFrame for this dataset
        plot_data = pd.DataFrame({
            "likert_score": counts.index,
            "proportion": confidence_intervals[:, 1],
            "lower_bound": confidence_intervals[:, 0],
            "upper_bound": confidence_intervals[:, 2]
        })
        
        # Sort by likert score
        plot_data = plot_data.sort_values(
            "likert_score", 
            key=lambda x: x.map(LIKERT_SCORES_INVERSE)
        )
        
        # Convert to percentages
        plot_data["percentage"] = plot_data["proportion"] * 100
        plot_data["lower_bound_percentage"] = plot_data["lower_bound"] * 100
        plot_data["upper_bound_percentage"] = plot_data["upper_bound"] * 100
        
        plot_data_dict[column] = plot_data

    # Set up the plot
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Calculate bar positions
    n_datasets = len(utilities.columns)
    bar_width = 0.8 / n_datasets  # Adjust total width of group
    
    # Get unique sorted likert scores across all datasets
    all_scores = sorted(
        set().union(*[df["likert_score"] for df in plot_data_dict.values()]),
        key=lambda x: LIKERT_SCORES_INVERSE[x]
    )
    x = np.arange(len(all_scores))
    
    # Plot bars for each dataset
    for i, (column, plot_data) in enumerate(plot_data_dict.items()):
        # Calculate bar positions
        offset = (i - (n_datasets - 1) / 2) * bar_width
        x_pos = x + offset
        
        # Ensure data exists for all scores
        heights = []
        yerr = [[], []]
        for score in all_scores:
            score_data = plot_data[plot_data["likert_score"] == score]
            if len(score_data) > 0:
                heights.append(score_data["percentage"].iloc[0])
                yerr[0].append(
                    score_data["percentage"].iloc[0] - 
                    score_data["lower_bound_percentage"].iloc[0]
                )
                yerr[1].append(
                    score_data["upper_bound_percentage"].iloc[0] - 
                    score_data["percentage"].iloc[0]
                )
            else:
                heights.append(0)
                yerr[0].append(0)
                yerr[1].append(0)
        
        # Plot bars
        label = labels[column] if labels else column
        ax.bar(
            x_pos,
            heights,
            bar_width,
            yerr=yerr,
            capsize=5,
            label=label
        )
    
    # Customize plot
    ax.set_ylabel("Percentage of Population (%)")
    ax.set_xticks(x)
    ax.set_xticklabels(all_scores)
    ax.yaxis.set_major_locator(plt.MultipleLocator(10))
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Add legend
    ax.legend()
    
    # Adjust layout to prevent label cutoff
    plt.tight_layout()
    
    return fig


def plot_sorted_utility_distributions(utilities: pd.DataFrame, figsize: tuple[float, float] = (10, 6)) -> plt.Figure:
    """
    Plot sorted utility distributions for each column in the DataFrame.
    
    Args:
        utilities: DataFrame where each column contains utilities for different
            methods/conditions. Can have either a simple column index or a 2-level
            MultiIndex. If using a MultiIndex, columns with the same first-level
            index will be plotted with similar colors.
    
    Returns:
        matplotlib Figure object containing the plot
    """
    # Create figure and axis
    fig, ax = plt.subplots(figsize=figsize)
    
    # Check if we have a MultiIndex
    if isinstance(utilities.columns, pd.MultiIndex):
        # Get unique first-level indices
        first_level_indices = sorted(set(utilities.columns.get_level_values(0)))
        
        # Define primary colors for each group - using darker, more muted colors
        primary_colors = {
            idx: color for idx, color in zip(
                first_level_indices,
                [
                    '#1a365d',  # dark blue
                    '#7c2d12',  # dark orange
                    '#145214',  # dark green
                    '#7f1d1d',  # dark red
                    '#4c1d95',  # dark purple
                    '#713f12',  # dark brown
                    '#0f766e',  # dark teal
                    '#831843',  # dark pink
                ]
            )
        }
        
        # Create custom colormaps that transition from primary color to some average of the primary color and white
        colormaps = {}
        for idx in first_level_indices:
            primary = primary_colors[idx]
            # Convert hex to RGB
            primary_rgb = tuple(int(primary[i:i+2], 16)/255 for i in (1, 3, 5))
            # Create colormap from primary color to some average of the primary color and white
            colormaps[idx] = matplotlib.colors.LinearSegmentedColormap.from_list(
                f'custom_{idx}',
                [primary_rgb, tuple(0.7 + 0.3*x for x in primary_rgb)],
                N=8
            )
        
        # Track which first-level indices we've added to the legend
        legend_added = set()
        
        # Plot each column's sorted distribution
        for column in utilities.columns:
            first_level = column[0]
            second_level = column[1]
            sorted_values = utilities[column].sort_values(ascending=False).values
            indices = np.arange(len(sorted_values))
            
            # Get color from appropriate colormap
            cmap = colormaps[first_level]
            # Use index to get color, ensuring we use the full range of the colormap
            color_idx = list(utilities.columns.get_level_values(1)).index(second_level)
            color = cmap(color_idx / (len(utilities.columns.get_level_values(1)) - 1))
            
            # Only add label for the first occurrence of each first-level index
            label = first_level if first_level not in legend_added else None
            if label:
                legend_added.add(first_level)
            
            ax.plot(
                indices, 
                sorted_values, 
                label=label,
                color=color
            )
    else:
        # Original behavior for simple column index
        for column in utilities.columns:
            sorted_values = utilities[column].sort_values(ascending=False).values
            indices = np.arange(len(sorted_values))
            ax.plot(
                indices, 
                sorted_values, 
                label=column, 
                color=plt.cm.tab20(utilities.columns.get_loc(column))
            )
    
    # Customize plot
    ax.set_xlabel("Voter index (sorted by utility)")
    ax.set_ylabel("Utility")
    ax.grid(axis='both', linestyle='--', alpha=0.7)
    ax.legend(loc='lower left')
    
    # Adjust layout to prevent label cutoff
    plt.tight_layout()
    
    return fig


def plot_sorted_utility_CIs(
    utilities: pd.DataFrame, 
    confidence_level: float = 0.95,
    n_bootstrap: int = 400,
    figsize: tuple[float, float] = (10, 6)
) -> plt.Figure:
    """
    Plot confidence intervals for sorted utility distributions using bootstrapping.
    
    Instead of plotting each line individually, this function treats the lines as a 
    sample from a population and plots confidence intervals of the mean trajectory 
    across the domain using bootstrapping. When there is a MultiIndex on the columns, 
    it groups columns by the first level and plots separate confidence intervals for each group.
    
    Args:
        utilities: DataFrame where each column contains utilities for different
            methods/conditions. Can have either a simple column index or a 2-level
            MultiIndex. If using a MultiIndex, columns with the same first-level
            index will be grouped together.
        confidence_level: Width of the confidence interval (default: 0.95).
        n_bootstrap: Number of bootstrap samples to generate (default: 400).
    
    Returns:
        matplotlib Figure object containing the plot
    """
    # Create figure and axis
    fig, ax = plt.subplots(figsize=figsize)
    
    # Prepare data for bootstrap_df_rows
    # We need to create a DataFrame where rows represent sorted utility trajectories
    # and columns represent position indices
    
    # First, calculate sorted utilities for all columns
    sorted_utilities_dict = {}
    for column in utilities.columns:
        sorted_values = utilities[column].sort_values(ascending=False).values
        sorted_utilities_dict[column] = sorted_values
    
    # Create DataFrame with sorted trajectories as rows
    max_length = max(len(values) for values in sorted_utilities_dict.values())
    
    # Create position column names
    position_columns = [f"pos_{i}" for i in range(max_length)]
    
    # Prepare data for DataFrame
    data_rows = []
    row_indices = []
    
    if isinstance(utilities.columns, pd.MultiIndex):
        # Group by first level of MultiIndex
        for column in utilities.columns:
            # Pad shorter trajectories with NaN if needed
            values = sorted_utilities_dict[column]
            if len(values) < max_length:
                values = np.concatenate([values, [np.nan] * (max_length - len(values))])
            data_rows.append(values)
            
            # Create MultiIndex for rows: (group, sample_id)
            group_name = column[0]
            sample_id = column[1] if len(column) > 1 else column
            row_indices.append((group_name, sample_id))
        
        # Create MultiIndex for rows
        row_index = pd.MultiIndex.from_tuples(row_indices, names=['group', 'sample'])
    else:
        # Simple column index - treat as single group
        for i, column in enumerate(utilities.columns):
            values = sorted_utilities_dict[column]
            if len(values) < max_length:
                values = np.concatenate([values, [np.nan] * (max_length - len(values))])
            data_rows.append(values)
            
            # Create MultiIndex for rows: (group, sample_id)
            row_indices.append(('all', column))
        
        # Create MultiIndex for rows
        row_index = pd.MultiIndex.from_tuples(row_indices, names=['group', 'sample'])
    
    # Create DataFrame for bootstrap analysis
    bootstrap_data = pd.DataFrame(
        data_rows,
        index=row_index,
        columns=position_columns
    )
    
    # Use bootstrap_df_rows to calculate confidence intervals
    bootstrap_results = bootstrap_df_rows(
        bootstrap_data,
        confidence_level=confidence_level,
        n_bootstrap=n_bootstrap
    )
    
    # Define colors for each group
    colors = [
        '#1a365d',  # dark blue
        '#7c2d12',  # dark orange
        '#145214',  # dark green
        '#7f1d1d',  # dark red
        '#4c1d95',  # dark purple
        '#713f12',  # dark brown
        '#0f766e',  # dark teal
        '#831843',  # dark pink
    ]
    
    # Get unique groups from the bootstrap results
    if isinstance(bootstrap_results.index, pd.MultiIndex):
        # Get unique group names (all levels except the last 'statistic' level)
        unique_groups = bootstrap_results.index.droplevel(-1).unique()
    else:
        unique_groups = ['all']
    
    # Create statistic labels
    confidence_pct = int(confidence_level * 100)
    mean_label = 'Mean'
    lower_label = f'{confidence_pct}% lower bound'
    upper_label = f'{confidence_pct}% upper bound'
    
    # Plot results for each group
    for i, group_name in enumerate(unique_groups):
        # Extract mean, lower, and upper bounds for all positions
        if isinstance(bootstrap_results.index, pd.MultiIndex):
            means = bootstrap_results.loc[(group_name, mean_label), :].values
            lower_bounds = bootstrap_results.loc[(group_name, lower_label), :].values
            upper_bounds = bootstrap_results.loc[(group_name, upper_label), :].values
        else:
            means = bootstrap_results.loc[mean_label, :].values
            lower_bounds = bootstrap_results.loc[lower_label, :].values
            upper_bounds = bootstrap_results.loc[upper_label, :].values
        
        # Find last valid index (remove trailing NaNs)
        valid_indices = ~np.isnan(means)
        if not np.any(valid_indices):
            continue
            
        last_valid = np.where(valid_indices)[0][-1] + 1
        
        means = means[:last_valid]
        lower_bounds = lower_bounds[:last_valid]
        upper_bounds = upper_bounds[:last_valid]
        indices = np.arange(len(means))
        
        # Count number of samples in this group
        if isinstance(utilities.columns, pd.MultiIndex):
            n_samples = sum(1 for col in utilities.columns if col[0] == group_name)
        else:
            n_samples = len(utilities.columns)
        
        # Plot confidence interval as shaded region
        color = colors[i % len(colors)]
        ax.fill_between(
            indices, 
            lower_bounds, 
            upper_bounds, 
            alpha=0.3, 
            color=color,
            label=f"{group_name} {confidence_level:.0%} CI (n={n_samples})"
        )
        
        # Plot sample mean trajectory
        ax.plot(
            indices, 
            means, 
            color=color, 
            linewidth=2,
            label=f"{group_name} (sample mean, n={n_samples})"
        )
    
    # Customize plot
    ax.set_xlabel("Voter index (sorted by utility)")
    ax.set_ylabel("Utility")
    ax.grid(axis='both', linestyle='--', alpha=0.7)
    ax.legend(loc='lower left')
    
    # Adjust layout to prevent label cutoff
    plt.tight_layout()
    
    return fig


def plot_candidate_distribution_stacked(assignments: pd.DataFrame) -> plt.Figure:
    """
    Create a stacked bar chart showing the distribution of candidate assignments.
    Each stack represents a different method/condition, with bars stacked by
    frequency of candidate assignment in descending order.

    Args:
        assignments: DataFrame where each column contains candidate assignments for
            different methods/conditions

    Returns:
        matplotlib Figure object containing the plot
    """
    # Create figure and axis
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Get all unique candidates across all columns, sorted by name
    all_candidates = sorted(set().union(*[
        set(assignments[col].unique()) for col in assignments.columns
    ]))
    
    # Process each column
    x_positions = np.arange(len(assignments.columns))
    max_height = 0
    
    for i, column in enumerate(assignments.columns):
        # Count occurrences of each candidate
        counts = assignments[column].value_counts()
        sorted_counts = counts.sort_values(ascending=True)
        
        # Plot stacked bars
        bottom = 0
        for candidate, count in sorted_counts.items():
            ax.bar(
                i,
                count,
                bottom=bottom,
                label=candidate if i == 0 else "",
                width=0.8,
                # Use same color for same candidate across stacks
                color=plt.cm.tab20(all_candidates.index(candidate) % 20)
            )
            bottom += count
        
        max_height = max(max_height, bottom)
    
    # Customize plot
    ax.set_ylabel("Number of Voters")
    ax.set_xticks(x_positions)
    ax.set_xticklabels(assignments.columns, rotation=45, ha='right')
    ax.yaxis.set_major_locator(plt.MultipleLocator(10))
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Create custom legend handles and labels for all candidates
    handles = [plt.Rectangle((0,0),1,1, 
                           color=plt.cm.tab20(i % 20)) 
              for i in range(len(all_candidates))]
    
    # Add legend outside the plot with all candidates
    ax.legend(
        handles,
        all_candidates,
        bbox_to_anchor=(1.05, 1),
        loc='upper left',
        title="Candidates"
    )
    
    # Adjust layout to prevent label cutoff
    plt.tight_layout()
    
    return fig