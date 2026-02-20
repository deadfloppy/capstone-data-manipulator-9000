"""Helper functions for data loading and manipulation."""

import io
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st


@st.cache_data
def load_csv(_file, file_name: str, file_size: int) -> pd.DataFrame:
    """
    Load a CSV file with automatic encoding and delimiter detection.

    Args:
        _file: File-like object from Streamlit uploader (unhashable, excluded from cache key)
        file_name: Name of the file (used as cache key)
        file_size: Size of the file in bytes (used as cache key)

    Returns:
        pandas DataFrame
    """
    encodings = ['utf-8', 'latin-1', 'cp1252']

    for encoding in encodings:
        try:
            _file.seek(0)
            df = pd.read_csv(_file, encoding=encoding)
            return df
        except UnicodeDecodeError:
            continue
        except Exception as e:
            # Try with different delimiter if comma fails
            try:
                _file.seek(0)
                df = pd.read_csv(_file, encoding=encoding, sep=';')
                return df
            except:
                continue

    # Final fallback
    _file.seek(0)
    return pd.read_csv(_file, encoding='latin-1', on_bad_lines='skip')


def sample_data(df: pd.DataFrame, max_points: int = 50000) -> tuple[pd.DataFrame, bool]:
    """
    Randomly sample data if it exceeds max_points.

    Args:
        df: Input DataFrame
        max_points: Maximum number of rows to return

    Returns:
        Tuple of (sampled DataFrame, was_sampled boolean)
    """
    if len(df) <= max_points:
        return df, False

    return df.sample(n=max_points, random_state=42), True


def get_numeric_columns(df: pd.DataFrame) -> list[str]:
    """Return list of numeric column names."""
    return df.select_dtypes(include=[np.number]).columns.tolist()


def get_categorical_columns(df: pd.DataFrame) -> list[str]:
    """Return list of categorical/object column names."""
    return df.select_dtypes(include=['object', 'category']).columns.tolist()


@st.cache_data
def get_column_info(df: pd.DataFrame) -> pd.DataFrame:
    """
    Generate summary info about DataFrame columns.

    Returns DataFrame with column name, dtype, non-null count, and unique count.
    """
    info = pd.DataFrame({
        'Column': df.columns,
        'Type': df.dtypes.astype(str).values,
        'Non-Null': df.notna().sum().values,
        'Unique': df.nunique().values
    })
    return info


def create_matplotlib_figure(
    df: pd.DataFrame,
    chart_type: str,
    x_col: str,
    y_col: str,
    color_col: str | None = None,
    x_label: str | None = None,
    y_label: str | None = None,
    x_range: list | None = None,
    y_range: list | None = None,
) -> bytes:
    """
    Create a matplotlib figure and return it as PNG bytes.

    Args:
        df: DataFrame to plot
        chart_type: One of "Scatter", "Line", "Bar (frequency)"
        x_col: Column name for X axis
        y_col: Column name for Y axis
        color_col: Optional column name for color grouping
        x_label: Optional label for X axis (defaults to x_col)
        y_label: Optional label for Y axis (defaults to y_col)
        x_range: Optional [min, max] for X axis
        y_range: Optional [min, max] for Y axis

    Returns:
        PNG image as bytes
    """
    x_label = x_label or x_col
    y_label = y_label or y_col
    is_bar = chart_type.startswith("Bar")

    fig, ax = plt.subplots(figsize=(10, 6))

    if color_col is not None:
        groups = df[color_col].unique()
        for group in groups:
            group_df = df[df[color_col] == group]
            if chart_type == "Scatter":
                ax.scatter(group_df[x_col], group_df[y_col], label=str(group), alpha=0.7)
            elif chart_type == "Line":
                sorted_df = group_df.sort_values(x_col)
                ax.plot(sorted_df[x_col], sorted_df[y_col], label=str(group))
            elif is_bar:
                ax.bar(group_df[x_col].astype(str), group_df[y_col], label=str(group), alpha=0.7)
        ax.legend(title=color_col)
    else:
        if chart_type == "Scatter":
            ax.scatter(df[x_col], df[y_col], alpha=0.7)
        elif chart_type == "Line":
            sorted_df = df.sort_values(x_col)
            ax.plot(sorted_df[x_col], sorted_df[y_col])
        elif is_bar:
            ax.bar(df[x_col].astype(str), df[y_col])

    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    if is_bar:
        ax.set_title(f"Frequency of {x_label}" if chart_type == "Bar (frequency)" else f"{y_label} by {x_label}")
    else:
        ax.set_title(f"{y_label} vs {x_label}")

    if is_bar:
        plt.xticks(rotation=45, ha='right')

    if x_range and not is_bar:
        ax.set_xlim(x_range)
    if y_range:
        ax.set_ylim(y_range)

    plt.tight_layout()

    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=150)
    buf.seek(0)
    plt.close(fig)

    return buf.getvalue()


def create_correlation_heatmap(
    corr_matrix: pd.DataFrame,
    labels: list[str] | None = None,
) -> bytes:
    """
    Create a correlation matrix heatmap and return it as PNG bytes.

    Args:
        corr_matrix: Correlation matrix DataFrame
        labels: Optional list of labels for axes

    Returns:
        PNG image as bytes
    """
    fig, ax = plt.subplots(figsize=(10, 8))

    im = ax.imshow(corr_matrix.values, cmap='RdBu_r', vmin=-1, vmax=1, aspect='auto')

    # Add colorbar
    cbar = ax.figure.colorbar(im, ax=ax)
    cbar.ax.set_ylabel("Correlation", rotation=-90, va="bottom")

    # Set ticks and labels
    tick_labels = labels if labels else corr_matrix.columns.tolist()
    ax.set_xticks(range(len(tick_labels)))
    ax.set_yticks(range(len(tick_labels)))
    ax.set_xticklabels(tick_labels, rotation=45, ha='right')
    ax.set_yticklabels(tick_labels)

    # Add correlation values as text
    for i in range(len(corr_matrix)):
        for j in range(len(corr_matrix)):
            val = corr_matrix.iloc[i, j]
            color = "white" if abs(val) > 0.5 else "black"
            ax.text(j, i, f"{val:.2f}", ha="center", va="center", color=color, fontsize=8)

    ax.set_title("Correlation Matrix")
    plt.tight_layout()

    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=150)
    buf.seek(0)
    plt.close(fig)

    return buf.getvalue()
