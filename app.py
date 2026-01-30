"""CSV Data Analysis Tool - A lightweight Streamlit app for data visualization."""

import streamlit as st
import pandas as pd
import plotly.express as px

from utils import (
    load_csv,
    sample_data,
    get_numeric_columns,
    get_categorical_columns,
    get_column_info,
    create_matplotlib_figure,
    create_correlation_heatmap,
)

st.set_page_config(
    page_title="CSV Data Analysis",
    page_icon="📊",
    layout="wide",
)

st.title("CSV Data Analysis Tool")

# File uploads
col_data, col_dict = st.columns(2)

with col_data:
    uploaded_file = st.file_uploader(
        "Upload a CSV file",
        type=["csv"],
        help="Drag and drop or click to browse",
        max_upload_size=500,  # 500 MB
    )

with col_dict:
    dict_file = st.file_uploader(
        "Upload a dictionary file (optional)",
        type=["csv"],
        help="CSV with variable names and descriptions",
        key="dict_uploader",
    )

# Load and configure dictionary
var_descriptions = {}
if dict_file is not None:
    if "dict_df" not in st.session_state or st.session_state.get("dict_filename") != dict_file.name:
        st.session_state.dict_df = load_csv(dict_file)
        st.session_state.dict_filename = dict_file.name

    dict_df = st.session_state.dict_df

    with st.expander("Dictionary Configuration", expanded=True):
        dict_cols = dict_df.columns.tolist()
        dcol1, dcol2 = st.columns(2)
        with dcol1:
            var_name_col = st.selectbox("Variable name column", dict_cols, index=0)
        with dcol2:
            var_desc_col = st.selectbox(
                "Description column",
                dict_cols,
                index=min(1, len(dict_cols) - 1),
            )

        # Build mapping
        var_descriptions = dict(zip(dict_df[var_name_col].astype(str), dict_df[var_desc_col].astype(str)))
        st.success(f"Loaded {len(var_descriptions)} variable descriptions")


def format_var_label(var_name: str) -> str:
    """Format variable name with description for display."""
    if var_name in var_descriptions:
        desc = var_descriptions[var_name]
        if len(desc) > 50:
            desc = desc[:47] + "..."
        return f"{var_name} - {desc}"
    return var_name


def get_var_description(var_name: str) -> str:
    """Get full description for a variable."""
    return var_descriptions.get(var_name, var_name)

if uploaded_file is not None:
    # Load data into session state
    if "df" not in st.session_state or st.session_state.get("filename") != uploaded_file.name:
        with st.spinner("Loading data..."):
            st.session_state.df = load_csv(uploaded_file)
            st.session_state.filename = uploaded_file.name

    df = st.session_state.df

    # Data overview
    st.header("Data Overview")
    col1, col2, col3 = st.columns(3)
    col1.metric("Rows", f"{len(df):,}")
    col2.metric("Columns", len(df.columns))
    col3.metric("Memory", f"{df.memory_usage(deep=True).sum() / 1024 / 1024:.1f} MB")

    # Tabs for preview and info
    tab1, tab2, tab3 = st.tabs(["Preview", "Column Info", "Statistics"])

    with tab1:
        n_rows = st.slider("Rows to display", 5, 100, 10)
        st.dataframe(df.head(n_rows), use_container_width=True)

    with tab2:
        st.dataframe(get_column_info(df), use_container_width=True)

    with tab3:
        st.dataframe(df.describe(), use_container_width=True)

    # Chart configuration in sidebar
    st.sidebar.header("Chart Configuration")

    numeric_cols = get_numeric_columns(df)
    categorical_cols = get_categorical_columns(df)
    all_cols = df.columns.tolist()

    chart_type = st.sidebar.selectbox(
        "Chart Type",
        ["Scatter", "Line", "Bar (frequency)", "Correlation Matrix"],
    )

    # Variable selection depends on chart type
    x_col = None
    y_col = None
    color_col = None
    corr_cols = None

    if chart_type == "Correlation Matrix":
        # Multi-select for correlation matrix
        default_numeric = numeric_cols[:10] if len(numeric_cols) > 10 else numeric_cols
        corr_cols = st.sidebar.multiselect(
            "Variables (numeric only)",
            numeric_cols,
            default=default_numeric,
            format_func=format_var_label,
        )
        if len(corr_cols) < 2:
            st.sidebar.warning("Select at least 2 variables")
    else:
        x_col = st.sidebar.selectbox(
            "X Axis" if chart_type != "Bar (frequency)" else "Variable",
            all_cols,
            index=0,
            format_func=format_var_label,
            help=get_var_description(all_cols[0]) if all_cols else None,
        )
        if x_col:
            st.sidebar.caption(f"*{get_var_description(x_col)}*")

        # Y axis only for non-frequency charts
        if chart_type not in ["Bar (frequency)"]:
            y_col = st.sidebar.selectbox(
                "Y Axis",
                all_cols,
                index=min(1, len(all_cols) - 1),
                format_func=format_var_label,
            )
            if y_col:
                st.sidebar.caption(f"*{get_var_description(y_col)}*")

        # Color by (optional)
        color_options = ["None"] + all_cols
        color_col = st.sidebar.selectbox(
            "Color By",
            color_options,
            index=0,
            format_func=lambda x: format_var_label(x) if x != "None" else "None",
        )
        color_col = None if color_col == "None" else color_col
        if color_col:
            st.sidebar.caption(f"*{get_var_description(color_col)}*")

    # Filter to single X value
    x_filter_value = None
    if chart_type not in ["Correlation Matrix", "Bar (frequency)"] and x_col:
        with st.sidebar.expander("Filter by X Value"):
            enable_x_filter = st.checkbox("Filter to single X value", value=False)
            if enable_x_filter:
                unique_x = sorted(df[x_col].dropna().unique().tolist())
                if len(unique_x) > 0:
                    x_filter_value = st.selectbox(
                        f"Select {x_col} value",
                        unique_x,
                        format_func=lambda v: str(v),
                    )

    # Axis range controls
    with st.sidebar.expander("Axis Ranges"):
        st.caption("Leave empty for auto-scale")
        x_range_col1, x_range_col2 = st.columns(2)
        with x_range_col1:
            x_min = st.number_input("X Min", value=None, key="x_min")
        with x_range_col2:
            x_max = st.number_input("X Max", value=None, key="x_max")

        y_range_col1, y_range_col2 = st.columns(2)
        with y_range_col1:
            y_min = st.number_input("Y Min", value=None, key="y_min")
        with y_range_col2:
            y_max = st.number_input("Y Max", value=None, key="y_max")

    # Build axis ranges (None if not set)
    x_range = [x_min, x_max] if x_min is not None and x_max is not None else None
    y_range = [y_min, y_max] if y_min is not None and y_max is not None else None

    # Chart rendering
    st.header("Visualization")

    # Prepare data for plotting
    plot_df = df
    was_sampled = False
    bar_truncated = False
    x_filtered = False
    # For frequency charts, ignore color if same as x_col
    freq_color = color_col if color_col and color_col != x_col else None

    # Apply X value filter
    if x_filter_value is not None and x_col:
        plot_df = df[df[x_col] == x_filter_value]
        x_filtered = True

    # Sample for scatter/line if needed
    if chart_type in ["Scatter", "Line"] and len(plot_df) > 50000:
        plot_df, was_sampled = sample_data(df, max_points=50000)

    # Frequency bar chart - count occurrences
    if chart_type == "Bar (frequency)":
        if freq_color is None:
            plot_df = plot_df[x_col].value_counts().reset_index()
            plot_df.columns = [x_col, "Frequency"]
        else:
            plot_df = plot_df.groupby([x_col, freq_color]).size().reset_index(name="Frequency")

        max_bars = 50
        if plot_df[x_col].nunique() > max_bars:
            top_categories = plot_df.groupby(x_col)["Frequency"].sum().nlargest(max_bars).index
            plot_df = plot_df[plot_df[x_col].isin(top_categories)]
            bar_truncated = True

    if x_filtered:
        st.info(f"Filtered to {x_col} = {x_filter_value} ({len(plot_df):,} rows)")
    if was_sampled:
        st.warning(f"Data sampled to 50,000 points for performance (original: {len(df):,} rows)")
    if bar_truncated:
        st.warning(f"Showing top 50 categories by frequency")

    # Create chart
    try:
        if chart_type == "Correlation Matrix":
            if corr_cols and len(corr_cols) >= 2:
                corr_matrix = df[corr_cols].corr()
                # Create labels with descriptions
                corr_labels = [get_var_description(c) for c in corr_cols]
                fig = px.imshow(
                    corr_matrix,
                    x=corr_labels,
                    y=corr_labels,
                    color_continuous_scale="RdBu_r",
                    zmin=-1,
                    zmax=1,
                    title="Correlation Matrix",
                    text_auto=".2f",
                )
                fig.update_layout(height=600)
                st.plotly_chart(fig, use_container_width=True)

                # Export button for correlation matrix
                png_bytes = create_correlation_heatmap(corr_matrix, corr_labels)
                st.download_button(
                    label="Export as PNG (Matplotlib)",
                    data=png_bytes,
                    file_name="correlation_matrix.png",
                    mime="image/png",
                )
            else:
                st.info("Select at least 2 numeric variables to show correlation matrix.")
        else:
            x_label = get_var_description(x_col) if x_col else ""
            y_label = get_var_description(y_col) if y_col else "Frequency"

            if chart_type == "Scatter":
                fig = px.scatter(
                    plot_df,
                    x=x_col,
                    y=y_col,
                    color=color_col,
                    title=f"{y_label} vs {x_label}",
                    labels={x_col: x_label, y_col: y_label},
                )
            elif chart_type == "Line":
                fig = px.line(
                    plot_df.sort_values(x_col),
                    x=x_col,
                    y=y_col,
                    color=color_col,
                    title=f"{y_label} vs {x_label}",
                    labels={x_col: x_label, y_col: y_label},
                )
            else:  # Bar (frequency)
                fig = px.bar(
                    plot_df,
                    x=x_col,
                    y="Frequency",
                    color=freq_color,
                    title=f"Frequency of {x_label}",
                    labels={x_col: x_label, "Frequency": "Frequency"},
                )

            fig.update_layout(height=500)
            if x_range:
                fig.update_xaxes(range=x_range)
            if y_range:
                fig.update_yaxes(range=y_range)
            st.plotly_chart(fig, use_container_width=True)

            # Export button
            export_color = freq_color if chart_type == "Bar (frequency)" else color_col
            png_bytes = create_matplotlib_figure(
                plot_df,
                chart_type,
                x_col,
                y_col if y_col else "Frequency",
                export_color,
                x_label=x_label,
                y_label=y_label,
                x_range=x_range,
                y_range=y_range,
            )
            st.download_button(
                label="Export as PNG (Matplotlib)",
                data=png_bytes,
                file_name=f"{chart_type.lower().replace(' ', '_')}_{x_col}.png",
                mime="image/png",
            )

    except Exception as e:
        st.error(f"Could not create chart: {e}")

else:
    st.info("Upload a CSV file to get started.")
