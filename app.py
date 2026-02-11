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
        max_upload_size=1000,
    )

with col_dict:
    dict_file = st.file_uploader(
        "Upload a dictionary file (optional)",
        type=["csv"],
        help="CSV with variable names and descriptions",
        key="dict_uploader",
    )

# Early check for data availability (used to conditionally show merge section)
has_data = uploaded_file is not None or "df" in st.session_state

# Merge datasets section (only visible when no data is loaded)
if not has_data:
    with st.expander("Create merged dataset"):
        merge_col1, merge_col2 = st.columns(2)
        with merge_col1:
            merge_file_1 = st.file_uploader("First CSV", type=["csv"], key="merge_file_1")
        with merge_col2:
            merge_file_2 = st.file_uploader("Second CSV", type=["csv"], key="merge_file_2")

        if merge_file_1 is not None and merge_file_2 is not None:
            # Read only headers
            headers_1 = pd.read_csv(merge_file_1, nrows=0).columns.tolist()
            merge_file_1.seek(0)
            headers_2 = pd.read_csv(merge_file_2, nrows=0).columns.tolist()
            merge_file_2.seek(0)

            # Step 1: Select columns from each dataset
            st.markdown("**Select columns to include from each dataset:**")
            sel_col1, sel_col2 = st.columns(2)
            with sel_col1:
                selected_1 = st.multiselect(
                    "Columns from File 1",
                    headers_1,
                    default=headers_1,
                    key="merge_sel_1",
                )
            with sel_col2:
                selected_2 = st.multiselect(
                    "Columns from File 2",
                    headers_2,
                    default=headers_2,
                    key="merge_sel_2",
                )

            # Step 2: Pick key columns from each selection
            if selected_1 and selected_2:
                shared_selected = sorted(set(selected_1) & set(selected_2))
                if shared_selected:
                    merge_keys = st.multiselect(
                        "Key columns (must appear in both selections — rows are matched on these)",
                        shared_selected,
                        help="Rows where all key values match will be combined",
                    )

                    if merge_keys:
                        # Non-key columns from each side
                        extra_1 = [c for c in selected_1 if c not in merge_keys]
                        extra_2 = [c for c in selected_2 if c not in merge_keys]
                        st.caption(
                            f"Result will have: **{', '.join(merge_keys)}** (keys)"
                            + (f" + **{', '.join(extra_1)}** (from file 1)" if extra_1 else "")
                            + (f" + **{', '.join(extra_2)}** (from file 2)" if extra_2 else "")
                        )

                        if st.button("Merge datasets", type="primary"):
                            with st.spinner("Merging datasets..."):
                                df1 = load_csv(merge_file_1)[selected_1]
                                df2 = load_csv(merge_file_2)[selected_2]
                                for k in merge_keys:
                                    df1[k] = df1[k].astype(str)
                                    df2[k] = df2[k].astype(str)
                                merged = pd.merge(df1, df2, on=merge_keys, how="inner")
                                st.session_state.df = merged
                                st.session_state.filename = "merged_dataset.csv"
                                st.session_state.restored_from_file = False
                                st.rerun()
                else:
                    st.warning("No shared columns between your selections — select at least one common column to use as a merge key.")

with st.sidebar:
    if "df" in st.session_state:
        csv_bytes = st.session_state.df.to_csv(index=False).encode("utf-8")
        dl_name = st.session_state.get("filename", "data.csv")
        st.download_button(
            label="Download dataset",
            data=csv_bytes,
            file_name=dl_name,
            mime="text/csv",
            key="download_dataset",
        )
        if st.button("Clear loaded data", key="clear_data"):
            for key in ["df", "filename", "restored_from_file", "encoded_cols", "derivative_preview_cols", "data_modified"]:
                st.session_state.pop(key, None)
            st.rerun()
        st.divider()

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
            name_idx = 0
            if "var_name_col" in st.session_state and st.session_state.var_name_col in dict_cols:
                name_idx = dict_cols.index(st.session_state.var_name_col)
            var_name_col = st.selectbox("Variable name column", dict_cols, index=name_idx, key="var_name_col")
        with dcol2:
            desc_idx = min(1, len(dict_cols) - 1)
            if "var_desc_col" in st.session_state and st.session_state.var_desc_col in dict_cols:
                desc_idx = dict_cols.index(st.session_state.var_desc_col)
            var_desc_col = st.selectbox(
                "Description column",
                dict_cols,
                index=desc_idx,
                key="var_desc_col",
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
    # Load data into session state from uploaded file (skip if data was modified e.g. by merge)
    if "df" not in st.session_state or (
        st.session_state.get("filename") != uploaded_file.name
        and not st.session_state.get("data_modified")
    ):
        with st.spinner("Loading data..."):
            st.session_state.df = load_csv(uploaded_file)
            st.session_state.filename = uploaded_file.name
            st.session_state.restored_from_file = False
            st.session_state.data_modified = False

if has_data and "df" in st.session_state:
    df = st.session_state.df

    # Data overview
    st.header("Data Overview")
    if st.session_state.get("restored_from_file"):
        st.caption(f"*Restored from session: {st.session_state.get('filename', 'unknown')}*")
    col1, col2, col3 = st.columns(3)
    col1.metric("Rows", f"{len(df):,}")
    col2.metric("Columns", len(df.columns))
    col3.metric("Memory", f"{df.memory_usage(deep=True).sum() / 1024 / 1024:.1f} MB")

    # Tabs for preview and info
    tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs(["Preview", "Column Info", "Statistics", "Encode Variables", "Create Derivative", "Merge with Dataset", "Manipulate"])

    with tab1:
        n_rows = st.slider("Rows to display", 5, 100, 10)
        # Build column config with tooltips from variable descriptions
        col_config = {}
        for col in df.columns:
            desc = var_descriptions.get(col)
            if desc:
                col_config[col] = st.column_config.Column(help=desc)
        st.dataframe(df.head(n_rows), column_config=col_config, use_container_width=True)

    with tab2:
        st.dataframe(get_column_info(df), use_container_width=True)

    with tab3:
        st.dataframe(df.describe(), use_container_width=True)

    with tab4:
        st.caption("Convert categorical columns to numerical values for analysis")

        # Get categorical and boolean columns
        cat_cols = df.select_dtypes(include=['object', 'category', 'bool']).columns.tolist()

        if cat_cols:
            encode_tab1, encode_tab2 = st.tabs(["One-Hot Encoding", "Label Encoding"])

            with encode_tab1:
                st.markdown("**One-Hot Encoding**: Each unique value becomes a new binary (0/1) column")
                onehot_cols = st.multiselect(
                    "Select columns to one-hot encode",
                    cat_cols,
                    key="onehot_cols",
                )

                if onehot_cols:
                    for col in onehot_cols:
                        unique_vals = df[col].dropna().unique()
                        st.caption(f"**{col}**: {len(unique_vals)} values → creates {len(unique_vals)} new columns")

                    if st.button("Apply One-Hot Encoding", type="primary", key="apply_onehot"):
                        encoded_df = pd.get_dummies(df, columns=onehot_cols, prefix=onehot_cols, dtype=int)
                        st.session_state.df = encoded_df
                        prev_encoded = st.session_state.get("encoded_cols") or []
                        st.session_state.encoded_cols = prev_encoded + onehot_cols
                        st.success(f"Encoded {len(onehot_cols)} column(s). New columns: {len(encoded_df.columns) - len(df.columns)}")
                        st.rerun()

            with encode_tab2:
                st.markdown("**Label Encoding**: Convert categories to numbers (0, 1, 2, ...)")
                label_col = st.selectbox(
                    "Select column to label encode",
                    [""] + cat_cols,
                    key="label_col",
                )

                if label_col:
                    unique_vals = df[label_col].dropna().unique().tolist()
                    st.caption(f"**{len(unique_vals)} unique values found**")

                    # Option for custom ordering
                    use_custom_order = st.checkbox(
                        "Specify custom order (for ordinal data like low/medium/high)",
                        key="custom_order",
                    )

                    if use_custom_order:
                        st.caption("Drag to reorder, or enter comma-separated values:")
                        default_order = ", ".join(str(v) for v in unique_vals)
                        custom_order_str = st.text_input(
                            "Order (first = 0, second = 1, ...)",
                            value=default_order,
                            key="order_input",
                        )
                        try:
                            ordered_vals = [v.strip() for v in custom_order_str.split(",")]
                            # Show preview
                            st.caption("Preview: " + ", ".join(f"{v}={i}" for i, v in enumerate(ordered_vals)))
                        except:
                            ordered_vals = unique_vals
                    else:
                        ordered_vals = sorted(unique_vals, key=str)
                        st.caption("Alphabetical order: " + ", ".join(f"{v}={i}" for i, v in enumerate(ordered_vals)))

                    new_col_name = st.text_input(
                        "New column name",
                        value=f"{label_col}_encoded",
                        key="new_col_name",
                    )

                    if st.button("Apply Label Encoding", type="primary", key="apply_label"):
                        value_map = {v: i for i, v in enumerate(ordered_vals)}
                        df[new_col_name] = df[label_col].map(value_map)
                        st.session_state.df = df
                        prev_encoded = st.session_state.get("encoded_cols") or []
                        st.session_state.encoded_cols = prev_encoded + [new_col_name]
                        st.success(f"Created column '{new_col_name}' with values 0-{len(ordered_vals)-1}")
                        st.rerun()

            # Show reset option if any encoding was applied
            if st.session_state.get("encoded_cols"):
                st.divider()
                st.caption(f"Encoded columns: {', '.join(st.session_state.encoded_cols)}")
                if uploaded_file is not None:
                    if st.button("Reset to Original Data", key="reset_encoding"):
                        st.session_state.df = load_csv(uploaded_file)
                        st.session_state.encoded_cols = None
                        st.rerun()
                else:
                    st.caption("(Reset unavailable for restored sessions)")
        else:
            st.info("No categorical or boolean columns found to encode.")

    with tab5:
        st.caption("Create a new dataset with selected columns from the current data")

        all_columns = df.columns.tolist()

        # Initialize preview columns
        if "derivative_preview_cols" not in st.session_state:
            st.session_state.derivative_preview_cols = all_columns

        # Form prevents rerun until submitted
        with st.form("derivative_form"):
            valid_default = [c for c in st.session_state.derivative_preview_cols if c in all_columns]
            if not valid_default:
                valid_default = all_columns

            selected_cols = st.multiselect(
                "Select columns to include",
                all_columns,
                default=valid_default,
                help="Choose which columns to include in the derivative dataset",
            )

            submitted = st.form_submit_button("Update preview")
            if submitted:
                st.session_state.derivative_preview_cols = selected_cols

        # Use stored preview columns for display
        preview_cols = [c for c in st.session_state.derivative_preview_cols if c in all_columns]

        if preview_cols:
            derivative_df = df[preview_cols]
            st.caption(f"**Preview**: {len(derivative_df):,} rows × {len(preview_cols)} columns")
            st.dataframe(derivative_df.head(10), use_container_width=True)

            col_use, col_download = st.columns(2)

            with col_use:
                if st.button("Use as current dataset", type="primary", key="use_derivative"):
                    st.session_state.df = derivative_df
                    st.session_state.encoded_cols = None
                    st.success(f"Dataset updated to {len(preview_cols)} columns")
                    st.rerun()

            with col_download:
                csv_bytes = derivative_df.to_csv(index=False).encode("utf-8")
                base_name = st.session_state.get("filename", "data.csv").rsplit(".", 1)[0]
                st.download_button(
                    label="Download as CSV",
                    data=csv_bytes,
                    file_name=f"{base_name}_derivative.csv",
                    mime="text/csv",
                    key="download_derivative",
                )
        else:
            st.warning("Select at least one column")

    with tab6:
        st.caption("Merge the current dataset with another CSV file")

        merge_upload = st.file_uploader("Upload CSV to merge with", type=["csv"], key="merge_tab_file")

        if merge_upload is not None:
            merge_headers = pd.read_csv(merge_upload, nrows=0).columns.tolist()
            merge_upload.seek(0)
            current_cols = df.columns.tolist()

            # Select columns from current dataset
            mcol1, mcol2 = st.columns(2)
            with mcol1:
                sel_current = st.multiselect(
                    "Columns from current dataset",
                    current_cols,
                    default=current_cols,
                    key="merge_tab_sel_current",
                )
            with mcol2:
                sel_other = st.multiselect(
                    "Columns from uploaded file",
                    merge_headers,
                    default=merge_headers,
                    key="merge_tab_sel_other",
                )

            if sel_current and sel_other:
                shared = sorted(set(sel_current) & set(sel_other))
                if shared:
                    tab_merge_keys = st.multiselect(
                        "Key columns (must appear in both selections — rows are matched on these)",
                        shared,
                        key="merge_tab_keys",
                    )

                    if tab_merge_keys:
                        extra_cur = [c for c in sel_current if c not in tab_merge_keys]
                        extra_oth = [c for c in sel_other if c not in tab_merge_keys]
                        st.caption(
                            f"Result will have: **{', '.join(tab_merge_keys)}** (keys)"
                            + (f" + **{', '.join(extra_cur)}** (current)" if extra_cur else "")
                            + (f" + **{', '.join(extra_oth)}** (uploaded)" if extra_oth else "")
                        )

                        if st.button("Merge datasets", type="primary", key="merge_tab_btn"):
                            with st.spinner("Merging datasets..."):
                                df_current = df[sel_current].copy()
                                df_other = load_csv(merge_upload)[sel_other]
                                for k in tab_merge_keys:
                                    df_current[k] = df_current[k].astype(str)
                                    df_other[k] = df_other[k].astype(str)
                                merged = pd.merge(df_current, df_other, on=tab_merge_keys, how="inner")
                                st.session_state.df = merged
                                st.session_state.filename = "merged_dataset.csv"
                                st.session_state.encoded_cols = None
                                st.session_state.data_modified = True
                                st.rerun()
                else:
                    st.warning("No shared columns between your selections — select at least one common column to use as a merge key.")

    with tab7:
        manip_remove, manip_convert = st.tabs(["Remove Rows", "Convert Types"])

        with manip_remove:
            st.caption("Remove all rows where a variable equals a specific value")

            manip_col = st.selectbox("Select variable", df.columns.tolist(), key="manip_col")

            if manip_col:
                unique_vals = df[manip_col].dropna().unique().tolist()
                manip_value = st.text_input(
                    f"Remove rows where **{manip_col}** equals:",
                    key="manip_value",
                    help=f"Column has {len(unique_vals)} unique values. Examples: {', '.join(str(v) for v in unique_vals[:5])}",
                )

                if manip_value:
                    col_dtype = df[manip_col].dtype
                    try:
                        typed_value = col_dtype.type(manip_value)
                    except (ValueError, TypeError):
                        typed_value = manip_value
                    mask = (df[manip_col] == manip_value) | (df[manip_col] == typed_value)
                    match_count = mask.sum()

                    st.caption(f"**{match_count:,}** rows match (out of {len(df):,})")

                    if match_count > 0 and st.button("Remove matching rows", type="primary", key="manip_apply"):
                        st.session_state.df = df[~mask].reset_index(drop=True)
                        st.rerun()

        with manip_convert:
            st.caption("Convert column data types (e.g. object → int, float → int)")

            # Show current types
            type_info = pd.DataFrame({
                "Column": df.columns,
                "Current Type": [str(df[c].dtype) for c in df.columns],
                "Sample Value": [str(df[c].dropna().iloc[0]) if df[c].notna().any() else "N/A" for c in df.columns],
            })
            st.dataframe(type_info, use_container_width=True, hide_index=True)

            conv_col = st.selectbox("Column to convert", df.columns.tolist(), key="conv_col")

            if conv_col:
                st.caption(f"**{conv_col}** is currently `{df[conv_col].dtype}`")
                target_type = st.selectbox(
                    "Convert to",
                    ["int", "float", "str"],
                    key="conv_target",
                )

                # Preview conversion
                errors_option = "coerce" if target_type in ["int", "float"] else None
                try:
                    if target_type == "int":
                        preview = pd.to_numeric(df[conv_col], errors="coerce")
                        failed = preview.isna().sum() - df[conv_col].isna().sum()
                        preview = preview.astype("Int64")
                    elif target_type == "float":
                        preview = pd.to_numeric(df[conv_col], errors="coerce")
                        failed = preview.isna().sum() - df[conv_col].isna().sum()
                    else:
                        preview = df[conv_col].astype(str)
                        failed = 0

                    if failed > 0:
                        st.warning(f"**{failed:,}** values can't be converted and will become NaN")
                        # Show the unconvertible values
                        failed_mask = pd.to_numeric(df[conv_col], errors="coerce").isna() & df[conv_col].notna()
                        failed_values = df.loc[failed_mask, conv_col].unique()
                        st.caption(
                            f"Values that can't be parsed as {target_type}: "
                            + ", ".join(f"`{v}`" for v in failed_values[:10])
                            + (f" ... and {len(failed_values) - 10} more" if len(failed_values) > 10 else "")
                        )

                    if st.button("Convert", type="primary", key="conv_apply"):
                        st.session_state.df[conv_col] = preview
                        st.rerun()
                except Exception as e:
                    st.error(f"Conversion failed: {e}")

    # Chart configuration in sidebar
    st.sidebar.header("Chart Configuration")

    # Refresh column lists after potential encoding
    numeric_cols = get_numeric_columns(df)
    categorical_cols = get_categorical_columns(df)
    all_cols = df.columns.tolist()

    chart_types = ["Scatter", "Line", "Bar (frequency)", "Distribution", "Correlation Matrix"]
    chart_type = st.sidebar.selectbox(
        "Chart Type",
        chart_types,
        key="chart_type",
    )

    # Variable selection depends on chart type
    x_col = None
    y_col = None
    color_col = None
    corr_cols = None

    if chart_type == "Correlation Matrix":
        # Multi-select for correlation matrix
        default_corr = st.session_state.get("corr_cols")
        if default_corr is None:
            default_corr = numeric_cols[:10] if len(numeric_cols) > 10 else numeric_cols
        # Filter to valid columns only
        default_corr = [c for c in default_corr if c in numeric_cols]
        corr_cols = st.sidebar.multiselect(
            "Variables (numeric only)",
            numeric_cols,
            default=default_corr,
            format_func=format_var_label,
            key="corr_cols",
        )
        if len(corr_cols) < 2:
            st.sidebar.warning("Select at least 2 variables")
    else:
        # Compute default index for x_col from session state
        x_default_idx = 0
        if "x_col" in st.session_state and st.session_state.x_col in all_cols:
            x_default_idx = all_cols.index(st.session_state.x_col)
        x_col = st.sidebar.selectbox(
            "X Axis" if chart_type not in ["Bar (frequency)", "Distribution"] else "Variable",
            all_cols,
            index=x_default_idx,
            format_func=format_var_label,
            help=get_var_description(all_cols[0]) if all_cols else None,
            key="x_col",
        )
        if x_col:
            st.sidebar.caption(f"*{get_var_description(x_col)}*")

        # Y axis only for non-frequency charts
        if chart_type not in ["Bar (frequency)", "Distribution"]:
            y_default_idx = min(1, len(all_cols) - 1)
            if "y_col" in st.session_state and st.session_state.y_col in all_cols:
                y_default_idx = all_cols.index(st.session_state.y_col)
            y_col = st.sidebar.selectbox(
                "Y Axis",
                all_cols,
                index=y_default_idx,
                format_func=format_var_label,
                key="y_col",
            )
            if y_col:
                st.sidebar.caption(f"*{get_var_description(y_col)}*")

        # Color by (optional)
        color_options = ["None"] + all_cols
        color_default_idx = 0
        if "color_col" in st.session_state:
            color_val = st.session_state.color_col
            if color_val in color_options:
                color_default_idx = color_options.index(color_val)
        color_col = st.sidebar.selectbox(
            "Color By",
            color_options,
            index=color_default_idx,
            format_func=lambda x: format_var_label(x) if x != "None" else "None",
            key="color_col",
        )
        color_col = None if color_col == "None" else color_col
        if color_col:
            st.sidebar.caption(f"*{get_var_description(color_col)}*")

    # Filter to single X value
    x_filter_value = None
    if chart_type not in ["Correlation Matrix", "Bar (frequency)", "Distribution"] and x_col:
        with st.sidebar.expander("Filter by X Value"):
            enable_x_filter = st.checkbox("Filter to single X value", value=False, key="enable_x_filter")
            if enable_x_filter:
                unique_x = sorted(df[x_col].dropna().unique().tolist())
                if len(unique_x) > 0:
                    x_filter_value = st.selectbox(
                        f"Select {x_col} value",
                        unique_x,
                        format_func=lambda v: str(v),
                        key="x_filter_value",
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
            elif chart_type == "Distribution":
                fig = px.histogram(
                    plot_df,
                    x=x_col,
                    color=color_col,
                    title=f"Distribution of {x_label}",
                    labels={x_col: x_label},
                    marginal="box",
                )
            else:  # Bar (frequency)
                fig = px.bar(
                    plot_df,
                    x=x_col,
                    y="Frequency",
                    color=freq_color,
                    title=f"Frequency of {x_label}" + (f" by {get_var_description(freq_color)}" if freq_color else ""),
                    labels={x_col: x_label, "Frequency": "Count"},
                    barmode="stack",
                )
                fig.update_traces(
                    hovertemplate=f"<b>{x_label}</b>: %{{x}}<br><b>Count</b>: %{{y:,}}<extra>%{{fullData.name}}</extra>"
                )

            fig.update_layout(height=500)
            if x_range:
                fig.update_xaxes(range=x_range)
            if y_range:
                fig.update_yaxes(range=y_range)
            st.plotly_chart(fig, use_container_width=True)

            # Show frequency breakdown table for bar charts with color
            if chart_type == "Bar (frequency)" and freq_color:
                with st.expander("Frequency Breakdown Table", expanded=False):
                    pivot_table = plot_df.pivot_table(
                        index=x_col,
                        columns=freq_color,
                        values="Frequency",
                        aggfunc="sum",
                        fill_value=0,
                    )
                    pivot_table["Total"] = pivot_table.sum(axis=1)
                    pivot_table.loc["Total"] = pivot_table.sum()
                    st.dataframe(pivot_table, use_container_width=True)

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
    st.info("Upload a CSV file or restore a saved session to get started.")
