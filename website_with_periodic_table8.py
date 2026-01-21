import pandas as pd
import streamlit as st
import numpy as np
from pymcdm.methods import PROMETHEE_II
from pymcdm.methods import TOPSIS
from io import BytesIO
import plotly.graph_objects as go
import plotly.express as px
import re
import matplotlib.pyplot as plt
import random

# Custom CSS for styling
def set_custom_style():
    st.markdown("""
    <style>
        /* Lighten sidebar background */
        [data-testid="stSidebar"] {
            background-color: #f8f9fa !important;
            border-right: 1px solid #e0e0e0;
        }
        
        /* Dark text for light sidebar */
        [data-testid="stSidebar"] .stRadio > label,
        [data-testid="stSidebar"] p,
        [data-testid="stSidebar"] .stMarkdown {
            color: #333333 !important;
        }
        
        /* Sidebar hover effects */
        [data-testid="stSidebar"] .stRadio > div:hover {
            background-color: #e9ecef;
            border-radius: 5px;
        }
    </style>
    """, unsafe_allow_html=True)

@st.cache_data
def load_composition_database():
    df1 = pd.read_excel("8_material_properties_cleaned.xlsx")
    return df1

@st.cache_data
def load_high_confidence_database():
    df2 = pd.read_excel("materials_high_confidence_cleaned.xlsx")
    return df2.iloc[:, 1:]

def filter_dataframe(df, filters, selected_names=None):
    """Filter dataframe based on provided filters and optional names"""
    filtered = df.copy()
    
    # Apply each filter dynamically
    for filter_name, filter_range in filters.items():
        if filter_name in df.columns:
            filtered = filtered[
                filtered[filter_name].between(filter_range[0], filter_range[1], inclusive='both')
            ]
    
    if selected_names is not None:
        filtered = filtered[filtered["Name"].isin(selected_names)]
    
    return filtered

def run_topsis(matrix, weights, criteria_types):
    topsis = TOPSIS()
    return topsis(matrix, weights, criteria_types)

def run_promethee(matrix, weights, criteria_types):
    promethee = PROMETHEE_II('usual')
    return promethee(matrix, weights, criteria_types)

@st.cache_data
def prepare_plot_data(df, x_col, y_col, log_x=False, log_y=False):
    df_plot = df.copy()
    if log_x:
        df_plot[x_col] = np.log10(df_plot[x_col].clip(lower=1e-10))
    if log_y:
        df_plot[y_col] = np.log10(df_plot[y_col].clip(lower=1e-10))
    return df_plot

@st.cache_data
def create_full_output(filtered_df, results_df, weights_df):
    """Create Excel output with all MCDM analysis results"""
    output = BytesIO()
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        # Prepare full data with MCDM results
        full_data = filtered_df.copy()
        
        # Reset index of results_df to get Rank as a column
        results_reset = results_df.reset_index()
        
        # Merge MCDM scores/ranks with full data
        if 'Score' in results_reset.columns:
            # TOPSIS results
            score_map = dict(zip(results_reset['Material'], results_reset['Score']))
            rank_map = dict(zip(results_reset['Material'], results_reset['Rank']))
            full_data['TOPSIS_Score'] = full_data['Name'].map(score_map)
            full_data['TOPSIS_Rank'] = full_data['Name'].map(rank_map)
        else:
            # PROMETHEE results
            flow_map = dict(zip(results_reset['Material'], results_reset['Net Flow']))
            rank_map = dict(zip(results_reset['Material'], results_reset['Rank']))
            full_data['PROMETHEE_Net_Flow'] = full_data['Name'].map(flow_map)
            full_data['PROMETHEE_Rank'] = full_data['Name'].map(rank_map)
        
        # Write sheets
        full_data.to_excel(writer, sheet_name='Full Data', index=False)
        results_reset.to_excel(writer, sheet_name='Rankings', index=False)
        weights_df.reset_index().to_excel(writer, sheet_name='Weights', index=False)
        
        # Filter settings
        if 'filters' in st.session_state and st.session_state.filters:
            filter_settings = pd.DataFrame([
                {'Filter': k, 'Min': v[0], 'Max': v[1]} 
                for k, v in st.session_state.filters.items()
            ])
            filter_settings.to_excel(writer, sheet_name='Filter Settings', index=False)
    
    return output.getvalue()

def create_professional_plot(df, x_col, y_col, title, x_label, y_label, log_x=False, log_y=False):
    """Create a professional Plotly scatter plot"""
    df_plot = df.copy()
    
    # Professional colors
    primary_color = "#3498db"
    highlight_color = "#c5301f"
    
    # Handle negative/zero values for log scales
    if log_x:
        df_plot[x_col] = df_plot[x_col].clip(lower=1e-10)
    if log_y:
        df_plot[y_col] = df_plot[y_col].clip(lower=1e-10)
    
    # Highlight 10 random materials
    num_highlight = min(10, len(df_plot))
    highlight_indices = np.random.choice(len(df_plot), num_highlight, replace=False)
    df_plot['is_highlight'] = False
    df_plot.iloc[highlight_indices, df_plot.columns.get_loc('is_highlight')] = True
    
    # Create figure
    fig = go.Figure()
    
    # Add non-highlighted points
    df_regular = df_plot[~df_plot['is_highlight']]
    fig.add_trace(go.Scatter(
        x=df_regular[x_col],
        y=df_regular[y_col],
        mode='markers',
        name='All Materials',
        marker=dict(
            size=8,
            color=primary_color,
            opacity=0.6,
            line=dict(width=0)
        ),
        text=df_regular['Name'],
        hovertemplate='<b>%{text}</b><br>' + x_label + ': %{x}<br>' + y_label + ': %{y}<extra></extra>'
    ))
    
    # Add highlighted points
    df_highlight = df_plot[df_plot['is_highlight']]
    fig.add_trace(go.Scatter(
        x=df_highlight[x_col],
        y=df_highlight[y_col],
        mode='markers+text',
        name='Highlighted Materials',
        marker=dict(
            size=12,
            color=highlight_color,
            opacity=1.0,
            line=dict(width=0)
        ),
        text=df_highlight['Name'],
        textposition='top center',
        textfont=dict(color=highlight_color, size=10),
        hovertemplate='<b>%{text}</b><br>' + x_label + ': %{x}<br>' + y_label + ': %{y}<extra></extra>'
    ))
    
    # Update layout
    fig.update_layout(
        title=title,
        xaxis_title=f"{'log(' + x_label + ')' if log_x else x_label}",
        yaxis_title=f"{'log(' + y_label + ')' if log_y else y_label}",
        xaxis_type="log" if log_x else "linear",
        yaxis_type="log" if log_y else "linear",
        hovermode='closest',
        template='plotly_white',
        width=900,
        height=550,
        showlegend=True,
        legend=dict(x=0.99, y=0.99, bgcolor='rgba(255,255,255,0.7)')
    )
    
    return fig

def format_tons(value):
    """Format large numbers with appropriate units (M/B)"""
    if value >= 1_000_000_000_000:
        return f"{value / 1_000_000_000_000:.1f}TB tons"
    elif value >= 1_000_000_000:
        return f"{value / 1_000_000_000:.1f}B tons"
    elif value >= 1_000_000:
        return f"{value / 1_000_000:.1f}M tons"
    elif value >= 1_000:
        return f"{value / 1_000:.1f}K tons"
    else:
        return f"{value:.0f} tons"

def filter_by_excluded_elements(df, excluded_elements):
    """
    Filter out materials that contain any of the excluded elements.
    """
    if not excluded_elements:
        return df.copy()
    
    element_columns = ['Element_1', 'Element_2', 'Element_3', 'Element_4', 
                       'Element_5', 'Element_6', 'Element_7']
    
    mask = pd.Series([True] * len(df), index=df.index)
    excluded_elements_normalized = [str(elem).strip() for elem in excluded_elements]
    
    for col in element_columns:
        if col in df.columns:
            column_values = df[col].fillna('').astype(str).str.strip()
            mask &= ~column_values.isin(excluded_elements_normalized)
    
    return df[mask].copy()

def main():
    set_custom_style()
    df1 = load_composition_database()
    df2 = load_high_confidence_database()

    # Sidebar navigation
    st.sidebar.title("Material Analysis")
    st.sidebar.markdown("---")
    selected_page = st.sidebar.radio(
        "Navigation Menu", 
        ["Home", "Bandgap Information", "Decision-making Assistant"],
        captions=["Welcome page", "Commonly researched semiconductors", "Multi-criteria decision making tool"]
    )
    
    # Add footer
    st.markdown("""
    <div class="footer">
        Semiconductor Database © 2025 | v3.0 | Developed by HERAWS
    </div>
    """, unsafe_allow_html=True)

    if selected_page == "Home":
        st.title("Semiconductor Database")
        
        cols = st.columns(2)
        with cols[0]:
            st.markdown("""
            ### 🔍 About This Tool
            This interactive platform enables comprehensive analysis of environmental impacts and sustainability of semiconductors with:
            - **Extensive database** on ESG scores, CO₂ footprints, and more
            - **Visualizations** to explore relationships between parameters
            - **Multi-criteria** decision making tools (TOPSIS, PROMETHEE)
            - **Export capabilities** for further analysis
            """)
            
        with cols[1]:
            st.markdown("""
            ### 🚀 Getting Started
            1. Select an analysis page from the sidebar
            2. Configure your filters and parameters
            3. Visualize the relationships
            4. Download results for further use
            
            **Pro Tip:** Use the MCDM analysis for ranking the most promising semiconductors.
            """)
        
        st.markdown("---")
        
        st.markdown("### 📚 Database Information")
        cols = st.columns(2)
        with cols[0]:
            st.metric("Total Materials", len(df1))
            prod_min = df1['Production (ton)'].min()
            prod_max = df1['Production (ton)'].max()
            st.metric("Production Range", f"{format_tons(prod_min)} - {format_tons(prod_max)}")
        with cols[1]:
            st.metric("Bandgap Range", f"{df1['Bandgap'].min():.1f} - {df1['Bandgap'].max():.1f} eV")

        
    elif selected_page == "Bandgap Information":
        st.title("Bandgap Information")
        st.markdown("Most commonly researched semiconductors and their band gap range.")

        if "included_elements" not in st.session_state:
            st.session_state.included_elements = []
        if "filters_applied" not in st.session_state:
            st.session_state.filters_applied = False

        # ELEMENT INCLUSION SECTION
        st.markdown("### Element Inclusion")
        
        periodic_table_url = "https://upload.wikimedia.org/wikipedia/commons/thumb/2/2e/Simple_Periodic_Table_Chart-en.svg/1200px-Simple_Periodic_Table_Chart-en.svg.png"
        st.image(periodic_table_url, caption="Periodic Table of Elements", width='stretch')

        element_cols_df2 = df2.columns[3:-2].tolist()
        element_cols_df1 = [f'Element_{i}' for i in range(1, 8)]
        
        df = df1.copy()
        
        all_elements = set(element_cols_df2)
        
        for col in element_cols_df1:
            if col in df.columns:
                elements = df[col].dropna().unique()
                all_elements.update(elements)
        
        all_elements = {elem for elem in all_elements if elem and str(elem).strip()}
        all_elements = sorted(list(all_elements))
        
        def filter_df1_by_included_elements(dataframe, included_list, element_cols):
            if not included_list:
                return dataframe.iloc[0:0].copy()
            
            if not element_cols:
                return dataframe.copy()
            
            def _is_empty(x):
                return pd.isna(x) or str(x).strip() == ""
            
            mask = pd.Series(True, index=dataframe.index)
            included_set = set(included_list)
            
            for c in element_cols:
                if c in dataframe.columns:
                    mask &= dataframe[c].isin(included_set) | dataframe[c].apply(_is_empty)
            
            return dataframe[mask].copy()
        
        def filter_df2_by_included_elements(dataframe, included_list, element_cols):
            if not included_list:
                return dataframe.iloc[0:0].copy()
            
            if not element_cols:
                return dataframe.copy()
            
            included_set = set(included_list)
            mask = pd.Series(True, index=dataframe.index)
            
            for elem_col in element_cols:
                if elem_col in dataframe.columns:
                    if elem_col not in included_set:
                        mask &= (dataframe[elem_col] == 0)
            
            return dataframe[mask].copy()
        
        df1_filtered = filter_df1_by_included_elements(df, st.session_state.included_elements, element_cols_df1)
        df2_filtered_preview = filter_df2_by_included_elements(df2, st.session_state.included_elements, element_cols_df2)
        
        if st.session_state.included_elements:
            removed_count_df1 = len(df) - len(df1_filtered)
            st.info(f"🔬 Element filter active: Included {', '.join(sorted(st.session_state.included_elements))} | "
                    f"Showing {len(df1_filtered)} of {len(df)} materials")
        
        st.markdown("**Enter element symbols to include (separated by commas)**")

        element_text_input = st.text_input(
            "Element symbols:",
            value=", ".join(st.session_state.included_elements) if st.session_state.included_elements else "",
            key="element_text_input",
            placeholder="e.g., Ti, O, Zn",
            help="Enter element symbols separated by commas."
        )
        
        if element_text_input.strip():
            selected_elements = [elem.strip() for elem in element_text_input.split(',') if elem.strip()]
        else:
            selected_elements = []
        
        if st.session_state.included_elements:
            st.markdown(f"**Currently Included Elements:** {len(st.session_state.included_elements)}")
        
        if selected_elements != st.session_state.included_elements:
            preview_df1 = filter_df1_by_included_elements(df, selected_elements, element_cols_df1)
            
            if len(preview_df1) > 0:
                st.success(f"✅ Preview: {len(preview_df1)} materials ({len(preview_df1)/len(df)*100:.1f}%)")
            else:
                st.warning(f"⚠️ Preview: No materials contain only these elements.")
        
        st.session_state.included_elements = selected_elements
        
        colA, colB = st.columns([1, 3])
        with colA:
            apply_clicked = st.button("Apply Filters", key="apply_initial_filters")
            if apply_clicked:
                st.session_state.filters_applied = True
                st.rerun()
        
        df_filtered = df1_filtered.copy()

        bandgap_col = None
        possible_names = ['Bandgap', 'bandgap', 'Band_gap', 'band_gap', 'Value', 'value', 'BandGap']
        for col_name in possible_names:
            if col_name in df_filtered.columns:
                bandgap_col = col_name
                break

        def pick_palette(n: int):
            colors = px.colors.qualitative.Set3 if n <= 12 else px.colors.qualitative.Light24
            return colors[:n]

        df_agg = (
            df_filtered.groupby("Name")
            .size()
            .reset_index(name="Count")
            .sort_values(by="Count", ascending=False)
        ).head(9)

        st.session_state.df_agg = df_agg

        top_names = df_agg["Name"].head(9).tolist()
        st.markdown(
            """
            <h3 style='
                font-size:20px;
                font-weight:600; 
                color:#222;
                margin-top:10px;
                margin-bottom:20px;
                font-family:Arial;
            '>
            Exploratory Data Analysis of Database
            </h3>
            """, 
            unsafe_allow_html=True
        )
        st.markdown("***The scatterplot provides a broad visual overview of the relative bandgap ranges across the materials.***")

        if bandgap_col is None or not top_names:
            st.warning(f"⚠️ No bandgap column found or no top names available.")
        else:
            filtered_df_top = df_filtered[df_filtered["Name"].isin(top_names)].copy()

            if filtered_df_top.empty:
                st.info("No data to display for the current selection of elements.")
            else:
                palette = pick_palette(len(top_names))
                
                fig = px.scatter(
                    filtered_df_top,
                    x=bandgap_col,
                    y='Name',
                    color='Name',
                    color_discrete_sequence=palette,
                    title='Bandgap Distribution by Semiconductor',
                    labels={bandgap_col: 'Bandgap (eV)', 'Name': 'Semiconductor'},
                    height=500,
                    hover_data={bandgap_col: ':.2f'}
                )
                
                fig.update_traces(marker=dict(size=10, opacity=0.9))
                fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='LightGray')
                fig.update_yaxes(showgrid=False)
                fig.update_layout(template='plotly_white', hovermode='closest')
                
                st.plotly_chart(fig, use_container_width=True, key="bandgap_scatter")

        st.markdown("***Histogram plot shows the frequency distribution of bandgaps.***")

        if bandgap_col is None:
            st.warning(f"⚠️ No bandgap column found.")
        else:
            df_hist = df_filtered[df_filtered["Name"].isin(top_names)].copy()
            df_hist = df_hist[pd.to_numeric(df_hist[bandgap_col], errors="coerce").notna()]
            df_hist[bandgap_col] = df_hist[bandgap_col].astype(float)

            if df_hist.empty:
                st.info("No data to plot after filtering.")
            else:
                x_min, x_max = df_hist[bandgap_col].min(), 10
                top9 = top_names[:9]

                fig, axes = plt.subplots(nrows=3, ncols=3, figsize=(12, 10), sharex=True)
                axes = axes.flatten()

                for i, name in enumerate(top9):
                    ax = axes[i]
                    sub = df_hist.loc[df_hist["Name"] == name, bandgap_col].dropna().values

                    if sub.size >= 1:
                        ax.hist(sub, bins='auto', density=False, alpha=0.85)
                        if sub.size >= 2:
                            ax.axvline(np.median(sub), linestyle="--", linewidth=1)
                        ax.set_yscale('log')
                    else:
                        ax.text(0.5, 0.5, "No data", ha="center", va="center",
                                transform=ax.transAxes, fontsize=9, alpha=0.7)

                    ax.set_title(f"{name} (n={sub.size})", fontsize=11)
                    ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.4)
                    ax.set_xlim(x_min, x_max)
                    if i % 3 != 0: ax.set_ylabel("")
                    if i < 6: ax.set_xlabel("")

                for j in range(len(top9), 9):
                    axes[j].axis("off")

                fig.supylabel("Count (log scale)")
                fig.supxlabel("Bandgap (eV)")
                fig.tight_layout(rect=[0, 0.02, 1, 0.95])

                st.pyplot(fig, clear_figure=True)

        st.markdown("***The scatter plot visualizes the trends of recently researched materials and their corresponding bandgap values.***")

        unique_list = df_agg['Name'].dropna().unique().tolist()

        if 'Name' not in df2.columns:
            st.warning("Missing required column: Name")
            st.stop()

        df2_set = df2.copy()
        df2_set['Date'] = pd.to_datetime(df2_set['Date'], errors='coerce')

        min_date = df2_set['Date'].min().date()
        max_date = df2_set['Date'].max().date()

        start_date, end_date = st.date_input(
            "Select start and end date",
            value=(min_date, max_date),
            min_value=min_date,
            max_value=max_date
        )

        mask = (df2_set['Date'].dt.date >= start_date) & (df2_set['Date'].dt.date <= end_date)
        df_filtered = df2_set.loc[mask].copy()

        df2_doi = df_filtered.copy()
        df2_doi = df2_doi[df2_doi['Name'].isin(unique_list)].copy()
        df2_doi = df2_doi.drop(columns=['index','Composition','Confidence','Publisher'], errors= 'ignore')
        df2_filtered = df2_doi.groupby(['Date','Name','Value']).size().reset_index(name='Frequency').reset_index()
        
        required_cols = {'Date', 'Value', 'Frequency', 'Name'}
        missing_cols = required_cols - set(df2_filtered.columns)
        if missing_cols:
            st.warning(f"Missing required columns: {', '.join(sorted(missing_cols))}")
            st.stop()

        if df2_filtered.empty:
            st.info("No materials found for the selected names.")
        else:
            df2_filtered['Date'] = pd.to_datetime(df2_filtered['Date'], errors='coerce')
            df2_filtered['Value'] = pd.to_numeric(df2_filtered['Value'], errors='coerce')
            df2_filtered['Frequency'] = pd.to_numeric(df2_filtered['Frequency'], errors='coerce')
            df2_filtered = df2_filtered.dropna(subset=['Date', 'Value', 'Frequency', 'Name'])

            df2_plot = df2_filtered.sort_values('Date').copy()

            fig, ax = plt.subplots(figsize=(9, 6))

            ax.axhspan(0, 1.6,  color='yellow', alpha=0.10, label='Infrared (0–1.6 eV)')
            ax.axhspan(1.6, 3.26, color='green',  alpha=0.10, label='Visible (1.6–3.26 eV)')
            ax.axhspan(3.26, df2_plot['Value'].max(), color='red',    alpha=0.10, label='Ultraviolet (3.26–4.0 eV)')

            groups = df2_plot['Name'].unique().tolist()
            cmap = plt.cm.get_cmap('tab10' if len(groups) <= 10 else 'tab20', len(groups))

            for idx, g in enumerate(groups):
                gdata = df2_plot[df2_plot['Name'] == g]
                if gdata.empty:
                    continue
                ax.scatter(
                    gdata['Date'], gdata['Value'],
                    s=(gdata['Frequency'].clip(lower=1) * 20),
                    c=[cmap(idx)], alpha=0.75,
                    edgecolors='white', linewidth=0.5, label=g
                )

            ax.set_xlabel("Publication Date")
            ax.set_ylabel("Bandgap Energy (eV)")
            ax.grid(True, linewidth=0.3, alpha=0.4)
            ax.legend(title='Material', loc='upper right', frameon=False, fontsize=8)

            st.pyplot(fig, clear_figure=True)

            st.markdown("***The table displays ten(10) sampled journals relating to the filtered semiconductors.***")

            n = min(10, len(df2_doi))

            if "sample_seed" not in st.session_state:
                st.session_state.sample_seed = 42

            if st.button("🔀 Shuffle sample"):
                st.session_state.sample_seed = random.randint(0, 10**9)

            st.table(df2_doi.sample(n=n, random_state=st.session_state.sample_seed))

            @st.cache_data
            def to_csv_bytes(df: pd.DataFrame) -> bytes:
                return df.to_csv(index=False).encode("utf-8")

            csv_bytes = to_csv_bytes(df2_doi)

            st.download_button(
                label="⬇️ Download excel file as CSV",
                data=csv_bytes,
                file_name="bandgap-filtered.csv",
                mime="text/csv",
            )

    elif selected_page == "Decision-making Assistant":
            st.title("Decision-making Assistant")
            st.markdown("Facilitate semiconductor selection with advanced filtering and visualization")
            
            if 'filters' not in st.session_state:
                st.session_state.filters = {}
            if 'initial_filter_name' not in st.session_state:
                st.session_state.initial_filter_name = None
            if 'initial_filters_only' not in st.session_state:
                st.session_state.initial_filters_only = {}
            if 'plot_x_col' not in st.session_state:
                st.session_state.plot_x_col = 'Bandgap'
            if 'plot_y_col' not in st.session_state:
                st.session_state.plot_y_col = 'Reserve (ton)'
            if 'excluded_elements' not in st.session_state:
                st.session_state.excluded_elements = []
            if 'additional_dynamic_filters' not in st.session_state:
                st.session_state.additional_dynamic_filters = []
            if 'filters_applied' not in st.session_state:
                st.session_state.filters_applied = False

            df_after_element_exclusion = filter_by_excluded_elements(df1, st.session_state.excluded_elements)
            
            st.markdown("### 1. Element Exclusion")
            
            periodic_table_url = "https://upload.wikimedia.org/wikipedia/commons/thumb/2/2e/Simple_Periodic_Table_Chart-en.svg/1200px-Simple_Periodic_Table_Chart-en.svg.png"
            st.image(periodic_table_url, caption="Periodic Table of Elements", width='stretch')
 
            if st.session_state.excluded_elements:
                removed_count = len(df1) - len(df_after_element_exclusion)
                st.info(f"🔬 Element filter active: Excluded {', '.join(sorted(st.session_state.excluded_elements))} | "
                        f"Removed {removed_count} materials | Showing {len(df_after_element_exclusion)} of {len(df1)} materials")
            
            all_elements = set()
            element_columns = ['Element_1', 'Element_2', 'Element_3', 'Element_4', 
                            'Element_5', 'Element_6', 'Element_7']
            
            for col in element_columns:
                if col in df1.columns:
                    elements = df1[col].dropna().unique()
                    all_elements.update(elements)
            
            all_elements = {elem for elem in all_elements if elem and str(elem).strip()}
            all_elements = sorted(list(all_elements))
            
            st.markdown("**Enter element symbols to exclude (separated by commas)**")

            element_text_input = st.text_input(
                "Element symbols:",
                value=", ".join(st.session_state.excluded_elements) if st.session_state.excluded_elements else "",
                key="element_text_input",
                placeholder="e.g., Au, Ag, Si, Pb",
                help="Enter element symbols separated by commas."
            )
            
            if element_text_input.strip():
                selected_elements = [elem.strip() for elem in element_text_input.split(',') if elem.strip()]
            else:
                selected_elements = []
            
            if st.session_state.excluded_elements:
                st.markdown(f"**Currently Excluded Elements:** {len(st.session_state.excluded_elements)}")
            
            if selected_elements != st.session_state.excluded_elements:
                preview_filtered = filter_by_excluded_elements(df1, selected_elements)
                would_remove = len(df1) - len(preview_filtered)
                if would_remove > 0:
                    st.warning(f"⚠️ Preview: This will remove {would_remove} materials ({would_remove/len(df1)*100:.1f}%) from the dataset.")
            
            st.markdown("### 2. Initial Filters")
            cols = st.columns(2)

            with cols[0]:
                st.markdown("#### Bandgap Selection")
                col1, col2 = st.columns(2)
                with col1:
                    bandgap_min = st.number_input(
                        "Min (eV)",
                        min_value=0.0,
                        max_value=35.0,
                        value=0.0,
                        step=0.1,
                        key="bandgap_min"
                    )
                with col2:
                    bandgap_max = st.number_input(
                        "Max (eV)",
                        min_value=0.0,
                        max_value=35.0,
                        value=3.0,
                        step=0.1,
                        key="bandgap_max"
                    )

                if bandgap_min > bandgap_max:
                    st.error("Minimum bandgap must be less than or equal to maximum bandgap")
                    
                bandgap_range = (bandgap_min, bandgap_max)

            with cols[1]:
                st.markdown("#### Additional Filter")
                filter_options = [
                    'Reserve (ton)', 'Production (ton)', 'HHI (USGS)',
                    'ESG Score', 'CO2 footprint max (kg/kg)', 
                    'Embodied energy max (MJ/kg)', 'Water usage max (l/kg)', 
                    'Toxicity', 'Companionality'
                ]
                
                selected_filter = st.selectbox("Choose a filter", filter_options, key="selected_filter")
                
                if selected_filter:
                    temp_filtered = filter_by_excluded_elements(df1, selected_elements) if selected_elements else df1
                    filter_min = float(temp_filtered[selected_filter].min())
                    filter_max = float(temp_filtered[selected_filter].max())
                    
                    if selected_filter in ['Production (ton)', 'Reserve (ton)']:
                        filter_min_input = st.number_input(
                            f"Minimum Requirement (tonnes)",
                            min_value=filter_min,
                            max_value=filter_max,
                            value=filter_min,
                            step=1000.0,
                            format="%.2f",
                            key="filter_min_input"
                        )
                        
                        filter_range = (filter_min_input, filter_max)
                        st.caption(f"**Minimum Required:** {format_tons(filter_min_input)}")
                    
                    elif selected_filter == 'Toxicity':
                        filter_range = st.slider(
                            f"{selected_filter} Range",
                            int(filter_min),
                            int(filter_max),
                            (int(filter_min), int(filter_max)),
                            step=1,
                            key="initial_filter_slider"
                        )
                    
                    else:
                        filter_range = st.slider(
                            f"{selected_filter} Range",
                            filter_min,
                            filter_max,
                            (filter_min, filter_max),
                            key="initial_filter_slider"
                        )
                else:
                    filter_range = None
            
            st.markdown("### 3. Additional Filters (Optional)")
            
            all_filter_options = [
                'Reserve (ton)', 'Production (ton)', 'HHI (USGS)',
                'ESG Score', 'CO2 footprint max (kg/kg)', 
                'Embodied energy max (MJ/kg)', 'Water usage max (l/kg)', 
                'Toxicity', 'Companionality'
            ]
            
            available_for_dynamic = [f for f in all_filter_options if f != selected_filter]
            
            col_add_btn, col_info = st.columns([1, 3])
            with col_add_btn:
                if st.button("➕ Add Filter", key="add_dynamic_filter"):
                    if len(st.session_state.additional_dynamic_filters) < len(available_for_dynamic):
                        st.session_state.additional_dynamic_filters.append({
                            'filter_name': None,
                            'filter_range': None
                        })
                        st.rerun()
            with col_info:
                st.caption(f"You can add up to {len(available_for_dynamic)} additional filters")
            
            dynamic_filter_values = {}
            filters_to_remove = []
            
            for idx, filter_config in enumerate(st.session_state.additional_dynamic_filters):
                st.markdown(f"#### Filter #{idx + 2}")
                
                col1, col2, col3 = st.columns([2, 3, 1])
                
                with col1:
                    used_filters = [selected_filter] + [f['filter_name'] for f in st.session_state.additional_dynamic_filters if f['filter_name']]
                    available_options = [f for f in available_for_dynamic if f not in used_filters or f == filter_config.get('filter_name')]
                    
                    if available_options:
                        dynamic_filter_name = st.selectbox(
                            "Select filter",
                            options=available_options,
                            index=available_options.index(filter_config['filter_name']) if filter_config.get('filter_name') in available_options else 0,
                            key=f"dynamic_filter_name_{idx}"
                        )
                        filter_config['filter_name'] = dynamic_filter_name
                    else:
                        st.warning("No more filters available")
                        dynamic_filter_name = None
                
                with col2:
                    if dynamic_filter_name:
                        temp_filtered = filter_by_excluded_elements(df1, selected_elements) if selected_elements else df1
                        dyn_filter_min = float(temp_filtered[dynamic_filter_name].min())
                        dyn_filter_max = float(temp_filtered[dynamic_filter_name].max())
                        
                        if dynamic_filter_name == 'Toxicity':
                            dyn_filter_range = st.slider(
                                f"{dynamic_filter_name} Range",
                                int(dyn_filter_min),
                                int(dyn_filter_max),
                                (int(dyn_filter_min), int(dyn_filter_max)),
                                step=1,
                                key=f"dynamic_filter_range_{idx}"
                            )
                        elif dynamic_filter_name in ['Production (ton)', 'Reserve (ton)']:
                            dyn_filter_range = st.slider(
                                f"{dynamic_filter_name} Range",
                                dyn_filter_min,
                                dyn_filter_max,
                                (dyn_filter_min, dyn_filter_max),
                                format="",
                                key=f"dynamic_filter_range_{idx}"
                            )
                            st.caption(f"**Range:** {format_tons(dyn_filter_range[0])} to {format_tons(dyn_filter_range[1])}")
                        else:
                            dyn_filter_range = st.slider(
                                f"{dynamic_filter_name} Range",
                                dyn_filter_min,
                                dyn_filter_max,
                                (dyn_filter_min, dyn_filter_max),
                                key=f"dynamic_filter_range_{idx}"
                            )
                        
                        dynamic_filter_values[dynamic_filter_name] = dyn_filter_range
                
                with col3:
                    if st.button("🗑️", key=f"remove_filter_{idx}", help="Remove this filter"):
                        filters_to_remove.append(idx)
            
            if filters_to_remove:
                for idx in sorted(filters_to_remove, reverse=True):
                    st.session_state.additional_dynamic_filters.pop(idx)
                st.rerun()
            
            if st.button("Apply Filters", key="apply_all_filters", type="primary"):
                if filter_range is not None:
                    st.session_state.excluded_elements = selected_elements
                    
                    all_filters = {
                        "Bandgap": bandgap_range,
                        selected_filter: filter_range
                    }
                    
                    all_filters.update(dynamic_filter_values)
                    
                    st.session_state.filters = all_filters
                    st.session_state.initial_filters_only = all_filters.copy()
                    st.session_state.initial_filter_name = selected_filter
                    st.session_state.plot_x_col = 'Bandgap'
                    st.session_state.plot_y_col = selected_filter
                    st.session_state.filters_applied = True
                    
                    filter_count = len(all_filters)
                    st.success(f"✅ {filter_count} filter(s) applied successfully!")
                    st.rerun()
                else:
                    st.warning("Please select an additional filter and set its range.")
            
            st.subheader("Filtered Results")
            
            if st.session_state.initial_filters_only:
                df_filtered = filter_dataframe(df_after_element_exclusion, st.session_state.initial_filters_only)
                
                filter_summary = ", ".join([f"{k}" for k in st.session_state.initial_filters_only.keys()])
                st.info(f"📊 Showing {len(df_filtered)} materials | Filters applied: {filter_summary} | Available: {len(df_after_element_exclusion)} (after element exclusion)")
            else:
                df_filtered = df_after_element_exclusion.copy()
                st.info(f"📈 Showing all {len(df_after_element_exclusion)} available materials (after element exclusion)")
            
            x_col = st.session_state.plot_x_col
            y_col = st.session_state.plot_y_col
            
            col1, col2, col3 = st.columns([2, 2, 1])
            with col1:
                st.write(f"**X-axis:** {x_col}")
            with col2:
                st.write(f"**Y-axis:** {y_col}")
            with col3:
                log_y = st.checkbox(f"Log Y-axis", key="log_y_main")
            
            plot_title = f"{x_col} vs {y_col}"
            
            if not df_filtered.empty:
                p = create_professional_plot(
                    df_filtered, x_col, y_col, plot_title, x_col, y_col, False, log_y
                )
                st.plotly_chart(p, use_container_width=True, key="decision_making_scatter_plot")
            else:
                st.warning("⚠️ No materials match the current filters")

            if st.session_state.filters_applied and not df_filtered.empty:
                st.markdown("---")
                st.subheader("4. Multi-Criteria Decision Making")
                st.info(f"Analyze the {len(df_filtered)} filtered materials using TOPSIS or PROMETHEE methods")
                
                cols_mcdm = st.columns(2)
                with cols_mcdm[0]:
                    mcdm_method = st.selectbox(
                        "Method",
                        ["TOPSIS", "PROMETHEE"],
                        help="TOPSIS: Technique for Order Preference by Similarity to Ideal Solution\nPROMETHEE: Preference Ranking Organization Method for Enrichment Evaluation",
                        key="mcdm_method_custom"
                    )
                with cols_mcdm[1]:
                    weighting_method = st.radio(
                        "Weighting",
                        ["Entropy Weighting", "Manual Weights"],
                        horizontal=True,
                        key="mcdm_weighting_custom"
                    )
                
                criteria_options = {
                    'Reserve (ton)': 1, 'Production (ton)': 1, 'HHI (USGS)': -1,
                    'ESG Score': -1, 'CO2 footprint max (kg/kg)': -1,
                    'Embodied energy max (MJ/kg)': -1, 'Water usage max (l/kg)': -1,
                    'Toxicity': -1, 'Companionality': -1
                }
                available_criteria = {k: v for k, v in criteria_options.items() if k in df_filtered.columns}
                
                if not available_criteria:
                    st.error("❌ No criteria columns found in filtered data.")
                    st.stop()
                
                if weighting_method == "Entropy Weighting":
                    if len(df_filtered) < 20:
                        st.warning(f"⚠️ Warning: Only {len(df_filtered)} materials available.")
                    
                    try:
                        matrix_for_entropy = df_filtered[list(available_criteria.keys())].values
                        
                        if np.isnan(matrix_for_entropy).any():
                            nan_count = np.isnan(matrix_for_entropy).sum()
                            st.error(f"❌ Cannot calculate entropy weights: {nan_count} missing values found.")
                            weights = None
                        
                        elif np.any(matrix_for_entropy < 0):
                            st.error("❌ Cannot calculate entropy weights: Negative values found.")
                            weights = None
                        
                        else:
                            n = matrix_for_entropy.shape[0]
                            m = matrix_for_entropy.shape[1]
                            
                            probability_matrix = np.zeros_like(matrix_for_entropy, dtype=float)
                            
                            for j in range(m):
                                col_data = matrix_for_entropy[:, j]
                                col_sum = np.sum(col_data)
                                
                                if col_sum < 1e-10:
                                    st.warning(f"⚠️ All values in column zero.")
                                    probability_matrix[:, j] = 1.0 / n
                                else:
                                    probability_matrix[:, j] = col_data / col_sum
                            
                            entropies = []
                            diversities = []
                            
                            for j in range(m):
                                p = probability_matrix[:, j]
                                p_safe = np.where(p > 1e-10, p, 1e-10)
                                e_j = -np.sum(p_safe * np.log(p_safe)) / np.log(n)
                                d_j = 1 - e_j
                                
                                entropies.append(e_j)
                                diversities.append(d_j)
                            
                            entropies = np.array(entropies)
                            diversities = np.array(diversities)
                            
                            diversity_sum = np.sum(diversities)
                            if diversity_sum > 1e-10:
                                weights = diversities / diversity_sum
                            else:
                                st.warning("⚠️ All criteria have zero diversity.")
                                weights = np.ones(m) / m
                            
                            if weights is None or np.isnan(weights).any() or np.isinf(weights).any():
                                st.error("❌ Entropy weighting failed.")
                                weights = np.ones(len(available_criteria)) / len(available_criteria)
                                st.success(f"✅ Using equal weights: {1/len(available_criteria):.2%}")
                            else:
                                st.success("✅ Entropy weights calculated successfully")
                                    
                    except Exception as e:
                        st.error(f"❌ Error calculating entropy weights: {str(e)}")
                        weights = np.ones(len(available_criteria)) / len(available_criteria)
                        st.success(f"✅ Using equal weights: {1/len(available_criteria):.2%}")

                else:
                    st.markdown("**📊 Criteria Weights** - Assign importance (0–5 scale):")
                    
                    if 'preset_weights' not in st.session_state:
                        st.session_state.preset_weights = {col: 3 for col in available_criteria.keys()}
                    
                    st.markdown("##### Quick Presets")
                    preset_cols = st.columns(3)
                    
                    with preset_cols[0]:
                        if st.button("Balanced", key="preset_balanced"):
                            st.session_state.preset_weights = {col: 3 for col in available_criteria.keys()}
                            st.rerun()
                    
                    with preset_cols[1]:
                        if st.button("Long-term goal", key="preset_long_term"):
                            st.session_state.preset_weights = {}
                            for col in available_criteria.keys():
                                if col in ['ESG Score', 'Toxicity', 'Companionality', 'Reserve (ton)']:
                                    st.session_state.preset_weights[col] = 5
                                else:
                                    st.session_state.preset_weights[col] = 1
                            st.rerun()
                    
                    with preset_cols[2]:
                        if st.button("Short-term goal", key="preset_short_term"):
                            st.session_state.preset_weights = {}
                            for col in available_criteria.keys():
                                if col in ['Production (ton)', 'HHI (USGS)', 'CO2 footprint max (kg/kg)', 'Water usage max (l/kg)', 'Embodied energy max (MJ/kg)']:
                                    st.session_state.preset_weights[col] = 5
                                else:
                                    st.session_state.preset_weights[col] = 1
                            st.rerun()
                    
                    st.markdown("##### Adjust Individual Weights")
                    
                    weights = []
                    criteria_list = list(available_criteria.items())
                    
                    mid_point = (len(criteria_list) + 1) // 2
                    
                    cols_row1 = st.columns(mid_point)
                    for i, (col, direction) in enumerate(criteria_list[:mid_point]):
                        with cols_row1[i]:
                            default_value = st.session_state.preset_weights.get(col, 3)
                            weight = st.slider(
                                f"{col}",
                                0, 5, 
                                value=default_value,
                                key=f"weight_custom_{col}"
                            )
                            weights.append(weight)
                    
                    if len(criteria_list) > mid_point:
                        cols_row2 = st.columns(len(criteria_list) - mid_point)
                        for i, (col, direction) in enumerate(criteria_list[mid_point:]):
                            with cols_row2[i]:
                                default_value = st.session_state.preset_weights.get(col, 3)
                                weight = st.slider(
                                    f"{col}",
                                    0, 5, 
                                    value=default_value,
                                    key=f"weight_custom_{col}"
                                )
                                weights.append(weight)
                    
                    if sum(weights) == 0:
                        st.warning("All weights set to 0 - using equal weights")
                        weights = np.ones(len(weights)) / len(weights)
                    else:
                        weights = np.array(weights) / sum(weights)
                
                weights_df = pd.DataFrame({
                    'Criterion': list(available_criteria.keys()),
                    'Weight': weights,
                    'Direction': ['Maximize' if d == 1 else 'Minimize' for d in available_criteria.values()]
                }).sort_values('Weight', ascending=False).reset_index(drop=True)
                
                weights_df.index = weights_df.index + 1
                weights_df.index.name = 'Rank'
                
                st.subheader("Criteria Weights")
                
                if weights is None:
                    st.error("❌ Error: Weights are None.")
                elif len(weights) == 0:
                    st.error("❌ Error: No weights calculated.")
                elif np.isnan(weights).any():
                    st.error("❌ Error: Some weights are NaN.")
                    st.dataframe(weights_df)
                else:
                    st.dataframe(
                        weights_df.style.format({'Weight': '{:.2%}'}),
                        use_container_width=True
                    )
                
                if st.button("🚀 Run MCDM Analysis", type="primary", key="run_mcdm_custom"):
                    with st.spinner("Performing analysis..."):
                        matrix = df_filtered[list(available_criteria.keys())].values
                        types = np.array([available_criteria[k] for k in available_criteria])
                        
                        if np.isnan(matrix).any():
                            nan_count = np.isnan(matrix).sum()
                            st.error(f"❌ Error: Found {nan_count} missing values.")
                            st.stop()
                        
                        if weights is None or len(weights) == 0:
                            st.error("❌ Error: Weights are not defined.")
                            st.stop()
                        
                        if np.isnan(weights).any():
                            st.error("❌ Error: Weights contain NaN values.")
                            st.stop()
                        
                        if not np.isclose(np.sum(weights), 1.0):
                            st.warning(f"⚠️ Normalizing weights to 1.0")
                            weights = weights / np.sum(weights)
                        
                        try:
                            if mcdm_method == "TOPSIS":
                                scores = run_topsis(matrix, weights, types)
                                
                                if np.isnan(scores).any():
                                    st.error("❌ TOPSIS returned NaN scores.")
                                    st.stop()
                                
                                results = pd.DataFrame({
                                    'Material': df_filtered['Name'].values,
                                    'Bandgap (eV)': df_filtered['Bandgap'].values,
                                    'DOI': df_filtered['DOI'].values,
                                    'Score': scores
                                }).sort_values('Score', ascending=False).reset_index(drop=True)
                                
                                results.index = results.index + 1
                                results.index.name = 'Rank'
                                
                            else:
                                flows = run_promethee(matrix, weights, types)
                                
                                if np.isnan(flows).any():
                                    st.error("❌ PROMETHEE returned NaN flows.")
                                    st.stop()
                                
                                results = pd.DataFrame({
                                    'Material': df_filtered['Name'].values,
                                    'Bandgap (eV)': df_filtered['Bandgap'].values,
                                    'Net Flow': flows
                                }).sort_values('Net Flow', ascending=False).reset_index(drop=True)
                                
                                results.index = results.index + 1
                                results.index.name = 'Rank'
                                
                        except Exception as e:
                            st.error(f"❌ Error running {mcdm_method}: {str(e)}")
                            st.stop()
                    
                    st.subheader("MCDM Results")

                    display_cols = ['Material', 'Bandgap (eV)', 'DOI']
                    if 'Score' in results.columns:
                        display_cols.append('Score')
                    else:
                        display_cols.append('Net Flow')

                    top_n = 100
                    st.write(f"Showing top {top_n} results (out of {len(results)} total)")

                    format_dict = {'Bandgap (eV)': '{:.2f}'}
                    if 'Score' in results.columns:
                        format_dict['Score'] = '{:.4f}'
                    else:
                        format_dict['Net Flow'] = '{:.4f}'
                    
                    st.dataframe(
                        results[display_cols].head(top_n).style.format(format_dict),
                        use_container_width=True
                    )
                    
                    st.subheader("🏆 Top Materials")
                    
                    unique_top_materials = results.drop_duplicates(subset=['Material'], keep='first').head(3)
                    top_n = len(unique_top_materials)
                    
                    if top_n > 0:
                        cols_top = st.columns(top_n)
                        for i in range(top_n):
                            with cols_top[i]:
                                material = unique_top_materials.iloc[i]['Material']
                                rank_num = unique_top_materials.index[i]
                                bandgap = unique_top_materials.iloc[i]['Bandgap (eV)']
                                st.metric(
                                    label=f"Rank #{rank_num}",
                                    value=material
                                )
                    else:
                        st.info("No materials to display")
                    
                    excel_data = create_full_output(df_filtered, results, weights_df)
                    st.download_button(
                        label="📥 Download Full MCDM Report",
                        data=excel_data,
                        file_name=f"mcdm_analysis_{mcdm_method}.xlsx",
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                        key="download_mcdm_custom"
                    )

if __name__ == "__main__":
    main()