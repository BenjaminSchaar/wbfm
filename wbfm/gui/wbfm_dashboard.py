import argparse
import os
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Optional

import numpy as np
from dash import Dash, dcc, html, Output, Input
import plotly.express as px
import pandas as pd
from scipy import interpolate


def _correlate_return_cross_terms(df0: pd.DataFrame, df1: pd.DataFrame) -> pd.DataFrame:
    """
    Like df.corr(), but acts on two dataframes, returning only the cross terms

    Parameters
    ----------
    df0
    df1

    Returns
    -------

    """
    df_corr = pd.concat([df0, df1], axis=1).corr()
    return get_corner_from_corr_df(df0, df_corr)


def get_corner_from_corr_df(df0: pd.DataFrame, df_corr: pd.DataFrame):
    """
    If correlations are calculated between a concatenated version of df_trace and something else, this gets only the
    cross terms from df_corr

    Parameters
    ----------
    df0
    df_corr

    Returns
    -------

    """
    ind_nonneuron = np.arange(df0.shape[1], df_corr.shape[1])
    ind_neurons = np.arange(0, df0.shape[1])
    return df_corr.iloc[ind_neurons, ind_nonneuron]


def _get_names_from_df(df, level=0):
    """
    Simpler copy of get_names_from_df utility
    """
    # Handle both multi-level and simple column structures
    if hasattr(df.columns, 'nlevels') and df.columns.nlevels > 1:
        names = list(set(df.columns.get_level_values(level)))
    else:
        # Simple column structure - return column names directly
        names = list(df.columns)
    names.sort()
    return names


def _load_custom_timeseries_csvs(custom_timeseries_path: Path) -> dict:
    """
    Load all CSV files from custom_timeseries folder and validate format.
    
    Parameters
    ----------
    custom_timeseries_path : Path
        Path to custom_timeseries folder
        
    Returns
    -------
    pd.DataFrame
        DataFrame with custom timeseries data, columns are timeseries names
    """
    print(f"Loading custom timeseries from: {custom_timeseries_path}")
    if not custom_timeseries_path.exists():
        print("No custom_timeseries folder found")
        return pd.DataFrame()
    
    custom_data = {}
    csv_files = list(custom_timeseries_path.glob("*.csv"))
    
    if len(csv_files) == 0:
        print("No CSV files found in custom_timeseries folder")
        return pd.DataFrame()
    else:
        print(f"Found {len(csv_files)} custom timeseries CSV files")
    
    for csv_file in csv_files:
        try:
            print(f"DEBUG: Attempting to load {csv_file.name}")
            
            # Load CSV and validate format
            df_custom = pd.read_csv(csv_file)
            
            print(f"DEBUG: CSV shape: {df_custom.shape}")
            print(f"DEBUG: CSV columns (raw): {repr(df_custom.columns.tolist())}")
            print(f"DEBUG: CSV dtypes: {df_custom.dtypes.to_dict()}")
            print(f"DEBUG: First few rows:\n{df_custom.head()}")
            
            # Clean column names (remove whitespace, BOM, etc.)
            df_custom.columns = df_custom.columns.str.strip()
            print(f"DEBUG: CSV columns (cleaned): {df_custom.columns.tolist()}")
            
            # Validate required columns
            if list(df_custom.columns) != ['frame', 'value']:
                print(f"ERROR: Custom timeseries file {csv_file.name} has incorrect format.")
                print(f"Expected columns: ['frame', 'value'], got: {list(df_custom.columns)}")
                print("Skipping this file.")
                continue
                
            # Validate data types
            if not pd.api.types.is_numeric_dtype(df_custom['frame']):
                print(f"ERROR: 'frame' column in {csv_file.name} must be numeric. Skipping this file.")
                print(f"DEBUG: frame column dtype: {df_custom['frame'].dtype}")
                print(f"DEBUG: frame column sample values: {df_custom['frame'].head().tolist()}")
                continue
                
            if not pd.api.types.is_numeric_dtype(df_custom['value']):
                print(f"ERROR: 'value' column in {csv_file.name} must be numeric. Skipping this file.")
                print(f"DEBUG: value column dtype: {df_custom['value'].dtype}")
                print(f"DEBUG: value column sample values: {df_custom['value'].head().tolist()}")
                continue
            
            # Set frame as index and use filename (without .csv) as column name
            timeseries_name = csv_file.stem
            df_custom = df_custom.set_index('frame')
            custom_data[timeseries_name] = df_custom['value']
            
            print(f"SUCCESS: Loaded custom timeseries: {timeseries_name} ({len(df_custom)} frames)")
            
        except Exception as e:
            print(f"ERROR loading custom timeseries {csv_file.name}: {e}")
            print(f"DEBUG: Exception type: {type(e)}")
            import traceback
            print(f"DEBUG: Full traceback:\n{traceback.format_exc()}")
            continue
    
    if len(custom_data) == 0:
        return {}
    
    # Return the raw data dictionary for individual processing
    return custom_data


def _process_individual_custom_timeseries(csv_data: dict, target_length: int) -> pd.DataFrame:
    """
    Process each custom timeseries individually and resample as needed.
    
    Parameters
    ----------
    csv_data : dict
        Dictionary of {timeseries_name: Series} with raw data
    target_length : int
        Target number of frames to match traces
        
    Returns
    -------
    pd.DataFrame
        Processed DataFrame with all timeseries aligned to target_length
    """
    if not csv_data:
        return pd.DataFrame()
    
    processed_data = {}
    
    for timeseries_name, series_data in csv_data.items():
        original_length = len(series_data)
        
        # Check if resampling is needed for this individual file
        if original_length == target_length:
            print(f"Custom timeseries '{timeseries_name}' length matches target ({original_length} frames) - keeping original resolution")
            # Reindex to ensure consistent index (0 to target_length-1)
            processed_data[timeseries_name] = series_data.reindex(range(target_length))
        else:
            # Resample this individual timeseries
            print(f"Resampling '{timeseries_name}' from {original_length} to {target_length} frames")
            
            # Create interpolation function for this series
            original_indices = np.linspace(0, target_length - 1, original_length)
            new_indices = np.arange(target_length)
            
            f = interpolate.interp1d(original_indices, series_data.values, 
                                   kind='linear', bounds_error=False, fill_value='extrapolate')
            
            processed_data[timeseries_name] = f(new_indices)
    
    # Combine all processed timeseries
    df_result = pd.DataFrame(processed_data, index=range(target_length))
    print(f"Successfully processed {len(processed_data)} custom timeseries, all aligned to {target_length} frames")
    
    return df_result


def _downsample_custom_timeseries(df_custom: pd.DataFrame, target_length: int) -> pd.DataFrame:
    """
    Resample custom timeseries to match target frame count.
    
    Parameters
    ----------
    df_custom : pd.DataFrame
        Custom timeseries data with frame index
    target_length : int
        Target number of frames to match traces
        
    Returns
    -------
    pd.DataFrame
        Resampled DataFrame with target_length rows
    """
    if df_custom.empty:
        return df_custom
    
    original_length = len(df_custom)
    
    # Check if resampling is needed
    if original_length == target_length:
        print(f"Custom timeseries length matches target ({original_length} frames) - keeping original resolution")
        return df_custom.copy()
    
    # Create new index with target length
    new_index = np.arange(target_length)
    
    # Interpolate each column to the new length
    resampled_data = {}
    
    for col in df_custom.columns:
        # Create interpolation function
        original_indices = np.linspace(0, target_length - 1, original_length)
        f = interpolate.interp1d(original_indices, df_custom[col].values, 
                               kind='linear', bounds_error=False, fill_value='extrapolate')
        
        # Apply interpolation
        resampled_data[col] = f(new_index)
    
    df_resampled = pd.DataFrame(resampled_data, index=new_index)
    
    # Log the resampling operation
    if original_length > target_length:
        print(f"Downsampled custom timeseries from {original_length} to {target_length} frames")
    else:
        print(f"Upsampled custom timeseries from {original_length} to {target_length} frames")
    
    return df_resampled


@dataclass
class DashboardDataset:
    project_path: str
    port: int = None
    allow_public_access: bool = False

    df_final: pd.DataFrame = None
    df_custom_timeseries: pd.DataFrame = None

    current_dataset: Optional[str] = None
    current_neuron: str = None

    def __post_init__(self):
        # Read data
        if isinstance(self.project_path, str) and self.project_path.endswith('.h5'):
            # Maybe the user passed the filename, not the project config name
            fname = self.project_path
            project_folder = Path(self.project_path).parent
        else:
            project_folder = Path(self.project_path).parent
            fname = project_folder.joinpath('final_dataframes/df_final.h5')
        self.df_final = pd.read_hdf(fname)

        if self.df_final.columns.nlevels == 4:
            # Multi dataset
            self.dataset_names = _get_names_from_df(self.df_final)
            self.current_dataset = self.dataset_names[0]
        elif self.df_final.columns.nlevels == 3:
            # Single dataset
            self.dataset_names = None
            self.current_dataset = None
        else:
            raise NotImplementedError

        # Load custom timeseries if available
        custom_timeseries_path = project_folder.joinpath('behavior/custom_timeseries')
        print(f"DEBUG: Full custom timeseries path: {custom_timeseries_path}")
        print(f"DEBUG: Path exists: {custom_timeseries_path.exists()}")
        if custom_timeseries_path.exists():
            files_in_folder = list(custom_timeseries_path.iterdir())
            print(f"DEBUG: Files in custom_timeseries folder: {[f.name for f in files_in_folder]}")
        csv_data = _load_custom_timeseries_csvs(custom_timeseries_path)
        
        if csv_data:  # csv_data is now a dict, not a DataFrame
            print(f"Loaded {len(csv_data)} custom timeseries: {list(csv_data.keys())}")
            # Get target length from traces data - access directly from df_final
            if self.current_dataset is None:
                df_traces = self.df_final['traces']
            else:
                df_traces = self.df_final[self.current_dataset]['traces']
            
            trace_names = _get_names_from_df(df_traces)
            if trace_names:
                target_length = len(df_traces[trace_names[0]])
                self.df_custom_timeseries = _process_individual_custom_timeseries(csv_data, target_length)
                print(f"Successfully integrated custom timeseries with traces (aligned to {target_length} frames)")
            else:
                print("WARNING: No trace data found for downsampling custom timeseries")
                self.df_custom_timeseries = pd.DataFrame()
        else:
            self.df_custom_timeseries = pd.DataFrame()

    def dataset_of_current_neuron(self) -> str:
        # In case current_dataset == 'all'
        if self.current_dataset != 'all':
            return self.current_dataset
        else:
            # Hacky: attempt to get it from the current neuron name
            return self.current_neuron.split('-')[1]

    @property
    def df_behavior(self):
        # Always a single dataset
        if self.current_dataset is None:
            return self.df_final['behavior']['behavior']
        else:
            return self.df_final[self.dataset_of_current_neuron()]['behavior']['behavior']

    @property
    def df_curvature(self):
        # Always a single dataset
        if self.current_dataset is None:
            return self.df_final['behavior']['curvature']
        else:
            return self.df_final[self.dataset_of_current_neuron()]['behavior']['curvature']

    @staticmethod
    def rename_joined_neurons(neuron_name, dataset_name):
        return f"{neuron_name}-{dataset_name}"

    @property
    def df_all_traces(self):
        if self.current_dataset is None:
            return self.df_final['traces']
        else:
            return self.df_final[self.dataset_of_current_neuron()]['traces']

    @property
    def df_behavior_with_custom(self):
        """
        Combine regular behavior data with custom timeseries for correlation analysis
        """
        df_behavior = self.df_behavior
        if not self.df_custom_timeseries.empty:
            # Check if behavior data has multi-level columns
            if hasattr(df_behavior.columns, 'nlevels') and df_behavior.columns.nlevels > 1:
                # Flatten behavior columns for easier combination
                df_behavior_flat = df_behavior.copy()
                if df_behavior.columns.nlevels > 1:
                    df_behavior_flat.columns = ['_'.join(col).strip() for col in df_behavior.columns.values]
                combined = pd.concat([df_behavior_flat, self.df_custom_timeseries], axis=1)
            else:
                combined = pd.concat([df_behavior, self.df_custom_timeseries], axis=1)
            
            return combined
        else:
            return df_behavior

    def get_trace_type(self, trace_type: str):
        # May be a joined version of multiple datasets
        if self.current_dataset == 'all':
            # Build the dataset from all individual dataframes
            dataset_name = self.dataset_names[0]
            mapper = partial(self.rename_joined_neurons, dataset_name=dataset_name)
            df_joined = self.df_final[dataset_name]['traces'][trace_type].copy().rename(columns=mapper)
            for dataset_name in self.dataset_names[1:]:
                mapper = partial(self.rename_joined_neurons, dataset_name=dataset_name)
                df_to_join = self.df_final[dataset_name]['traces'][trace_type].copy().rename(columns=mapper)
                df_joined = df_joined.join(df_to_join)

            df = df_joined

        elif trace_type in self.df_all_traces:
            df = self.df_all_traces[trace_type]
        else:
            raise NotImplementedError
        return df

    def serve_wbfm_dashboard(self):
        """
        Builds a dash/plotly dashboard for exploring single neuron correlations to behavior
        """

        app = Dash(__name__)

        # Initialize hardcoded paths to files (will open in new tab)
        path_to_grid_plot = Path(self.project_path).parent.joinpath('traces').\
            joinpath('ratio_integration_rolling_mean_beh_pc1-grid-.png')

        # Define layout
        curvature_names = _get_names_from_df(self.df_curvature)
        behavior_names = _get_names_from_df(self.df_behavior_with_custom)
        trace_names = _get_names_from_df(self.df_all_traces)
        neuron_names = _get_names_from_df(self.df_all_traces[trace_names[0]])
        if self.dataset_names is None:
            dataset_names = [None]
        else:
            dataset_names = self.dataset_names.copy()
        if dataset_names is not None:
            dataset_names.append('all')

        app.layout = html.Div([
            build_dropdowns(behavior_names, curvature_names, trace_names, neuron_names, dataset_names),
            build_second_row_options(path_to_grid_plot),
            build_plots_html(),
            build_kymograph(self.df_curvature)
            ]
        )

        @app.callback(
            Output('neuron-select-dropdown', 'options'),
            Input('dataset-dropdown', 'value')
        )
        def _update_neuron_dropdown(current_dataset):
            # Get any trace type; assume they have the same names
            self.current_dataset = current_dataset
            new_neuron_names = _get_names_from_df(self.get_trace_type('ratio'))
            return new_neuron_names

        # Main scatter plot changing callback (change axes)
        @app.callback(
            Output('correlation-scatterplot', 'figure'),
            Input('scatter-xaxis', 'value'),
            Input('behavior-scatter-yaxis', 'value'),
            Input('neuron-select-dropdown', 'value'),
            Input('regression-type', 'value'),
            Input('trace-select-dropdown', 'value'),
            Input('dataset-dropdown', 'value')
        )
        def _update_scatter_plot(x_name, y_name, neuron_name, regression_type, trace_type, current_dataset):
            self.current_dataset = current_dataset
            df_traces = self.get_trace_type(trace_type)
            return update_scatter_plot(self.df_behavior_with_custom, df_traces, x_name, y_name, neuron_name, regression_type)

        # Neuron selection updates
        # Logic: everything goes through the dropdown menu. A click will update that, which updates other things
        @app.callback(
            Output('neuron-select-dropdown', 'value'),
            Output('correlation-scatterplot', 'clickData'),
            Output('kymograph-all-neuron-max-segment-correlation', 'clickData'),
            Input('correlation-scatterplot', 'clickData'),
            Input('kymograph-all-neuron-max-segment-correlation', 'clickData'),
            Input('neuron-select-dropdown', 'value'),
            Input('neuron-select-dropdown', 'options')
        )
        def _use_click_to_update_neuron_dropdown(correlation_clickData, kymograph_clickData,
                                                 previous_neuron_name, neuron_options):
            # Note: needs neuron_options as input because this is the only function that can change the
            # neuron-select-dropdown value

            # Resets the clickData of each plot (multiple outputs)
            if correlation_clickData:
                neuron_name = correlation_clickData["points"][0]["customdata"][0]
            elif kymograph_clickData:
                neuron_name = kymograph_clickData["points"][0]["customdata"][0]
            else:
                neuron_name = previous_neuron_name

            if neuron_name not in neuron_options:
                neuron_name = list(neuron_options)[0]

            if neuron_name:
                self.current_neuron = neuron_name

            return neuron_name, None, None

        @app.callback(
            Output('kymograph-select-dropdown', 'value'),
            Input('kymograph-per-segment-correlation', 'clickData')
        )
        def _use_click_to_update_kymograph_segment(clickData):
            if clickData is None:
                return 'segment_001'
            kymograph_segment_name = clickData["points"][0]["x"]
            return kymograph_segment_name

        @app.callback(
            Output('neuron-trace', 'figure'),
            Input('neuron-select-dropdown', 'value'),
            Input('regression-type', 'value'),
            Input('trace-select-dropdown', 'value'),
            Input('dataset-dropdown', 'value')
        )
        def _update_neuron_trace(neuron_name, regression_type, trace_type, current_dataset):
            self.current_dataset = current_dataset
            df_traces = self.get_trace_type(trace_type)
            self.df_behavior_and_neurons = pd.concat([self.df_behavior_with_custom, df_traces], axis=1)
            return update_neuron_trace_plot(self.df_behavior_and_neurons, neuron_name, regression_type)

        @app.callback(
            Output('trace-and-behavior-scatterplot', 'figure'),
            Input('neuron-select-dropdown', 'value'),
            Input('behavior-scatter-yaxis', 'value'),
            Input('regression-type', 'value'),
            Input('trace-select-dropdown', 'value'),
            Input('dataset-dropdown', 'value')
        )
        def _update_behavior_scatter(neuron_name, behavior_name, regression_type, trace_type, current_dataset):
            self.current_dataset = current_dataset
            df_traces = self.get_trace_type(trace_type)
            self.df_behavior_and_neurons = pd.concat([self.df_behavior_with_custom, df_traces], axis=1)
            return update_behavior_scatter_plot(self.df_behavior_and_neurons, behavior_name, neuron_name, regression_type)

        # Behavior updates
        @app.callback(
            Output('behavior-trace', 'figure'),
            Input('behavior-scatter-yaxis', 'value'),
            Input('regression-type', 'value'),
            Input('dataset-dropdown', 'value')
        )
        def _update_behavior_trace(behavior_name, regression_type, current_dataset):
            self.current_dataset = current_dataset
            return update_behavior_trace_plot(self.df_behavior_with_custom, behavior_name, regression_type)

        @app.callback(
            Output('kymograph-scatter', 'figure'),
            Input('kymograph-select-dropdown', 'value'),
            Input('neuron-select-dropdown', 'value'),
            Input('regression-type', 'value'),
            Input('trace-select-dropdown', 'value'),
            Input('dataset-dropdown', 'value')
        )
        def _update_kymograph_scatter(kymograph_segment_name, neuron_name, regression_type, trace_type,
                                      current_dataset):
            self.current_dataset = current_dataset
            df_traces = self.get_trace_type(trace_type)
            df_all = pd.concat([self.df_behavior_with_custom, self.df_curvature, df_traces], axis=1)
            return update_kymograph_scatter_plot(df_all, kymograph_segment_name, neuron_name, regression_type)

        @app.callback(
            Output('kymograph-per-segment-correlation', 'figure'),
            Input('neuron-select-dropdown', 'value'),
            Input('regression-type', 'value'),
            Input('trace-select-dropdown', 'value'),
            Input('dataset-dropdown', 'value')
        )
        def _update_kymograph_correlation(neuron_name, regression_type, trace_type, current_dataset):
            self.current_dataset = current_dataset
            df_traces = self.get_trace_type(trace_type)
            return update_kymograph_correlation_per_segment(df_traces, self.df_behavior_with_custom, self.df_curvature, neuron_name,
                                                            regression_type)

        @app.callback(
            Output('kymograph-all-neuron-max-segment-correlation', 'figure'),
            Input('regression-type', 'value'),
            Input('trace-select-dropdown', 'value'),
            Input('neuron-select-dropdown', 'value'),
            Input('kymograph-range-slider', 'value'),
            Input('dataset-dropdown', 'value')
        )
        def _update_kymograph_max_segment(regression_type, trace_type, neuron_name, kymograph_range, current_dataset):
            self.current_dataset = current_dataset
            df_traces = self.get_trace_type(trace_type)
            return update_max_correlation_over_all_segment_plot(self.df_behavior_with_custom, df_traces, self.df_curvature,
                                                                regression_type,
                                                                neuron_name, kymograph_range)

        @app.callback(
            Output('kymograph-image', 'figure'),
            Input('dataset-dropdown', 'value')
        )
        def _update_kymograph(current_dataset):
            self.current_dataset = current_dataset
            return update_kymograph(self.df_curvature)

        if self.port is None:
            port = 8050
        else:
            port = self.port
        opt = dict(debug=False, port=port)
        if self.allow_public_access:
            app.run_server(**opt, host="0.0.0.0")
        else:
            app.run_server(**opt)


def update_scatter_plot(df_behavior, df_traces, x_name, y_name, neuron_name, regression_type):
    if regression_type == 'Rectified regression':
        rev_idx = df_behavior.reversal
        y_corr_rev = df_traces.corrwith(df_behavior[y_name][rev_idx])
        y_corr_fwd = df_traces.corrwith(df_behavior[y_name][~rev_idx])
        x_corr = df_traces.corrwith(df_behavior[x_name])
        # x_corr_rev = df_traces.corrwith(df_all_time_series[x_name][rev_idx])
        # x_corr_fwd = df_traces.corrwith(df_all_time_series[x_name][~rev_idx])
        # Combine in a dataframe for plotting
        # df_dict = {'y_rev': y_corr_rev, 'y_fwd': y_corr_fwd, 'x_rev': x_corr_rev, 'x_fwd': x_corr_fwd}
        df_dict = {'y_rev': y_corr_rev, 'y_fwd': y_corr_fwd, x_name: x_corr}
        df_corr = pd.DataFrame(df_dict)
        y_names = ['y_fwd', 'y_rev']
    else:
        y_corr = df_traces.corrwith(df_behavior[y_name])
        x_corr = df_traces.corrwith(df_behavior[x_name])
        # Combine in a dataframe for plotting
        df_dict = {y_name: y_corr, x_name: x_corr}
        df_corr = pd.DataFrame(df_dict)
        y_names = y_name
    xaxis_title = f"Correlation with {x_name}"
    yaxis_title = f"Correlation with {y_name}"

    # Create a fake "selected" column, because the default doesn't work when you return a new figure
    df_corr['selected'] = 1
    df_corr.loc[neuron_name, 'selected'] = 5

    # Reset index to make x ticks smaller
    df_corr.reset_index(inplace=True)
    df_corr.rename(columns={'index': 'neuron_name'}, inplace=True)
    # Top scatter plot
    _fig = px.scatter(df_corr, x=x_name, y=y_names, hover_name="neuron_name",
                      custom_data=["neuron_name"],
                      size='selected',
                      marginal_y='histogram',
                      title="Are neurons correlated to 2 behaviors?",
                      trendline="ols").update_layout(xaxis_title=xaxis_title, yaxis_title=yaxis_title)
    # _fig.update_layout(clickmode='event+select')
    # selected_points = [int(neuron_name.split('_')[1]) - 1]
    # print(df_corr, selected_points)
    # _fig.update_traces(selectedpoints=[selected_points])
    # Half as tall
    _fig.update_layout(height=325, margin={'l': 20, 'b': 30, 'r': 10, 't': 30})
    return _fig


def update_neuron_trace_plot(df_behavior_and_neurons, neuron_name, regression_type):
    opt, px_func = switch_plot_func_using_rectification(regression_type)
    _fig = px_func(df_behavior_and_neurons, x=df_behavior_and_neurons.index, y=neuron_name,
                   title=f"Trace for {neuron_name}",
                   range_x=[0, df_behavior_and_neurons.shape[0]], **opt)
    _fig.update_layout(height=325, margin={'l': 40, 'b': 40, 't': 30, 'r': 0})
    return _fig


def update_behavior_scatter_plot(df_behavior_and_neurons, behavior_name, neuron_name, regression_type):
    if regression_type == 'Rectified regression':
        opt = {'color': 'reversal'}
    else:
        opt = {}
    _fig = px.scatter(df_behavior_and_neurons, x=behavior_name, y=neuron_name,
                      title=f"Is {neuron_name} correlated to behavior?",
                      trendline='ols', **opt)
    _fig.update_layout(showlegend=False)
    # results = px.get_trendline_results(_fig)
    # print([result.summary() for result in results.px_fit_results])
    _fig.update_layout(height=325, margin={'l': 20, 'b': 30, 'r': 10, 't': 30})
    return _fig


def update_kymograph_scatter_plot(df_all, kymograph_segment_name, neuron_name, regression_type):
    if regression_type == 'Rectified regression':
        opt = {'color': 'reversal'}
    else:
        opt = {}
    _fig = px.scatter(df_all, x=kymograph_segment_name, y=neuron_name,
                      title=f"Is {neuron_name} correlated to behavior?",
                      trendline='ols', **opt)
    _fig.update_layout(showlegend=False)
    # results = px.get_trendline_results(_fig)
    # print([result.summary() for result in results.px_fit_results])
    _fig.update_layout(height=325, margin={'l': 20, 'b': 30, 'r': 10, 't': 30})
    return _fig


def update_kymograph_correlation_per_segment(df_traces, df_behavior, df_curvature, neuron_name, regression_type):
    if regression_type == 'Rectified regression':
        rev_idx = df_behavior.reversal
        corr_rev = df_curvature.corrwith(df_traces[neuron_name][rev_idx])
        corr_fwd = df_curvature.corrwith(df_traces[neuron_name][~rev_idx])
        # Combine in a dataframe for plotting
        df_dict = {'rev': corr_rev, 'fwd': corr_fwd}
        df_corr = pd.DataFrame(df_dict)
        y_names = ['fwd', 'rev']
    else:
        corr = df_curvature.corrwith(df_traces[neuron_name])
        # Combine in a dataframe for plotting
        df_dict = {'correlation': corr}
        df_corr = pd.DataFrame(df_dict)
        y_names = ['correlation']

    df_corr.reset_index(inplace=True)
    df_corr.rename(columns={'index': 'neuron_name'}, inplace=True)
    _fig = px.line(df_corr, y=y_names, title=f"Which body segment correlates to {neuron_name}?", range_y=[-0.8, 0.8],
                   hover_name='neuron_name', custom_data=['neuron_name'])
    _fig.update_layout(showlegend=False)
    # results = px.get_trendline_results(_fig)
    # print([result.summary() for result in results.px_fit_results])
    _fig.update_layout(height=325, margin={'l': 20, 'b': 30, 'r': 10, 't': 30})
    return _fig


def update_max_correlation_over_all_segment_plot(df_behavior, df_traces, df_curvature, regression_type,
                                                 neuron_name, kymograph_range):
    # Will not actually be updated, except for changing the rectification
    df_curvature_subset = df_curvature.iloc[:, kymograph_range[0]:kymograph_range[1]]
    if regression_type == 'Rectified regression':
        rev_idx = df_behavior.reversal
        df_corr = _correlate_return_cross_terms(df_traces[rev_idx], df_curvature_subset)
        df_max_rev = df_corr.abs().max(axis=1)

        df_corr = _correlate_return_cross_terms(df_traces[~rev_idx], df_curvature_subset)
        df_max_fwd = df_corr.abs().max(axis=1)

        df_dict = {'rev': df_max_rev, 'fwd': df_max_fwd}
        df_corr_max = pd.DataFrame(df_dict)
        y_names = ['fwd', 'rev']
    else:
        df_corr = _correlate_return_cross_terms(df_traces, df_curvature_subset)
        df_corr_max = pd.DataFrame(df_corr.max(axis=1), columns=['correlation'])
        y_names = 'correlation'
    # For setting custom data

    # Create a fake "selected" column, because the default doesn't work when you return a new figure
    df_corr_max['selected'] = 1
    df_corr_max.loc[neuron_name, 'selected'] = 5

    # Shrink x ticks
    df_corr_max.reset_index(inplace=True)
    df_corr_max.rename(columns={'index': 'neuron_name'}, inplace=True)

    _fig = px.scatter(df_corr_max, y=y_names, title=f"Max curvature correlation over selected body segments",
                      range_y=[0, 0.8],
                      size='selected',
                      hover_name='neuron_name',
                      marginal_y='histogram', custom_data=['neuron_name']).update_traces()
    _fig.update_layout(height=325, margin={'l': 20, 'b': 30, 'r': 10, 't': 30})
    return _fig


def update_behavior_trace_plot(df_behavior, behavior_name, regression_type):
    opt, px_func = switch_plot_func_using_rectification(regression_type)
    _fig = px_func(df_behavior, x=df_behavior.index, y=behavior_name,
                   range_x=[0, len(df_behavior)],
                   title=f"Trace of {behavior_name}", **opt)
    _fig.update_layout(height=325, margin={'l': 20, 'b': 30, 'r': 10, 't': 30})
    return _fig


def switch_plot_func_using_rectification(regression_type):
    if regression_type == 'Rectified regression':
        opt = {'color': 'reversal'}
        px_func = px.scatter
    else:
        opt = {}
        px_func = px.line
    return opt, px_func


def build_dropdowns(behavior_names: list, curvature_names: list, trace_names: list, neuron_names: list,
                    dataset_names: list) -> html.Div:
    curvature_initial = curvature_names[0]
    trace_initial = 'ratio'
    neuron_initial = neuron_names[0]
    dataset_initial = dataset_names[0]
    x_initial = 'signed_speed_angular'
    behavior_initial = 'signed_middle_body_speed'

    header = html.H1("Dropdowns for changing all plots")

    dropdown_style = {'width': '20%'}
    dropdowns = html.Div([
        html.Div([
            html.Label(["Dataset"], style={'font-weight': 'bold', "text-align": "center"}),
            html.Div([
                dcc.Dropdown(
                    dataset_names,
                    dataset_initial,
                    id='dataset-dropdown',
                    clearable=False
                ),
            ])],
            style=dropdown_style),

        html.Div([
            html.Label(["Behavior to correlate (x axis)"], style={'font-weight': 'bold', "text-align": "center"}),
            html.Div([
                dcc.Dropdown(
                    behavior_names,
                    x_initial,
                    id='scatter-xaxis',
                    clearable=False
                ),
            ])],
            style=dropdown_style),

        html.Div([
            html.Label(["Behavior to show and correlate (y axis)"], style={'font-weight': 'bold', "text-align": "center"}),
            html.Div([
                dcc.Dropdown(
                    behavior_names,
                    behavior_initial,
                    id='behavior-scatter-yaxis',
                    clearable=False
                ),
            ])],
            style=dropdown_style),

        html.Div([
            html.Label(["Select neuron"], style={'font-weight': 'bold', "text-align": "center"}),
            html.Div([
                dcc.Dropdown(
                    neuron_names,
                    neuron_initial,
                    id='neuron-select-dropdown',
                    clearable=False
                ),
            ])],
            style=dropdown_style),

        html.Div([
            html.Label(["Select kymograph segment"], style={'font-weight': 'bold', "text-align": "center"}),
            html.Div([
                dcc.Dropdown(
                    curvature_names,
                    curvature_initial,
                    id='kymograph-select-dropdown',
                    clearable=False
                ),
            ])],
            style=dropdown_style),

        html.Div([
            html.Label(["Select trace type"], style={'font-weight': 'bold', "text-align": "center"}),
            html.Div([
                dcc.Dropdown(
                    trace_names,
                    trace_initial,
                    id='trace-select-dropdown',
                    clearable=False
                ),
            ])],
            style=dropdown_style)
        ], style={'display': 'flex', 'padding': '10px 5px'})

    # return html.Div(dropdowns, style={'display': 'flex', 'padding': '10px 5px'})
    return html.Div([header, dropdowns])


def build_second_row_options(path_to_grid_plot) -> html.Div:

    row_style = {'display': 'inline-block', 'width': '50%'}

    return html.Div([
        html.Div([
            html.Label(['Style of regression line'], style={'font-weight': 'bold', "text-align": "center"}),
            dcc.RadioItems(
                ['Overall regression', 'Rectified regression'],
                'Overall regression',
                id='regression-type'),
        ], style=row_style),

        html.Div([
            html.Label(['Range for kymograph max plot'], style={'font-weight': 'bold', "text-align": "center"}),
            dcc.RangeSlider(0, 99, 3, value=[5, 95], id='kymograph-range-slider', allowCross=False)
        ], style=row_style),

        # html.Div([
        #     html.A("Link to grid plot", href=path_to_grid_plot, target="_blank")
        # ])
        ]
    )


def build_plots_html() -> html.Div:
    # Second trace plot, which is actually initialized through a clickData field on the scatterplot
    initial_neuron_clickData = {'points': [{'customdata': ['neuron_001']}]}

    top_header = html.H2("Summary plots (some interactive)")

    top_row_style = {'width': '20%', 'display': 'inline-block'}
    top_row = html.Div([
        html.Div([dcc.Graph(id='correlation-scatterplot', clickData=initial_neuron_clickData)], style=top_row_style),
        html.Div([dcc.Graph(id='trace-and-behavior-scatterplot')], style=top_row_style),
        html.Div([dcc.Graph(id='kymograph-scatter')], style=top_row_style),
        html.Div([dcc.Graph(id='kymograph-per-segment-correlation')], style=top_row_style),
        html.Div([dcc.Graph(id='kymograph-all-neuron-max-segment-correlation',
                            clickData=initial_neuron_clickData)], style=top_row_style)
    ], style={'width': '100%', 'display': 'inline-block'})

    time_series_header = html.H2("Time Series plots")
    time_series_rows = html.Div([
        dcc.Graph(id='neuron-trace'),
        dcc.Graph(id='behavior-trace')
    ], style={'width': '100%', 'display': 'inline-block'}
    )

    return html.Div([top_header, top_row, time_series_header, time_series_rows])


def build_kymograph(df_curvature) -> html.Div:
    fig = update_kymograph(df_curvature)

    image = html.Div([
        dcc.Graph(id='kymograph-image', figure=fig)
    ], style={'width': '100%', 'display': 'inline-block'})
    return image


def update_kymograph(df_curvature):
    fig = px.imshow(df_curvature.T, zmin=-0.05, zmax=0.05, aspect=3, color_continuous_scale='RdBu')
    return fig


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Build GUI with a project')
    parser.add_argument('--project_path', '-p', default=None,
                        help='path to config file')
    parser.add_argument('--allow_public_access', default=False,
                        help='allow access using the intranet (NOT SECURE)')
    parser.add_argument('--port', default=None,
                        help='port')
    parser.add_argument('--DEBUG', default=False, help='')
    args = parser.parse_args()
    project_path = args.project_path
    port = args.port
    allow_public_access = args.allow_public_access
    allow_public_access = True if allow_public_access == "True" else False
    DEBUG = args.DEBUG

    # DATA_FOLDER = "/home/charles/Current_work/repos/dlc_for_wbfm/wbfm/notebooks/alternative_ideas/tmp_data"
    # project_path = "/scratch/neurobiology/zimmer/fieseler/wbfm_projects/2022-11-27_spacer_7b_2per_agar/ZIM2165_Gcamp7b_worm1-2022_11_28/project_config.yaml"
    dashboard = DashboardDataset(project_path, port=port, allow_public_access=allow_public_access)
    dashboard.serve_wbfm_dashboard()
