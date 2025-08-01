from wbfm.utils.projects.finished_project_data import ProjectData
from wbfm.utils.general.high_performance_pandas import get_names_from_df
from wbfm.utils.neuron_matching.utils_candidate_matches import rename_columns_using_matching
import pandas as pd
import numpy as np
from wbfm.utils.performance.comparing_ground_truth import calculate_accuracy_from_dataframes, calc_accuracy_of_pipeline_steps



def calculate_accuracy(df_gt, df_pred):
    # Align both DataFrames on the same rows and columns
    common_index = df_gt.index.intersection(df_pred.index)
    common_columns = df_gt.columns.intersection(df_pred.columns)
    
    df_gt = df_gt.loc[common_index, common_columns]
    df_pred = df_pred.loc[common_index, common_columns]

    # Conditions:
    # Ground truth has a value
    gt_valid = ~df_gt.isna()

    # Miss: ground truth valid, prediction NaN
    misses = (gt_valid) & (df_pred.isna())

    # Mismatch: both valid but values differ
    mismatches = (gt_valid) & (~df_pred.isna()) & (df_gt != df_pred)

    # Count
    total_misses = misses.sum().sum()
    total_mismatches = mismatches.sum().sum()
    total_gt_detections = gt_valid.sum().sum()

    accuracy = 1 - (total_misses + total_mismatches) / total_gt_detections

    return {
        "misses": int(total_misses),
        "mismatches": int(total_mismatches),
        "total_ground_truth": int(total_gt_detections),
        "accuracy": accuracy
    }


print("Loading data ...")
fname_gt = "/lisc/scratch/neurobiology/zimmer/fieseler/wbfm_projects_future/flavell_data/images_for_charlie/flavell_data.nwb"
project_data_gt = ProjectData.load_final_project_data(fname_gt)

fname_res = "/lisc/scratch/neurobiology/zimmer/schwartz/traces_mit_debug_embed/2025_07_01trial_28/project_config.yaml"
project_data_res = ProjectData.load_final_project_data(fname_res)

print("Calculating accuracy ...")


df_res = project_data_res.final_tracks
df_gt = project_data_gt.final_tracks
nan_rows = pd.DataFrame(np.nan, index=[len(df_res), len(df_res)+1], columns=df_res.columns)

df_res = pd.concat([df_res, nan_rows])

df_res_renamed, _ , _ , _ = rename_columns_using_matching(df_gt, df_res, column='raw_segmentation_id', try_to_fix_inf=True)

df_res_renamed = df_res_renamed.xs('raw_segmentation_id', axis=1, level=1)
df_gt = df_gt.xs('raw_segmentation_id', axis=1, level=1)

print(calculate_accuracy(df_gt, df_res_renamed))
