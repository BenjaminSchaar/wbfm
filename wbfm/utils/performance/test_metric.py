from wbfm.utils.projects.finished_project_data import ProjectData


from wbfm.utils.performance.comparing_ground_truth import calculate_accuracy_from_dataframes, calc_accuracy_of_pipeline_steps



fname_gt = "/lisc/scratch/neurobiology/zimmer/fieseler/wbfm_projects_future/flavell_data/images_for_charlie/flavell_data.nwb"
project_data_gt = ProjectData.load_final_project_data_from_config(fname_gt)

fname_res = "/lisc/scratch/neurobiology/zimmer/schwartz/traces_mit_debug_embed/2025_07_01trial_28/project_config.yaml"
project_data_res = ProjectData.load_final_project_data_from_config(fname_res)




df_res = project_data_gt.final_tracks
df_gt = project_data_res.final_tracks

print(df_res.head())
print(df_gt.head())
opt = dict(column_names=['raw_segmentation_id'])

#df_acc_pipeline = calculate_accuracy_from_dataframes(df_gt, df_res, **opt)
#print(df_acc_pipeline)

