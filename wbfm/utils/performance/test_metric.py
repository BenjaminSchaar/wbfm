from wbfm.utils.projects.finished_project_data import ProjectData


from wbfm.utils.performance.comparing_ground_truth import calculate_accuracy_from_dataframes, calc_accuracy_of_pipeline_steps



fname = "/lisc/scratch/neurobiology/zimmer/schwartz/traces_mit_debug_embed/2025_07_01trial_28/project_config.yaml"
project_data_gcamp = ProjectData.load_final_project_data_from_config(fname)



neuron_names = project_data_gcamp.finished_neuron_names()

df_pipeline = project_data_gcamp.initial_pipeline_tracks[neuron_names]
df_gt = project_data_gcamp.final_tracks[neuron_names]

print(df_pipeline.head())
print(df_gt.head())
#opt = dict(column_names=['raw_neuron_ind_in_list'])

#df_acc_pipeline = calculate_accuracy_from_dataframes(df_gt, df_pipeline, **opt)

