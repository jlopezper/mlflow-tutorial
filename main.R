library(mlflow)
library(dplyr)
library(tidymodels)
library(tidyflow)
library(tidyverse)

mlflow_create_experiment(name = 'ames_housing')

mlflow_run(entry_point = 'data-preparation.R', no_conda = TRUE)

run_infos <- mlflow_list_run_infos(experiment_id = "0")
run_infos
data_prep_run_id <- run_infos %>% 
  filter(status == "FINISHED") %>% 
  arrange(start_time) %>% 
  tail(1) %>% 
  pull(run_uuid)


path_train_data <- mlflow_download_artifacts("data/train.csv", run_id = data_prep_run_id)
path_validation_data <- mlflow_download_artifacts("data/validation.csv", run_id = data_prep_run_id)
path_recipe <- mlflow_download_artifacts("recipe/recipe.rds", run_id = data_prep_run_id)


for (t in seq.int(100, 600, 200)) {
  mlflow_run(entry_point = "train.R", parameters = list(
    path_train_data = path_train_data,
    path_validation_data = path_validation_data,
    path_recipe = path_recipe,
    n_trees = t
  ), no_conda = TRUE)
}



id_best_model <- map_dfr(mlflow_list_run_infos(experiment_id = "0")$run_uuid,
    function(x) {
      mlflow_get_run(run_id = x)
    }) %>% 
  unnest(metrics) %>% 
  filter(key == 'mae') %>% 
  filter(value == min(value)) %>% 
  pull(run_uuid)


mlflow_rfunc_serve(model_uri = paste0("mlruns/0/", id_best_model ,"/artifacts/model/"),
                   restore = TRUE)