library(mlflow)
library(tidyverse)
library(tidymodels)
library(carrier)
library(ranger)
library(yardstick)
library(jsonlite)

with(mlflow_start_run(), {
  # Declare parameters
  path_train_data <- mlflow_param("path_train_data", type = "string")
  path_validation_data <- mlflow_param("path_validation_data", type = "string")
  path_recipe <- mlflow_param("path_recipe", type = "string")
  n_trees <- mlflow_param("n_trees", 100)
  
  
  # Read data
  train <- read.csv(path_train_data, stringsAsFactors = TRUE) 
  validation <- read.csv(path_validation_data, stringsAsFactors = TRUE) 
  
  # Bake data
  data_recipe <- readRDS(path_recipe)
  train_preprocessed <- data_recipe %>% bake(train)
  validation_preprocessed <- data_recipe %>% bake(validation)
  
  
  rf_fit <- rand_forest(trees = n_trees, mode = "regression") %>%
    set_engine("ranger") %>%
    fit(Sale_Price ~ ., data = train_preprocessed)
  
  
  validation_test <- validation_preprocessed %>% 
    bind_cols(predict(rf_fit, validation_preprocessed))
  val_metrics <- metrics(data = validation_test, 
          truth = Sale_Price, 
          estimate = .pred)

  
  predictor <- crate(~ ranger:::predict.ranger(!!rf_fit$fit, as.data.frame(.x))[[1]])
  mlflow_log_param("n_trees", n_trees)
  mlflow_log_metric("rmse", filter(val_metrics, .metric == 'rmse') %>% pull(.estimate))
  mlflow_log_metric("r2", filter(val_metrics, .metric == 'rsq') %>% pull(.estimate))
  mlflow_log_metric("mae", filter(val_metrics, .metric == 'mae') %>% pull(.estimate))
  
  # mlflow_log_artifact(rf_fit, "rf_model")
  mlflow_log_model(predictor, "model")
}
)



# tflow <-
#   ames %>%
#   tidyflow(seed = 123) %>%
#   plug_recipe(~ recipe(Sale_Price ~ ., data = .x) %>% step_scale(all_numeric(), -Sale_Price)) %>%
#   plug_split(initial_split, prop = 0.8) %>%
#   plug_resample(vfold_cv, v = 5) %>%
#   plug_grid(expand.grid, trees = seq.int(100, 1200, 200)) %>%
#   plug_model(rand_forest(mode = "regression", trees = tune()) %>% set_engine("ranger"))
# 
# final_model <-
#   tflow %>%
#   fit() %>%
#   complete_tflow(metric = "rmse")
# 
# 
# final_model %>% 
#   pull_tflow_fit_tuning() %>%
#   autoplot()
# 
# 
# final_model %>% 
#   pull_tflow_fit()