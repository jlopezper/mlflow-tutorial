library(mlflow)
library(dplyr)
library(tidymodels)
library(readr)
library(purrr)

set.seed(123)


# Create temporary directories to write data
dir.create(path = 'data')
dir.create(path = 'recipe')
data("ames")

# ames <- ames %>% select(Lot_Frontage, Year_Built, Full_Bath, Garage_Cars, Year_Sold, Sale_Price, 
#                         Bedroom_AbvGr, Kitchen_AbvGr,Street)

with(mlflow_start_run(), {
  # Log some info
  mlflow_log_param("Task", "Create dataset")
  
  # Split the data into training and validation sets. (0.8, 0.2) split.
  iris_split <- initial_split(ames, prop = 0.9)
  train <- training(iris_split)
  validation <- testing(iris_split)
  data_recipe <- train %>% 
    recipe(Sale_Price ~.) %>% 
    step_corr(all_numeric()) %>%
    step_center(all_numeric(), -all_outcomes()) %>%
    step_scale(all_numeric(), -all_outcomes()) %>% 
    prep()
    
  # train <- ames[sampled, ]
  # validation <- ames[-sampled, ]
  
  # Save datasets
  write_csv(x = train, file.path('data', 'train.csv'))
  write_csv(x = validation, file.path('data', 'validation.csv'))
  # Save recipe
  saveRDS(data_recipe, file = file.path('recipe', 'recipe.rds'))
  
  # Log processed datasets
  mlflow_log_artifact('data', artifact_path = "data")
  mlflow_log_artifact('recipe', artifact_path = "recipe")
})