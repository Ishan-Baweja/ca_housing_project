# California Housing Project

This repository contains an end-to-end machine learning workflow using the **California housing dataset**. The project demonstrates best practices in organizing machine learning work, including separation of raw and processed data, structured notebooks for analysis, reproducible preprocessing pipelines, and modular notebooks for model development.

The dataset originates from the 1990 California census and includes features such as median income, housing age, number of rooms, latitude, longitude, and median house value. The goal of this project is to predict housing values based on these attributes using a variety of supervised learning models.

---

## Repository Structure


---

## Workflow Overview

The project is structured into **four main stages**:

### 1. Initial Data Analysis (IDA)
- Load `housing.csv` (raw data).  
- Inspect data types, missing values, and descriptive statistics.  
- Implement stratified train/test split based on income categories.  
- Save splits into `/data/train/housing_train.csv` and `/data/test/housing_test.csv`.  

### 2. Exploratory Data Analysis (EDA)
- Visualize geographic and statistical distributions of housing attributes.  
- Identify correlations with median house value.  
- Engineer new features (e.g., `rooms_per_household`, `bedrooms_per_room`, `population_per_household`).  
- Save processed training dataset with 24 features to `/data/train/housing_train_processed.csv`.  
- Export plots to `/images` for reference.  

### 3. Preprocessing Pipeline
- Implemented in `preprocessing_pipeline.py`.  
- Handles missing values, scaling, categorical encoding, and feature engineering.  
- Transforms raw training/test splits into the processed form used by all models.  
- Ensures reproducibility and consistency across experiments.  

### 4. Model Development
Each model notebook (in `/models`) follows a consistent structure:
1. **Data Loading**: Import processed training set.  
2. **Model Fitting**: Train model on processed features.  
3. **Cross-Validation**: Evaluate with k-fold CV.  
4. **Hyperparameter Tuning**: Use `GridSearchCV` or `RandomizedSearchCV`.  
5. **Model Saving**: Save trained model object into `/models` folder.  

The models explored include:
- Linear Regression  
- Decision Tree Regressor  
- Random Forest Regressor  
- Support Vector Regressor (SVR)  

---

## Dependencies

- Python 3.9+  
- pandas  
- numpy  
- scikit-learn  
- matplotlib  
- seaborn  
- joblib  

Install requirements with:  

```bash
pip install -r requirements.txt
