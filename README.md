# Sales Forecasting

A project for forecasting future sales using historical data, feature engineering, and machine learning / time series methods.

---

## Project Overview

This repository implements a pipeline to forecast sales. It includes data preprocessing, exploratory analysis, feature engineering, model training, and evaluation. The goal is to provide accurate predictions to support decision-making (e.g. inventory planning, demand forecasting).  

The project is organized into:  
- `notebooks/` — exploratory data analysis, experimentation, model development  
- `src/` — production code / modules for preprocessing, modeling, evaluation  
- `environment.yml` — conda environment specification  




---

## Installation & Setup

Here’s how to get the project running locally:

1. **Clone this repository**  
   ```bash
   git clone https://github.com/frsth/sales-forecasting.git
   cd sales-forecasting

2. **Create & activate environment**
If you use conda:

conda env create -f environment.yml
conda activate forecasting_env
