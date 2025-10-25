# Fashion Forward Forecasting

## Project Overview
This project was developed as part of the Udacity Data Scientist II for Porsche Nanodegree.  
The goal is to build a machine learning pipeline that predicts whether a customer would recommend a product based on their review text, age, product category, and other features or not.

The dataset is from an online women's clothing retailer called *StyleSense*.  
Many customer reviews are missing the "Recommended" label, but they still contain valuable information in the text.  
The objective of this project was to train a predictive model that can automatically infer the recommendation label from the available data.

By automating this task, the company can better understand customer satisfaction, detect trends in product feedback, and support future business and product decisions.


## Data
The dataset (`reviews.csv`) contains several types of data:
- **Text data**: `Review Text`, and optionally `Title`
- **Numerical data**: `Age`, `Positive Feedback Count`
- **Categorical data**: `Division Name`, `Department Name`, `Class Name`, `Clothing ID`
- **Target variable**: `Recommended IND` (0 = not recommended, 1 = recommended)

Before training, the data was cleaned and split into training and test sets (90/10 split, stratified by the target label).  
Missing values were handled separately for numeric and categorical columns.  


## Pipeline Design
A complete end-to-end machine learning pipeline was built using **scikit-learn**.  
The pipeline includes preprocessing for different data types and a classification model.

### 1 - Preprocessing
- **Text**: TF-IDF Vectorizer with English stopwords, 1-2 n-grams, and limits for rare and frequent terms  
- **Numerical**: Median imputation and standard scaling  
- **Categorical**: Mode imputation and one-hot encoding  

These steps were combined using a `ColumnTransformer`, ensuring that preprocessing and model training happen in one consistent workflow.

### 2 - Model
- **Classifier**: Logistic Regression  
- **Class imbalance handling**: `class_weight='balanced'`  
- **Training**: Performed with cross-validation to ensure stable results  

### 3 - Fine-Tuning
Hyperparameter tuning was performed using **GridSearchCV** with 5-fold cross-validation.  
The search included both TF-IDF parameters (e.g., `ngram_range`, `min_df`, `max_df`) and model parameters (e.g., `C`, `solver`).  
The scoring metric was **F1-macro**, to give equal importance to both classes.


## Model Evaluation and Results
After fine-tuning, the final pipeline achieved an **accuracy of 0.879** on the test data.  
For the positive class ("Recommended"), the model reached a **precision of 0.944**, a **recall of 0.906**, and an **F1-score of 0.925**.

Compared to the baseline model, these results show a small but clear improvement in both overall accuracy and the balance between classes.  
The optimized TF-IDF representation and adjusted model parameters helped the pipeline recognize negative reviews more accurately, while still keeping a very strong performance on positive reviews.

In practical terms, the model correctly predicts around 9 out of 10 customer recommendations based on the review text and other available features.  
This provides a reliable foundation for automated sentiment prediction and supports data-driven business insights.


## Technologies Used
- Python 3.11  
- scikit-learn  
- pandas, numpy  
- matplotlib (for visualization)  
- Jupyter Notebook (VS-Code)


## Repository Structure
- **data/raw**
  - `reviews.csv`
- **notebooks/**
  - `project2_fashion_forward_forecasting.ipynb`
- **models/**
- **README.md**
- **.gitignore**

Author
Johannes Golling
Data Science II for Porsche Nanodegree – Udacity
GitHub: johannesgolling-web
