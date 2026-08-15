# Intelligent Ride Fare Estimation & Predictive Analytics using Machine Learning

An end-to-end machine learning project for **predicting ride fares using geospatial, temporal, and trip-level features**. The project explores the relationship between trip characteristics and fare amounts while comparing multiple regression techniques.

---

##  Project Overview

Accurately estimating ride fares is an important component of ride-hailing platforms. Fare amounts can depend on several factors such as:

* 📍 Pickup and drop-off locations
* 🛣️ Trip distance
* 👥 Passenger count
* 🕐 Pickup time
* 📅 Day and month
* 📊 Temporal patterns

This project develops a complete **fare prediction pipeline**, starting from raw Uber trip data and progressing through data cleaning, exploratory data analysis, feature engineering, dimensionality reduction, regression modeling, and model evaluation.

The primary objective is to build and compare different regression approaches for predicting the **`fare_amount`** of a ride.

---

##  Objectives

* Clean and preprocess raw Uber trip data
* Handle missing values, duplicates, and invalid geographic coordinates
* Extract meaningful temporal features from pickup timestamps
* Calculate geographical trip distance from pickup and drop-off coordinates
* Detect and remove outliers using the IQR method
* Analyze relationships between predictors and fare amount
* Apply feature selection techniques
* Investigate dimensionality reduction using PCA
* Compare multiple regression algorithms
* Evaluate models using standard regression metrics

---

##  Dataset

The project uses the **Uber Fare Dataset (`uber.csv`)**.

### Target Variable

`fare_amount`

### Major Features

| Feature             | Description             |
| ------------------- | ----------------------- |
| `fare_amount`       | Target ride fare        |
| `passenger_count`   | Number of passengers    |
| `pickup_latitude`   | Pickup latitude         |
| `pickup_longitude`  | Pickup longitude        |
| `dropoff_latitude`  | Drop-off latitude       |
| `dropoff_longitude` | Drop-off longitude      |
| `pickup_datetime`   | Date and time of pickup |

The raw geographic coordinates are subsequently transformed into a **Distance** feature.

---

##  Project Workflow

```text
Raw Uber Dataset
       │
       ▼
Data Cleaning
       │
       ├── Missing Value Handling
       ├── Duplicate Removal
       └── Geographic Validation
       │
       ▼
Feature Engineering
       │
       ├── Trip Distance
       ├── Year
       ├── Month
       ├── Weekday
       ├── Hour
       ├── Monthly Quarter
       └── Hourly Segment
       │
       ▼
Exploratory Data Analysis
       │
       ├── Distribution Analysis
       ├── Boxplots
       ├── Categorical Analysis
       └── Correlation Analysis
       │
       ▼
Train-Test Split
       │
       ▼
Outlier Treatment
       │
       ▼
Feature Encoding & Standardization
       │
       ▼
Feature Selection
       │
       ├── VIF
       └── RFE
       │
       ▼
Dimensionality Reduction
       │
       └── PCA
       │
       ▼
Regression Models
       │
       ├── Multiple Linear Regression
       ├── Ridge Regression
       ├── Lasso Regression
       ├── Elastic Net
       └── Polynomial Regression
       │
       ▼
Model Evaluation & Comparison
```

---

##  Data Preprocessing

Several preprocessing steps were performed before modeling.

### 1. Missing Value Handling

Rows containing missing values were removed to obtain a clean modeling dataset.

### 2. Geographic Validation

Latitude and longitude values were checked to ensure they fall within valid geographic ranges.

### 3. Duplicate Removal

Duplicate observations were identified and removed from the dataset.

### 4. Temporal Feature Engineering

`pickup_datetime` was converted into several useful features:

* Year
* Month
* Weekday
* Hour
* Monthly Quarter
* Hourly Segment

This allows the model to capture temporal patterns in ride fares.

---

##  Geospatial Feature Engineering

One of the key features engineered in the project is **trip distance**.

The pickup and drop-off latitude/longitude coordinates were used to calculate geographical distance using `geopy`.

```python
Distance =
Geographical distance between
Pickup Location → Drop-off Location
```

The original latitude and longitude variables are then removed from the final modeling dataset to avoid retaining redundant raw coordinate features.

---

##  Exploratory Data Analysis

The dataset was explored using:

* Target variable distribution
* Numerical feature distributions
* Boxplots for outlier detection
* Categorical feature visualization
* Correlation matrix
* Descriptive statistics

These analyses were used to understand the structure of the dataset before model development.

---

##  Outlier Detection

Outliers were treated using the **Interquartile Range (IQR)** method.

For each numerical feature:

```text
IQR = Q3 - Q1

Lower Bound = Q1 - 1.5 × IQR
Upper Bound = Q3 + 1.5 × IQR
```

Importantly, the revised pipeline calculates the IQR boundaries using the **training data only** and applies those boundaries to both training and testing data.

This helps prevent test-set information from influencing preprocessing.

---

##  Feature Encoding & Standardization

Categorical variables were converted into numerical representations using:

* One-Hot Encoding
* Dummy Encoding

Numerical predictors were standardized using `StandardScaler`.

The scaler was fitted only on the training data and subsequently applied to the test data.

---

##  Feature Selection

Two feature-selection approaches were explored:

### Variance Inflation Factor (VIF)

VIF was used to investigate potential multicollinearity between predictor variables.

### Recursive Feature Elimination (RFE)

RFE was used with Linear Regression to identify a reduced subset of potentially useful features.

Feature-count experiments were performed to understand the relationship between model complexity and prediction error.

---

##  Principal Component Analysis

**Principal Component Analysis (PCA)** was investigated as a dimensionality-reduction technique.

The explained variance of principal components was analyzed to understand how much information could be retained with fewer dimensions.

The PCA transformation was fitted on the training data and then applied to the test data.

---

##  Machine Learning Models

Five regression approaches were implemented and compared:

### 1. Multiple Linear Regression

Provides a baseline linear relationship between the engineered predictors and fare amount.

### 2. Ridge Regression

Adds L2 regularization to reduce the impact of potentially correlated predictors.

### 3. Lasso Regression

Uses L1 regularization and can shrink coefficients toward zero, providing a form of feature selection.

### 4. Elastic Net Regression

Combines L1 and L2 regularization.

### 5. Polynomial Regression

Polynomial features were explored to capture potential nonlinear relationships between predictors and fare amount.

---

##  Model Evaluation

Models were evaluated using:

### R² Score

Measures the proportion of variance in the target explained by the model.

**Higher is better.**

### Mean Squared Error (MSE)

Measures the average squared prediction error.

**Lower is better.**

### Root Mean Squared Error (RMSE)

The square root of MSE, expressed in the same units as the target variable.

**Lower is better.**

### Residual Sum of Squares (RSS)

Measures the total squared prediction error.

**Lower is better.**

Both training and testing performance were compared to identify differences in model fit and generalization.

---

##  Models Compared

| Model                      | Regularization / Transformation | Purpose                            |
| -------------------------- | ------------------------------- | ---------------------------------- |
| Multiple Linear Regression | None                            | Baseline linear model              |
| Ridge Regression           | L2                              | Handle correlated predictors       |
| Lasso Regression           | L1                              | Regularization + feature selection |
| Elastic Net                | L1 + L2                         | Combined regularization            |
| Polynomial Regression      | Polynomial features             | Capture nonlinear relationships    |

---

##  Technologies Used

* **Python**
* **Pandas** — Data manipulation
* **NumPy** — Numerical computing
* **Matplotlib** — Visualization
* **Seaborn** — Statistical visualization
* **Scikit-learn** — Machine learning
* **Statsmodels** — Statistical modeling and VIF analysis
* **GeoPy** — Geographic distance calculation
* **Google Colab** — Development environment

---

##  Repository Structure

```text
Uber-Fare-Intelligence/
│
├── Uber_Fare_Colab.ipynb
├── uber.csv
└── README.md
```

> `uber.csv` is included only if permitted by the dataset's redistribution terms. Otherwise, download the dataset separately and place it in the project directory.

---

##  How to Run

### 1. Clone the repository

```bash
git clone https://github.com/your-username/your-repository-name.git
cd your-repository-name
```

### 2. Install dependencies

```bash
pip install numpy pandas matplotlib seaborn scikit-learn statsmodels geopy tqdm
```

### 3. Launch the notebook

```bash
jupyter notebook Uber_Fare_Colab.ipynb
```

Alternatively, open the notebook directly in **Google Colab**.

### 4. Provide the dataset

Place:

```text
uber.csv
```

in the working directory, or upload it when prompted by the notebook.

---

##  Methodological Note

The notebook includes exploratory VIF, RFE, and PCA scans where test-set performance is visualized to help determine feature/component choices.

For a strictly production-grade evaluation pipeline, feature selection and hyperparameter decisions should instead be made using a **training/validation split or cross-validation**, with the final test set used only once for unbiased performance estimation.

---

##  Key Takeaways

This project demonstrates an end-to-end approach to a real-world regression problem:

**Raw Data → Cleaning → Feature Engineering → EDA → Outlier Treatment → Feature Selection → Dimensionality Reduction → Regression → Evaluation**

The project particularly focuses on combining **geospatial and temporal feature engineering** with classical machine learning regression techniques to estimate ride fares.

---

##  Author

**Diksha**

M.Sc. Industrial Engineering & Operations Research
Indian Institute of Technology Bombay

