# Titanic Survival Prediction Pipeline

Welcome to the **Titanic Survival Prediction** project. This repository presents a robust and scalable machine learning pipeline designed to predict passenger survival on the Titanic. By blending custom feature engineering, advanced preprocessing techniques, and a powerful RandomForest model, this project serves as a comprehensive example of end-to-end machine learning in action.

## Getting Started

### Docker Deployment

To ensure consistency across environments, the project is fully containerized.

1. **Pull the Docker Image:**

   ```bash
   docker pull santhoshnagaraj0123/ml-pipeline:latest
   ```

2. **Run the Docker Container:**

   ```bash
   docker run santhoshnagaraj0123/ml-pipeline:latest
   ```

### Local Environment Setup

For those who prefer running the project locally:

1. **Clone the Repository:**

   ```bash
   git clone https://github.com/Santhosh143N/ML_Pipeline.git
   cd ML_Pipeline
   ```

2. **Install Dependencies:**

   ```bash
   pip install -r requirements.txt
   ```

3. **Execute the Pipeline:**

   ```bash
   python app.py
   ```

## Pipeline Overview

### Feature Engineering

- **Age Imputation:** Missing values in the `Age` column are filled using the mean.
- **Child Feature:** Adds an `IsChild` feature based on age.
- **Scaling:** Scales `Age` and `Fare` for improved model performance.
- **Encoding:** Transforms the `Sex` column into a binary format.

### Model

- **Algorithm:** RandomForestClassifier, selected for its balance between interpretability and predictive power.
- **Evaluation Metrics:** The model is rigorously evaluated using accuracy, precision, recall, and cross-validation scores.

## Dockerfile Insights

- **Base Image:** Python 3.9
- **Workdir:** `/app` directory for all operations.
- **Command:** Executes `app.py` upon container startup.

## Dependencies

Ensure the following libraries are included in your `requirements.txt`:

```
scikit-learn
mlflow
```
