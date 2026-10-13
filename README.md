# HealthTrack AI: Predictive Patient Management

HealthTrack AI is an ML-powered healthcare analytics project designed to help predict patient health outcomes and support better clinical decision-making and healthcare resource planning. It processes structured patient data, trains a classification model, and exposes it through a FastAPI-based prediction service.

This repository demonstrates a full ML lifecycle in a compact and interview-friendly format: data ingestion, preprocessing, feature engineering, model training, evaluation, artifact persistence, DVC-based reproducibility, and API deployment.

## 1. Project Overview

### Problem Statement
In healthcare environments, patients often have complex and time-varying health conditions, and clinicians need actionable guidance on:
- who may require attention sooner,
- which patients may be at risk of complications,
- how to allocate resources effectively, and
- which treatments or interventions are more likely to be beneficial.

HealthTrack AI addresses this by using historical healthcare data and machine learning to predict patient health status based on features such as:
- age,
- gender,
- blood type,
- medical condition,
- billing amount,
- room number,
- medication history.

### Objective
The goal of this project is to build a predictive healthcare model that:
- learns from patient records,
- generalizes to unseen data,
- transforms data using reproducible preprocessing pipelines,
- stores trained artifacts efficiently,
- exposes predictions through a web service for practical use.

### Project Vision
HealthTrack AI is built to be a prototype for intelligent patient management systems that can support healthcare operations with low-latency, explainable, and scalable ML-driven decision support.

## 2. Business and Technical Impact

### Business Value
- Helps identify high-risk patients earlier.
- Supports resource planning in hospitals and care centers.
- Improves treatment prioritization.
- Reduces manual decision burden on clinical staff.
- Provides a scalable baseline for future healthcare AI systems.

### Technical Value
- Demonstrates end-to-end ML pipeline development.
- Shows data preprocessing, model training, model persistence, and deployment.
- Includes DVC for version control of ML artifacts.
- Uses modern Python ML tooling and API deployment.

## 3. Solution Architecture

The project follows a typical production ML workflow:

1. Data Collection
   - Raw patient data is stored in CSV-like tabular formats.

2. Data Ingestion
   - Data is read and validated.

3. Feature & Target Processing
   - Numeric columns are scaled.
   - Categorical columns are encoded.
   - Target variable is processed and saved for prediction mapping.

4. Model Training
   - A model is trained using processed data.

5. Model Evaluation
   - Model performance is assessed to determine reliability.

6. Artifact Persistence
   - Trained model and preprocessing pipelines are saved using joblib.

7. API Deployment
   - FastAPI loads the saved artifacts and serves predictions.

## 4. Tech Stack

### Programming Language
- Python

### Data & ML Libraries
- pandas
- scikit-learn
- joblib
- DVC
- dvclive

### Web/API
- FastAPI
- Pydantic
- Uvicorn

### DevOps / Experiment Tracking
- DVC pipeline tracking
- dvclive logs
- model artifact persistence

## 5. Repository Structure

```text
HealthTrack-AI-Predictive-Patient-Management/
├── .dvc/                      # DVC metadata
├── .dvcignore                 # DVC ignore rules
├── .github/                  # GitHub automation/configuration
├── REPORT/                   # Supporting project reports
├── dvclive/                  # Experiment tracking outputs
├── exp/                      # Experiment artifact folder
├── feature_selector/         # Feature selection-related files
├── models/
│   └── model.pkl            # Trained ML model
├── processpipe/
│   ├── feature_pipe.pkl     # Feature transformation pipeline
│   └── target_pipe.pkl      # Target encoding pipeline
├── src/
│   ├── data_ingestion.py     # Data loading/validation
│   ├── data_processing.py    # Feature & target preprocessing
│   ├── feature_selection.py  # Feature selection logic
│   ├── model_training.py     # Model training logic
│   ├── model_evaluation.py   # Model evaluation scripts
│   ├── utils.py             # Logging and helper utilities
│   └── __inint__.py         # Package placeholder
├── dockerfile                # Container build file
├── dvc.yaml                  # DVC pipeline definition
├── dvc.lock                  # Locked DVC pipeline state
├── main.py                   # FastAPI prediction app
├── params.yaml               # Model/train parameters
├── requirements.txt          # Python dependencies
├── stream.py                 # Supporting runtime/streaming script
├── README.md                 # Project documentation (this file)
└── .gitignore
```

## 6. Key Files in Detail

### main.py
This is the heart of the project. It:
- initializes a FastAPI app,
- loads the trained model,
- loads preprocessing artifacts,
- defines request/response validation,
- exposes `/health` and `/predict` endpoints,
- transforms patient input before model inference,
- converts prediction back to readable labels.

#### API Endpoints
- `GET /` - basic app info
- `GET /health` - checks model and pipeline status
- `POST /predict` - makes predictions for patient data

#### Example Input Schema
```json
{
  "Age": 45,
  "Gender": "Male",
  "Blood": "B-",
  "condition": "Asthma",
  "Amount": 17695.91,
  "Room": 345,
  "Medication": "Penicillin"
}
```

#### Example Output
```json
{
  "HEALTH: ": "Healthy"
}
```

> Note: The exact target labels depend on the trained target encoder and the original dataset used in training.

### src/data_processing.py
This file creates the preprocessing logic for the ML pipeline.

#### Included functionality
- loading raw CSV data,
- identifying numeric and categorical columns,
- applying StandardScaler to numerical features,
- applying OrdinalEncoder to categorical features,
- fitting the target encoder,
- saving feature and target pipeline objects using joblib.

#### Key Infrastructure
```python
numpipe = Pipeline(steps=[("SCALING", StandardScaler())])
catpipe = Pipeline(steps=[("LABELENCODING", OrdinalEncoder())])
```

This is a standard ML preprocessing pattern used to keep data transformation reproducible at inference time.

### src/model_training.py
This file is responsible for training the predictive model. The repository uses a trained artifact stored in:
- `models/model.pkl`

The model is intended to be trained on a patient health dataset and saved for later use by the API.

### src/model_evaluation.py
This file contains evaluation logic for assessing model quality, including measures such as:
- accuracy,
- precision,
- recall,
- F1-score,
- confusion matrix or classification metrics depending on implementation.

### src/feature_selection.py
This file focuses on selecting the most relevant input features, improving interpretability and reducing noise in the model.

### src/data_ingestion.py
This file loads raw patient data for further processing. It plays a critical role in establishing a consistent data pipeline from source to model input.

## 7. Machine Learning Pipeline Explanation

### Data Flow
```text
Raw patient dataset
        ↓
Data ingestion
        ↓
Feature processing
        ↓
Target encoding
        ↓
Model training
        ↓
Model evaluation
        ↓
Model artifact saving
        ↓
FastAPI inference
```

### Preprocessing Strategy
The preprocessing flow uses:
- `StandardScaler` for numerical data
- `OrdinalEncoder` for categorical data
- `ColumnTransformer` to combine multiple feature pipelines

This ensures that the same transformations are applied to new input data as during training, preserving consistency and correctness.

### Training and Prediction Pattern
At inference time:
1. the incoming patient data is converted to a DataFrame,
2. the feature pipeline transforms the input,
3. the model predicts the target class,
4. the target pipeline inverse-transforms the prediction.

This allows the final output to be human-readable rather than encoded labels.

## 8. Data Description

The project is built around patient records with fields such as:
- `Age`
- `Gender`
- `Blood Type`
- `Medical Condition`
- `Billing Amount`
- `Room Number`
- `Medication`
- `Test Results` (target variable)

These features make the system useful for healthcare analytics, especially when combined with classification objectives such as health-risk prediction or outcome forecasting.

## 9. How the Project Works End-to-End

### Training Stage
- Raw patient data is loaded.
- Missing or inconsistent values are handled during preprocessing.
- Features are transformed.
- Target labels are encoded.
- Model is trained and saved.

### Deployment Stage
- Model and preprocessing pipelines are loaded into `main.py`.
- A request is received through the API.
- The model predicts health status.
- The output is returned in JSON format.

### Why this is useful
This is a strong demonstration of end-to-end ML engineering and is suitable for interview discussions because it shows:
- problem framing,
- workflow design,
- model preparation,
- reproducibility,
- deployment pipeline.

## 10. DVC and Reproducibility

The repository uses DVC artifacts to support reproducible machine learning workflows.

### Why DVC matters
DVC helps manage:
- dataset versions,
- model files,
- training pipeline outputs,
- experiment reproducibility.

### DVC-related files in the project
- `dvc.yaml`
- `dvc.lock`
- `.dvc/`
- `dvclive/`

This is valuable for MLOps and demonstrates experience with version-controlled model lifecycle management.

## 11. Setup Instructions

### Prerequisites
- Python 3.9+
- pip
- virtual environment recommended

### Installation
```bash
git clone https://github.com/CodewithAvijit/HealthTrack-AI-Predictive-Patient-Management.git
cd HealthTrack-AI-Predictive-Patient-Management
python -m venv venv
source venv/bin/activate   # Linux/macOS
# or venv\Scripts\activate  # Windows
pip install -r requirements.txt
```

### Run the API
```bash
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

### Check the app
Open:
- `http://localhost:8000/`
- `http://localhost:8000/health`
- `http://localhost:8000/docs`

## 12. Example API Usage

### Option 1: Browser Swagger UI
Go to:
```text
http://localhost:8000/docs
```
Use the Swagger interface to test the `/predict` endpoint.

### Option 2: curl
```bash
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "Age": 45,
    "Gender": "Male",
    "Blood": "B-",
    "condition": "Asthma",
    "Amount": 17695.91,
    "Room": 345,
    "Medication": "Penicillin"
  }'
```

## 13. Project Strengths

- End-to-end ML pipeline structure
- API deployment with FastAPI
- Reproducibility using DVC
- Model artifact persistence using joblib
- Real-world healthcare application concept
- Clear separation between data processing, training, and serving

## 14. Potential Improvements

The project can be further strengthened with:

### 1. Data Quality Improvements
- handle missing values robustly,
- validate schema at ingestion,
- include outlier detection,
- standardize naming conventions.

### 2. Model Enhancements
- test multiple models (RandomForest, XGBoost, Logistic Regression, SVM),
- implement cross-validation,
- use hyperparameter tuning,
- track metrics over time.

### 3. MLOps Improvements
- add CI/CD pipeline,
- deploy to Docker/Kubernetes,
- integrate a model registry,
- add monitoring for drift and latency.

### 4. Explainability and Trust
- add SHAP or LIME explanations,
- show feature importance for clinical review,
- document which features matter most.

### 5. Security and Compliance
- anonymize patient data,
- implement access restrictions,
- secure model outputs and API endpoints,
- ensure compliance with healthcare privacy standards.

## 15. Interview-Ready Summary

If someone asks, “What is your project about?” you can say:

> HealthTrack AI is a predictive healthcare analytics project that uses machine learning to estimate patient health outcomes based on clinical and administrative patient information. The project includes data preprocessing, feature transformation, model training, evaluation, DVC-based reproducibility, and a FastAPI API that serves predictions in real time. It demonstrates a complete end-to-end ML solution for patient risk prediction and healthcare resource optimization.

If someone asks, “What tech stack did you use?” you can respond:

> I used Python, pandas, scikit-learn, joblib, FastAPI, Pydantic, DVC, and uvicorn. The model is trained on structured patient healthcare data and served through an API so new patient inputs can be transformed and predicted in real time.

If someone asks, “Why is this project impressive?” you can emphasize:
- it is not just a notebook,
- it includes pipeline reproducibility,
- it handles deployment-ready artifacts,
- it demonstrates a real-world ML workflow,
- it connects model training to business-facing inference.

## 16. Final Conclusion

HealthTrack AI is a well-structured healthcare ML project that showcases a practical end-to-end workflow from raw data to a deployable prediction service. It is ideal for demonstration in interviews because it highlights technical depth, domain understanding, and ML engineering maturity.

The repository demonstrates the ability to:
- work with structured healthcare data,
- transform and prepare features,
- train a machine learning model,
- persist artifacts for production use,
- serve predictions with a web API,
- maintain reproducibility with DVC.

This makes the project both technically credible and professionally relevant.

## 17. Project Links

- GitHub Repository: https://github.com/CodewithAvijit/HealthTrack-AI-Predictive-Patient-Management
- Project Type: Healthcare ML / Predictive Patient Management / API-based ML service

## 18. Short “One-Liner” Pitch

HealthTrack AI is an intelligent patient management system that predicts health outcomes from patient records using machine learning, preprocessing pipelines, and a production-style FastAPI deployment.

---

This documentation is designed to be interview-ready, technical, and easy to explain to recruiters, hiring managers, or interviewers.
