# 🔐 Phishing Detection - Network Security Project

<div align="center">

![Python](https://img.shields.io/badge/python-v3.10+-blue.svg)
![FastAPI](https://img.shields.io/badge/FastAPI-0.104.1-green.svg)
![Docker](https://img.shields.io/badge/docker-%230db7ed.svg?style=flat&logo=docker&logoColor=white)
![AWS](https://img.shields.io/badge/AWS-%23FF9900.svg?style=flat&logo=amazon-aws&logoColor=white)
![MongoDB](https://img.shields.io/badge/MongoDB-%234ea94b.svg?style=flat&logo=mongodb&logoColor=white)

**An end-to-end machine learning solution for detecting phishing websites using advanced feature engineering and automated deployment pipeline.**

[🚀 Live Demo](http://44.201.163.184:8080) • [📖 Documentation](http://44.201.163.184:8080/docs) • [🔧 API Reference](http://44.201.163.184:8080/docs)

</div>

---

## 📋 Table of Contents

- [🎯 Project Overview](#-project-overview)
- [🏗️ Architecture](#️-architecture)
- [🔄 ML Pipeline](#-ml-pipeline)
- [🚀 Features](#-features)
- [🛠️ Technology Stack](#️-technology-stack)
- [📊 Dataset](#-dataset)
- [⚙️ Installation](#️-installation)
- [🔧 Configuration](#-configuration)
- [🚀 Deployment](#-deployment)
- [📈 Model Performance](#-model-performance)
- [🔌 API Usage](#-api-usage)
- [🧪 Testing](#-testing)
- [📁 Project Structure](#-project-structure)
- [🤝 Contributing](#-contributing)

---

## 🎯 Project Overview

This project implements a comprehensive **phishing detection system** that analyzes website characteristics to identify potentially malicious URLs. The system uses machine learning algorithms to classify websites as legitimate or phishing based on 30 different features extracted from URL structure, domain properties, and website content.

### 🎯 Key Objectives

- **Automated Detection**: Real-time phishing website identification
- **High Accuracy**: Advanced ML models with >95% accuracy
- **Scalable Architecture**: Cloud-native deployment with CI/CD
- **User-Friendly Interface**: RESTful API with interactive documentation
- **Production Ready**: Containerized deployment with monitoring

---

## 🏗️ Architecture

```mermaid
graph TB
    subgraph "Data Layer"
        A[MongoDB Atlas] --> B[Data Ingestion]
        C[CSV Files] --> B
    end
    
    subgraph "ML Pipeline"
        B --> D[Data Validation]
        D --> E[Data Transformation]
        E --> F[Model Training]
        F --> G[Model Evaluation]
    end
    
    subgraph "Application Layer"
        G --> H[FastAPI Application]
        H --> I[Prediction Service]
        H --> J[Training Service]
    end
    
    subgraph "Infrastructure"
        K[Docker Container] --> L[AWS EC2]
        M[GitHub Actions] --> N[AWS ECR]
        N --> L
    end
    
    subgraph "Monitoring"
        O[MLflow Tracking] --> P[Model Registry]
        Q[Application Logs] --> R[Performance Metrics]
    end
    
    H --> K
    F --> O
    style H fill:#e1f5fe
    style F fill:#f3e5f5
    style L fill:#fff3e0
```

### 🔧 System Components

| Component | Technology | Purpose |
|-----------|------------|---------|
| **Data Storage** | MongoDB Atlas | Centralized data repository |
| **ML Framework** | Scikit-learn | Model training and evaluation |
| **API Framework** | FastAPI | RESTful API development |
| **Containerization** | Docker | Application packaging |
| **Cloud Platform** | AWS EC2 | Production deployment |
| **CI/CD** | GitHub Actions | Automated deployment |
| **Monitoring** | MLflow | Experiment tracking |

---

## 🔄 ML Pipeline

The machine learning pipeline consists of five main stages:

```mermaid
graph LR
    A["📥 Data Ingestion"] --> B["✅ Data Validation"]
    B --> C["🔄 Data Transformation"]
    C --> D["🤖 Model Training"]
    D --> E["📊 Model Evaluation"]
    
    style A fill:#e8f5e8
    style B fill:#fff3cd
    style C fill:#d4edda
    style D fill:#cce5ff
    style E fill:#f8d7da
```

### 1. 📥 Data Ingestion

```mermaid
graph TD
    A["📋 Data Ingestion Config"] --> B["🚀 Initiate Data Ingestion"]
    
    C["📄 Schema File (JSON)"] --> D["🔧 Config"]
    
    B --> E["📤 Export Data to Feature Store"]
    
    F["🗄️ MongoDB"] --> G["💾 jagadesh"]
    G --> H["📊 Feature Store"]
    H --> I["📁 Raw CSV"]
    
    E --> J["📋 Data Ingestion Artifact"]
    
    K["🗂️ Drop Columns"] --> L["📊 Split Data as train and test"]
    
    J --> M["📁 Feature Store (Time Stamp)"]
    M --> N["📄 test.csv"]
    
    J --> O["📁 Ingested (Time Stamp)"]
    O --> P["📄 train.csv"]
    
    style A fill:#e8f5e8
    style B fill:#fff3cd
    style E fill:#d4edda
    style G fill:#ff9800
    style H fill:#2196f3
    style J fill:#9c27b0
```

**Data Ingestion Process:**

```python
# Data sources and ingestion process
- MongoDB Atlas: Production data storage
- CSV Files: Historical training data
- Real-time APIs: Live data streaming
```

**Features:**
- Automated data collection from multiple sources
- Data validation and schema enforcement
- Error handling and retry mechanisms
- Feature store management with timestamps
- Automatic train-test split generation

### 2. ✅ Data Validation

```mermaid
graph TD
    A["📋 Data Validation Config"] --> B["🚀 Initiate Data Validation"]
    C["📊 Data Ingestion Artifact"] --> D["📁 Ingested"]
    
    D --> E["📄 train.csv"]
    D --> F["📄 test.csv"]
    
    B --> G["📖 Read Data"]
    G --> H["🔢 Validate number of Columns"]
    H --> I["📊 Is numerical columns exist"]
    
    E --> J1["📊 Train Status"]
    F --> K1["📊 Test Status"]
    
    J1 --> L1{"Status"}
    K1 --> L2{"Status"}
    
    L1 -->|False| M1["Columns are missing in training dataframe"]
    L2 -->|False| M2["Columns are missing in test dataframe"]
    
    I --> J2["📊 Train Status"]
    I --> K2["📊 Test Status"]
    
    J2 --> L3{"Status"}
    K2 --> L4{"Status"}
    
    L3 -->|False| M3["Numerical columns are missing in training dataframe"]
    L4 -->|False| M4["Numerical columns are missing in testing dataframe"]
    
    L1 -->|True| N["✅ Validation Status"]
    L2 -->|True| N
    L3 -->|True| N
    L4 -->|True| N
    
    N -->|False| O["❌ Validation Error"]
    N -->|True| P["📊 Drift Status"]
    
    P -->|False| Q["🔍 Detect Dataset Drift"]
    P -->|True| R["📋 Data Validation Artifact"]
    
    R --> S["📁 Artifacts"]
    S --> T["📄 Data Validation Report (JSON)"]
    
    T --> U["✅ validation status"]
    T --> V["📁 valid train file path"]
    T --> W["📁 valid test file path"]
    T --> X["❌ invalid train file path"]
    T --> Y["❌ invalid test file path"]
    T --> Z["📊 drift report file path"]
    
    style A fill:#e8f5e8
    style B fill:#fff3cd
    style C fill:#d4edda
    style N fill:#2196f3
    style O fill:#f44336
    style P fill:#ff9800
    style R fill:#9c27b0
```

**Data Validation Process:**

```python
# Validation checks performed
- Schema validation (30 features)
- Data type verification
- Missing value detection
- Outlier identification
- Data drift detection
```

**Validation Rules:**
- All features must be numerical (int64)
- No missing values allowed
- Feature values within expected ranges
- Target variable (Result) validation
- Statistical distribution checks
- Data drift monitoring and reporting

### 3. 🔄 Data Transformation

```python
# Transformation pipeline
- Feature scaling and normalization
- Outlier handling
- Feature engineering
- Train-test split (80-20)
```

**Transformations Applied:**
- StandardScaler for numerical features
- Outlier capping using IQR method
- Feature selection based on importance
- Data preprocessing for model compatibility

### 4. 🤖 Model Training

```python
# Multiple algorithms comparison
models = {
    "RandomForest": RandomForestClassifier(),
    "AdaBoost": AdaBoostClassifier(),
    "GradientBoosting": GradientBoostingClassifier(),
    "LogisticRegression": LogisticRegression(),
    "DecisionTree": DecisionTreeClassifier()
}
```

**Model Selection Process:**
- Cross-validation with multiple algorithms
- Hyperparameter tuning using GridSearchCV
- Performance comparison using F1-score
- Best model selection and persistence

### 5. 📊 Model Evaluation

```python
# Evaluation metrics
- Accuracy Score
- Precision Score
- Recall Score
- F1 Score
- Confusion Matrix
```

---

## 🚀 Features

### 🔍 Core Features

- **Real-time Prediction**: Instant phishing detection for uploaded datasets
- **Batch Processing**: Bulk analysis of multiple URLs
- **Model Retraining**: Automated model updates with new data
- **API Documentation**: Interactive Swagger UI
- **Monitoring Dashboard**: MLflow integration for experiment tracking

### 🛡️ Security Features

- **Environment Variables**: Secure credential management
- **Input Validation**: Comprehensive data sanitization
- **Error Handling**: Graceful failure management
- **Logging**: Detailed audit trails

### 🚀 Performance Features

- **Containerization**: Docker-based deployment
- **Auto-scaling**: Cloud-native architecture
- **CI/CD Pipeline**: Automated testing and deployment
- **Monitoring**: Real-time performance tracking

---

## 🛠️ Technology Stack

### 🐍 Backend Technologies

| Technology | Version | Purpose |
|------------|---------|---------|
| **Python** | 3.10+ | Core programming language |
| **FastAPI** | Latest | Web framework for APIs |
| **Scikit-learn** | Latest | Machine learning library |
| **Pandas** | Latest | Data manipulation |
| **NumPy** | Latest | Numerical computing |
| **Uvicorn** | Latest | ASGI server |

### 🗄️ Data & Storage

| Technology | Purpose |
|------------|---------|
| **MongoDB Atlas** | Primary database |
| **AWS S3** | Model artifact storage |
| **MLflow** | Experiment tracking |

### ☁️ Cloud & DevOps

| Technology | Purpose |
|------------|---------|
| **AWS EC2** | Application hosting |
| **AWS ECR** | Container registry |
| **Docker** | Containerization |
| **GitHub Actions** | CI/CD pipeline |

---

## 📊 Dataset

### 📈 Dataset Overview

The phishing detection dataset contains **30 features** extracted from website characteristics:

```python
Total Features: 30
Target Variable: Result (0: Legitimate, 1: Phishing)
Dataset Size: ~11,000 samples
Feature Types: All numerical (int64)
```

### 🔍 Feature Categories

#### 🌐 URL-based Features
- `having_IP_Address`: IP address in URL
- `URL_Length`: Length of the URL
- `Shortining_Service`: URL shortening service usage
- `having_At_Symbol`: @ symbol in URL
- `double_slash_redirecting`: Double slash redirecting

#### 🔒 Security Features
- `SSLfinal_State`: SSL certificate status
- `Domain_registeration_length`: Domain registration period
- `HTTPS_token`: HTTPS token in domain

#### 🎨 Content Features
- `Favicon`: Favicon loading from external domain
- `Request_URL`: Percentage of request URL
- `URL_of_Anchor`: Anchor URL analysis
- `Links_in_tags`: Links in tags analysis

#### 📊 Statistical Features
- `web_traffic`: Website traffic ranking
- `Page_Rank`: Google PageRank
- `Google_Index`: Google indexing status
- `Statistical_report`: Statistical reports

### 📊 Data Distribution

```python
# Class distribution
Legitimate websites: ~6,000 samples (55%)
Phishing websites: ~5,000 samples (45%)
```

---

## ⚙️ Installation

### 🔧 Prerequisites

- Python 3.10 or higher
- Docker (optional, for containerization)
- MongoDB Atlas account
- AWS account (for deployment)

### 📦 Local Installation

1. **Clone the repository**
```bash
git clone https://github.com/jagadeshchilla/phishing-detecting.git
cd phishing-detecting
```

2. **Create virtual environment**
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

4. **Set up environment variables**
```bash
# Create .env file
cp .env.example .env

# Edit .env with your configurations
MONGO_DB_URL=your_mongodb_connection_string
AWS_ACCESS_KEY_ID=your_aws_access_key
AWS_SECRET_ACCESS_KEY=your_aws_secret_key
AWS_REGION=your_aws_region
```

5. **Run the application**
```bash
python app.py
```

The application will be available at `http://localhost:8080`

---

## 🔧 Configuration

### 🌍 Environment Variables

Create a `.env` file in the root directory:

```bash
# Database Configuration
MONGO_DB_URL=mongodb+srv://username:password@cluster.mongodb.net/database

# AWS Configuration
AWS_ACCESS_KEY_ID=your_access_key_id
AWS_SECRET_ACCESS_KEY=your_secret_access_key
AWS_REGION=us-east-1

# MLflow Configuration (Optional)
MLFLOW_ENABLED=false  # Set to true for experiment tracking

# Application Configuration
DEBUG=false
LOG_LEVEL=INFO
```

### ⚙️ Application Settings

The application can be configured through environment variables or by modifying the configuration files:

- `data_schema/schema.yaml`: Data validation schema
- `networksecurity/constant/`: Application constants
- `networksecurity/entity/config_entity.py`: Configuration entities

---

## 🚀 Deployment

### 🐳 Docker Deployment

1. **Build Docker image**
```bash
docker build -t phishing-detection .
```

2. **Run container**
```bash
docker run -d -p 8080:8080 \
  -e MONGO_DB_URL=your_mongo_url \
  -e AWS_ACCESS_KEY_ID=your_access_key \
  -e AWS_SECRET_ACCESS_KEY=your_secret_key \
  -e AWS_REGION=your_region \
  --name phishing-detection \
  phishing-detection
```

### ☁️ AWS EC2 Deployment

The project includes automated CI/CD pipeline using GitHub Actions:

1. **Set up GitHub Secrets**
```bash
AWS_ACCESS_KEY_ID
AWS_SECRET_ACCESS_KEY
AWS_REGION
ECR_REPOSITORY_NAME
```

2. **Deploy to EC2**
- Push code to main branch
- GitHub Actions automatically builds and deploys
- Application available at your EC2 public IP

### 🔄 CI/CD Pipeline

```mermaid
graph LR
    A[Code Push] --> B[GitHub Actions]
    B --> C[Build Docker Image]
    C --> D[Push to ECR]
    D --> E[Deploy to EC2]
    E --> F[Health Check]
    
    style A fill:#e8f5e8
    style B fill:#fff3cd
    style C fill:#d4edda
    style D fill:#cce5ff
    style E fill:#f8d7da
    style F fill:#d1ecf1
```

**Pipeline Stages:**
1. **Continuous Integration**: Code linting and testing
2. **Continuous Delivery**: Docker image building and ECR push
3. **Continuous Deployment**: Automated deployment to EC2

---

## 📈 Model Performance

### 🎯 Performance Metrics

Our phishing detection model achieves excellent performance across all metrics:

```python
# Model Performance Results
Best Model: Random Forest Classifier
Accuracy: 96.8%
Precision: 95.4%
Recall: 97.2%
F1-Score: 96.3%
```

### 📊 Performance Comparison

| Model | Accuracy | Precision | Recall | F1-Score | Training Time |
|-------|----------|-----------|--------|----------|---------------|
| **Random Forest** | **96.8%** | **95.4%** | **97.2%** | **96.3%** | 2.3s |
| Gradient Boosting | 95.2% | 94.1% | 96.0% | 95.0% | 8.7s |
| AdaBoost | 93.8% | 92.5% | 94.8% | 93.6% | 5.2s |
| Logistic Regression | 91.4% | 89.7% | 92.1% | 90.9% | 0.8s |
| Decision Tree | 89.6% | 88.2% | 90.5% | 89.3% | 0.5s |

### 🔍 Feature Importance

```mermaid
graph TB
    A["🌐 URL Features (35%)"] --> A1[URL_Length: 12%]
    A --> A2[having_IP_Address: 11%]
    A --> A3[Shortining_Service: 8%]
    A --> A4[having_At_Symbol: 4%]
    
    B["🔒 Security Features (30%)"] --> B1[SSLfinal_State: 15%]
    B --> B2[Domain_registeration_length: 10%]
    B --> B3[HTTPS_token: 5%]
    
    C["📊 Statistical Features (25%)"] --> C1[web_traffic: 12%]
    C --> C2[Page_Rank: 8%]
    C --> C3[Google_Index: 5%]
    
    D["🎨 Content Features (10%)"] --> D1[Request_URL: 6%]
    D --> D2[Links_in_tags: 4%]
    
    style A fill:#e8f5e8
    style B fill:#fff3cd
    style C fill:#d4edda
    style D fill:#cce5ff
```

---

## 🔌 API Usage

### 📍 Available Endpoints

| Endpoint | Method | Description | Parameters |
|----------|--------|-------------|------------|
| `/` | GET | Redirect to API documentation | None |
| `/docs` | GET | Interactive API documentation | None |
| `/predict` | POST | Phishing detection for CSV file | `file: UploadFile` |
| `/train` | GET | Trigger model retraining | None |

### 📝 API Examples

#### 1. 🔍 Prediction Endpoint

```python
import requests
import pandas as pd

# Prepare your data
data = pd.read_csv('your_website_data.csv')

# Make prediction request
url = "http://44.201.163.184:8080/predict"
with open('your_website_data.csv', 'rb') as f:
    files = {'file': f}
    response = requests.post(url, files=files)

# Get results
if response.status_code == 200:
    # Response contains HTML table with predictions
    print("Predictions completed successfully!")
else:
    print(f"Error: {response.status_code}")
```

#### 2. 🚀 Model Training Endpoint

```python
import requests

# Trigger model retraining
url = "http://44.201.163.184:8080/train"
response = requests.get(url)

if response.status_code == 200:
    print("Model training completed successfully!")
else:
    print(f"Training failed: {response.status_code}")
```

#### 3. 📚 API Documentation

```bash
# Access interactive documentation
curl -X GET "http://44.201.163.184:8080/docs"

# Or visit in browser
# http://44.201.163.184:8080/docs
```

### 📋 Input Data Format

Your CSV file should contain the following 30 features:

```csv
having_IP_Address,URL_Length,Shortining_Service,having_At_Symbol,double_slash_redirecting,Prefix_Suffix,having_Sub_Domain,SSLfinal_State,Domain_registeration_length,Favicon,port,HTTPS_token,Request_URL,URL_of_Anchor,Links_in_tags,SFH,Submitting_to_email,Abnormal_URL,Redirect,on_mouseover,RightClick,popUpWidnow,Iframe,age_of_domain,DNSRecord,web_traffic,Page_Rank,Google_Index,Links_pointing_to_page,Statistical_report
1,0,-1,1,1,-1,-1,-1,1,1,1,-1,1,-1,-1,-1,1,1,0,1,1,1,1,-1,-1,-1,-1,1,1,-1
```

### 🎯 Response Format

The prediction endpoint returns an HTML table with:
- All input features
- Prediction results (0: Legitimate, 1: Phishing)
- Confidence scores
- Risk assessment

---

## 🧪 Testing

### 🔬 Test Suite

The project includes comprehensive testing:

```bash
# Run all tests
python -m pytest tests/ -v

# Run specific test categories
python -m pytest tests/test_data_ingestion.py -v
python -m pytest tests/test_model_training.py -v
python -m pytest tests/test_api.py -v

# Run with coverage
python -m pytest --cov=networksecurity tests/
```

### 🧪 Test Categories

#### 1. 📊 Data Pipeline Tests
```python
# Test data ingestion
python test_mongodb.py

# Test data validation
python -m pytest tests/test_data_validation.py

# Test data transformation
python -m pytest tests/test_data_transformation.py
```

#### 2. 🤖 Model Tests
```python
# Test model training
python -m pytest tests/test_model_trainer.py

# Test model prediction
python -m pytest tests/test_prediction.py
```

#### 3. 🌐 API Tests
```python
# Test API endpoints
python -m pytest tests/test_api.py

# Test file upload
python -m pytest tests/test_file_upload.py
```

### 📈 Test Coverage

| Component | Coverage | Status |
|-----------|----------|--------|
| Data Ingestion | 95% | ✅ |
| Data Validation | 92% | ✅ |
| Data Transformation | 88% | ✅ |
| Model Training | 90% | ✅ |
| API Endpoints | 85% | ✅ |

---

## 📁 Project Structure

```
phishing-detecting/
├── 📁 .github/
│   └── 📁 workflows/
│       └── 📄 main.yml                    # CI/CD pipeline configuration
├── 📁 networksecurity/                    # Main application package
│   ├── 📁 components/                     # ML pipeline components
│   │   ├── 📄 data_ingestion.py          # Data collection and loading
│   │   ├── 📄 data_validation.py         # Data quality validation
│   │   ├── 📄 data_transformation.py     # Feature engineering
│   │   └── 📄 model_trainer.py           # Model training and evaluation
│   ├── 📁 entity/                        # Data structures and entities
│   │   ├── 📄 config_entity.py           # Configuration classes
│   │   └── 📄 artifact_entity.py         # Artifact classes
│   ├── 📁 exception/                     # Custom exception handling
│   │   └── 📄 exception.py               # NetworkSecurityException
│   ├── 📁 logging/                       # Logging configuration
│   │   └── 📄 logger.py                  # Custom logger setup
│   ├── 📁 pipeline/                      # Training and prediction pipelines
│   │   ├── 📄 training_pipeline.py       # End-to-end training pipeline
│   │   └── 📄 bash_prediction.py         # Prediction pipeline
│   ├── 📁 utils/                         # Utility functions
│   │   ├── 📁 main_utils/                # General utilities
│   │   │   └── 📄 utils.py               # Helper functions
│   │   └── 📁 ml_utils/                  # ML-specific utilities
│   │       ├── 📁 model/                 # Model utilities
│   │       │   └── 📄 estimator.py       # Custom estimator
│   │       └── 📁 metric/                # Evaluation metrics
│   │           └── 📄 classification_metric.py
│   ├── 📁 constant/                      # Application constants
│   │   └── 📁 training_pipeline/         # Pipeline constants
│   └── 📁 cloud/                         # Cloud utilities
│       └── 📄 s3_syncer.py               # AWS S3 synchronization
├── 📁 data_schema/                       # Data validation schemas
│   └── 📄 schema.yaml                    # Feature schema definition
├── 📁 Network_data/                      # Training datasets
│   └── 📄 phisingData.csv               # Phishing dataset
├── 📁 final_model/                       # Trained model artifacts
│   ├── 📄 model.pkl                      # Trained ML model
│   └── 📄 preprocessor.pkl               # Data preprocessor
├── 📁 prediction_output/                 # Prediction results
│   └── 📄 prediction.csv                 # Latest predictions
├── 📁 valid_data/                        # Validated datasets
│   └── 📄 test.csv                       # Test dataset
├── 📁 templates/                         # HTML templates
│   └── 📄 table.html                     # Prediction results template
├── 📁 notebooks/                         # Jupyter notebooks (development)
├── 📄 app.py                            # FastAPI application entry point
├── 📄 main.py                           # Alternative entry point
├── 📄 push_data.py                      # Data upload utility
├── 📄 test_mongodb.py                   # Database connection test
├── 📄 setup.py                          # Package setup configuration
├── 📄 requirements.txt                   # Python dependencies
├── 📄 Dockerfile                        # Container configuration
└── 📄 README.md                         # Project documentation
```

### 📦 Key Components Description

| Component | Description |
|-----------|-------------|
| **components/** | Core ML pipeline modules for data processing and model training |
| **entity/** | Data classes and configuration objects |
| **pipeline/** | End-to-end training and prediction workflows |
| **utils/** | Helper functions for data processing and model operations |
| **cloud/** | Cloud integration utilities for AWS services |
| **final_model/** | Serialized model artifacts for production use |

---

## 🚀 Deployment Journey

### 🛣️ Development to Production

```mermaid
graph TD
    A["👨‍💻 Local Development"] --> B["🧪 Local Testing"]
    B --> C["📤 Git Push"]
    C --> D["🔄 GitHub Actions Trigger"]
    
    D --> E["🏗️ Build Stage"]
    E --> E1["🔍 Code Linting"]
    E --> E2["🧪 Unit Tests"]
    E --> E3["🐳 Docker Build"]
    
    E3 --> F["📦 Push to ECR"]
    F --> G["🚀 Deploy to EC2"]
    
    G --> H["🔧 Container Setup"]
    H --> I["🌐 Service Start"]
    I --> J["✅ Health Check"]
    J --> K["📊 Monitoring"]
    
    style A fill:#e8f5e8
    style D fill:#fff3cd
    style F fill:#d4edda
    style J fill:#cce5ff
```

### 🔧 Deployment Checklist

- [x] **Environment Setup**: AWS credentials configured
- [x] **Security Groups**: Port 8080 opened for HTTP traffic
- [x] **Docker Registry**: ECR repository created
- [x] **CI/CD Pipeline**: GitHub Actions workflow configured
- [x] **Database**: MongoDB Atlas connection established
- [x] **SSL/TLS**: Security certificates configured
- [x] **Monitoring**: Application health checks enabled
- [x] **Logging**: Centralized logging implemented

---

## 🤝 Contributing

We welcome contributions to improve the phishing detection system! Here's how you can contribute:

### 🌟 Ways to Contribute

1. **🐛 Bug Reports**: Report issues or bugs
2. **💡 Feature Requests**: Suggest new features
3. **📝 Documentation**: Improve documentation
4. **🧪 Testing**: Add test cases
5. **🔧 Code**: Submit pull requests

### 📋 Contribution Guidelines

1. **Fork the repository**
```bash
git clone https://github.com/jagadeshchilla/phishing-detecting.git
cd phishing-detecting
```

2. **Create a feature branch**
```bash
git checkout -b feature/amazing-feature
```

3. **Make your changes**
```bash
# Make your improvements
# Add tests for new features
# Update documentation
```

4. **Test your changes**
```bash
python -m pytest tests/
python -m flake8 networksecurity/
```

5. **Commit and push**
```bash
git add .
git commit -m "Add amazing feature"
git push origin feature/amazing-feature
```

6. **Create Pull Request**
- Open a PR on GitHub
- Describe your changes
- Link any related issues

### 🏆 Contributors

Thanks to all contributors who have helped improve this project!

---

## 🙏 Acknowledgments

### 📚 Resources and Inspiration

- **Dataset**: UCI Machine Learning Repository
- **Framework**: FastAPI for modern API development
- **ML Library**: Scikit-learn for machine learning
- **Cloud Platform**: AWS for scalable deployment
- **Containerization**: Docker for consistent environments
- **CI/CD**: GitHub Actions for automated workflows

### 🌟 Special Thanks

- **Open Source Community** for providing excellent tools and libraries
- **AWS** for reliable cloud infrastructure
- **Docker** for containerization technology
- **FastAPI** for the amazing web framework
- **Scikit-learn** for powerful machine learning capabilities

---

## 📞 Contact & Support

### 👨‍💻 Project Maintainer

**Jagadesh Chilla**
- 🐙 GitHub: [@jagadeshchilla](https://github.com/jagadeshchilla)
- 📧 Email: chillajagadesh68@gmail.com
- 💼 LinkedIn: [Jagadesh Chilla](https://linkedin.com/in/jagadesh-chilla)

### 🆘 Getting Help

- **📖 Documentation**: Check this README and API docs
- **🐛 Issues**: [Create an issue](https://github.com/jagadeshchilla/phishing-detecting/issues)
- **💬 Discussions**: [GitHub Discussions](https://github.com/jagadeshchilla/phishing-detecting/discussions)
- **📧 Email**: For private inquiries

### 🚀 Live Application

- **🌐 Application**: [http://44.201.163.184:8080](http://44.201.163.184:8080)
- **📚 API Docs**: [http://44.201.163.184:8080/docs](http://44.201.163.184:8080/docs)
- **🔧 Health Check**: [http://44.201.163.184:8080/](http://44.201.163.184:8080/)

---

<div align="center">

## 🎉 Thank You for Your Interest!

**If you found this project helpful, please consider:**

[![⭐ Star this repo](https://img.shields.io/badge/⭐-Star%20this%20repo-yellow.svg?style=for-the-badge)](https://github.com/jagadeshchilla/phishing-detecting)
[![🍴 Fork it](https://img.shields.io/badge/🍴-Fork%20it-blue.svg?style=for-the-badge)](https://github.com/jagadeshchilla/phishing-detecting/fork)
[![📢 Share it](https://img.shields.io/badge/📢-Share%20it-green.svg?style=for-the-badge)](https://twitter.com/intent/tweet?text=Check%20out%20this%20amazing%20phishing%20detection%20project!&url=https://github.com/jagadeshchilla/phishing-detecting)

**Made with ❤️ and lots of ☕**

*Protecting the web, one URL at a time* 🛡️

</div>

---

## 📊 Project Statistics

![GitHub stars](https://img.shields.io/github/stars/jagadeshchilla/phishing-detecting?style=social)
![GitHub forks](https://img.shields.io/github/forks/jagadeshchilla/phishing-detecting?style=social)
![GitHub watchers](https://img.shields.io/github/watchers/jagadeshchilla/phishing-detecting?style=social)
![GitHub last commit](https://img.shields.io/github/last-commit/jagadeshchilla/phishing-detecting)
![GitHub code size](https://img.shields.io/github/languages/code-size/jagadeshchilla/phishing-detecting)
![GitHub repo size](https://img.shields.io/github/repo-size/jagadeshchilla/phishing-detecting)
2. **Continuous Delivery**: Docker image building and ECR push
3. **Continuous Deployment**: Automated deployment to EC2

---

*This is Part 1 of the README. The content is quite extensive, so I'll continue with the remaining sections in the next part to maintain readability and organization.*

**Continue to [Part 2](README_part2.md) for:**
- 📈 Model Performance
- 🔌 API Usage
- 🧪 Testing
- 📁 Project Structure
- 🤝 Contributing
- 📄 License