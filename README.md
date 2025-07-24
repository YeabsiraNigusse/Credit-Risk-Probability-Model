# Credit Risk Probability Model

## Project Overview

This project develops a comprehensive credit scoring model for Bati Bank's buy-now-pay-later service partnership with an eCommerce platform. The model transforms behavioral data into predictive risk signals using customer Recency, Frequency, and Monetary (RFM) patterns to assess credit risk and inform loan decisions.

## Business Context

Bati Bank, with over 10 years of experience in financial services, is partnering with a successful eCommerce company to provide buy-now-pay-later services. This project creates a Credit Scoring Model using eCommerce platform data to evaluate customer creditworthiness and predict default probability.

## Project Objectives

1. **Define Proxy Variable**: Create a proxy variable to categorize users as high risk (bad) or low risk (good)
2. **Feature Selection**: Identify observable features with high correlation to the default proxy variable
3. **Risk Probability Model**: Develop a model that assigns risk probability for new customers
4. **Credit Score Assignment**: Create a model that converts risk probability to credit scores
5. **Loan Optimization**: Predict optimal loan amount and duration for qualified customers

## Credit Scoring Business Understanding

### 1. Basel II Accord's Impact on Model Requirements

The Basel II Accord's emphasis on risk measurement significantly influences our modeling approach in several key ways:

**Risk Quantification Standards**: Basel II requires banks to maintain capital reserves proportional to their risk exposure, making accurate risk measurement critical for regulatory compliance and capital efficiency. This necessitates models that can reliably quantify probability of default (PD), loss given default (LGD), and exposure at default (EAD).

**Model Validation and Documentation**: The accord mandates rigorous model validation, back-testing, and comprehensive documentation. Our model must be interpretable enough for regulators to understand the risk assessment logic, requiring clear feature importance explanations and decision pathways.

**Supervisory Review Process**: Under Pillar 2 of Basel II, supervisors evaluate banks' risk management processes. This requires our model to be transparent, well-documented, and capable of producing audit trails for regulatory review.

**Market Discipline Requirements**: Pillar 3 mandates public disclosure of risk management practices, emphasizing the need for explainable models that can be communicated to stakeholders and the public.

### 2. Proxy Variable Necessity and Business Risks

**Why Proxy Variables Are Necessary**:
Since we lack direct "default" labels in eCommerce transaction data, we must create proxy variables using behavioral patterns (RFM analysis) to identify customers likely to default. This approach leverages the correlation between purchasing behavior and creditworthiness.

**Potential Business Risks**:

- **Misclassification Risk**: Proxy variables may incorrectly classify good customers as high-risk (Type I error) or bad customers as low-risk (Type II error), leading to lost revenue or increased defaults
- **Behavioral Drift**: Customer behavior patterns may change over time, making historical proxies less predictive of future default risk
- **Selection Bias**: eCommerce behavior may not fully represent credit behavior, potentially excluding creditworthy customers with different shopping patterns
- **Regulatory Scrutiny**: Regulators may question the validity of proxy variables, requiring robust validation and justification
- **Model Stability**: Proxy-based models may be less stable than models built on actual default data, requiring more frequent recalibration

### 3. Model Complexity Trade-offs in Regulated Financial Context

**Simple, Interpretable Models (e.g., Logistic Regression with WoE)**:

*Advantages*:
- **Regulatory Compliance**: Easier to explain to regulators and auditors
- **Transparency**: Clear understanding of feature contributions to risk assessment
- **Stability**: More stable predictions across different market conditions
- **Debugging**: Easier to identify and fix model issues
- **Legal Defensibility**: Can provide clear explanations for adverse credit decisions

*Disadvantages*:
- **Lower Predictive Power**: May miss complex patterns in data
- **Feature Engineering Dependency**: Requires extensive manual feature engineering
- **Limited Adaptability**: Less capable of capturing non-linear relationships

**Complex, High-Performance Models (e.g., Gradient Boosting)**:

*Advantages*:
- **Higher Accuracy**: Better predictive performance and pattern recognition
- **Automatic Feature Interactions**: Captures complex relationships without manual engineering
- **Adaptability**: Better handles changing market conditions and customer behavior

*Disadvantages*:
- **Black Box Nature**: Difficult to explain individual predictions to regulators
- **Regulatory Risk**: May face scrutiny under fair lending laws and model risk management guidelines
- **Overfitting Risk**: May not generalize well to new data
- **Maintenance Complexity**: Requires sophisticated monitoring and validation processes

**Recommended Approach**: 
In a regulated financial context, we recommend starting with interpretable models for regulatory approval and baseline performance, then potentially incorporating ensemble methods that combine interpretable and complex models, with robust explainability frameworks (SHAP, LIME) to maintain transparency while improving performance.

## Project Structure

```
credit-risk-model/
├── .github/workflows/ci.yml   # CI/CD pipeline
├── data/                       # Data directory (gitignored)
│   ├── raw/                   # Raw data files
│   └── processed/             # Processed data for training
├── notebooks/
│   └── 1.0-eda.ipynb          # Exploratory data analysis
├── src/
│   ├── __init__.py
│   ├── data_processing.py     # Feature engineering scripts
│   ├── train.py               # Model training scripts
│   ├── predict.py             # Inference scripts
│   └── api/
│       ├── main.py            # FastAPI application
│       └── pydantic_models.py # API data models
├── tests/
│   └── test_data_processing.py # Unit tests
├── Dockerfile                 # Container configuration
├── docker-compose.yml         # Multi-container setup
├── requirements.txt           # Python dependencies
├── .gitignore                # Git ignore rules
└── README.md                 # Project documentation
```

## Key Features

- **RFM Analysis**: Customer segmentation based on Recency, Frequency, and Monetary patterns
- **Proxy Variable Creation**: Behavioral-based risk categorization
- **Multiple Model Approaches**: From interpretable logistic regression to advanced ensemble methods
- **API Integration**: FastAPI-based service for real-time credit scoring
- **Regulatory Compliance**: Model interpretability and documentation for regulatory requirements
- **Comprehensive Testing**: Unit tests and model validation frameworks

## Technology Stack

- **Python 3.8+**: Core programming language
- **Pandas/NumPy**: Data manipulation and analysis
- **Scikit-learn**: Machine learning algorithms
- **FastAPI**: API framework for model serving
- **Docker**: Containerization for deployment
- **Pytest**: Testing framework
- **Jupyter**: Interactive analysis and exploration

## Getting Started

1. **Clone the repository**
2. **Set up virtual environment**
3. **Install dependencies**: `pip install -r requirements.txt`
4. **Run exploratory analysis**: Open `notebooks/1.0-eda.ipynb`
5. **Process data**: Execute `src/data_processing.py`
6. **Train models**: Run `src/train.py`
7. **Start API service**: `uvicorn src.api.main:app --reload`

## Model Development Phases

1. **Data Understanding**: Explore eCommerce transaction data and customer behavior
2. **Proxy Variable Definition**: Create risk categories using RFM analysis
3. **Feature Engineering**: Develop predictive features from transaction data
4. **Model Training**: Build and validate multiple model approaches
5. **Model Evaluation**: Assess performance using appropriate metrics
6. **Model Deployment**: Deploy via API for real-time scoring
7. **Monitoring**: Implement model performance monitoring and drift detection

## Regulatory Considerations

- Model interpretability and explainability
- Comprehensive documentation and audit trails
- Regular model validation and back-testing
- Compliance with fair lending practices
- Risk management and governance frameworks

## Contributing

Please read our contributing guidelines and ensure all code follows our testing and documentation standards.

## License

This project is licensed under the MIT License - see the LICENSE file for details.
