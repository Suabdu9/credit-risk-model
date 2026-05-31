# Project Overview

This project develops an end-to-end credit risk scoring system for Bati Bank's Buy-Now-Pay-Later (BNPL) initiative using transaction data from the Xente eCommerce platform.

Because the dataset does not contain a direct default label, customer behavioral patterns are used to construct a proxy credit risk target based on Recency, Frequency, and Monetary (RFM) analysis. The resulting model predicts customer risk levels and supports future lending decisions.

The project follows an MLOps-oriented workflow including exploratory analysis, feature engineering, model development, experiment tracking, API deployment, containerization, testing, and CI/CD automation.

# Credit Scoring Business Understanding

## 1. How does the Basel II Accord's emphasis on risk measurement influence the need for an interpretable and well-documented model?

The Basel II Accord requires financial institutions to maintain robust risk management frameworks and demonstrate how credit risk is measured, monitored, and controlled. As a result, credit scoring models must be transparent, explainable, and thoroughly documented.

Interpretability is important because regulators, auditors, and business stakeholders need to understand how risk predictions are generated and which factors influence lending decisions. A well-documented model provides evidence of data sources, feature engineering steps, modeling assumptions, validation procedures, and performance metrics. This documentation supports regulatory compliance, facilitates model governance, and enables ongoing monitoring and recalibration when customer behavior or market conditions change.

Therefore, Basel II encourages the use of modeling approaches that balance predictive performance with transparency, reproducibility, and accountability.

## 2. Without a direct "default" label, why is a proxy variable necessary, and what business risks does proxy-based prediction introduce?

The available dataset contains transaction information but does not include a direct indicator of whether a customer defaulted on a loan. Since supervised machine learning models require labeled examples, a proxy variable must be created to represent credit risk.

In this project, customer behavior patterns derived from Recency, Frequency, and Monetary (RFM) metrics can be used to identify groups of customers who exhibit characteristics associated with higher or lower risk. Customers with infrequent activity, long inactivity periods, or low transaction value may be classified as higher-risk customers, while active and consistent customers may be classified as lower-risk customers.

However, proxy variables introduce several business risks:

* The proxy may not accurately represent actual default behavior.
* Incorrect labeling can introduce bias into the model.
* Good customers may be classified as risky (false positives).
* Risky customers may be classified as safe (false negatives).
* Lending decisions based on imperfect proxies can affect profitability and customer trust.

Therefore, the proxy definition must be carefully justified, documented, and validated to ensure it reasonably reflects credit risk.

## 3. What are the key trade-offs between a simple, interpretable model (e.g., Logistic Regression with WoE) and a high-performance model (e.g., Gradient Boosting) in a regulated financial context?

### Logistic Regression with Weight of Evidence (WoE)

Advantages:

* Highly interpretable and easy to explain.
* Coefficients directly indicate the direction and magnitude of risk factors.
* Easier to validate and document for regulatory review.
* Widely accepted in traditional credit scoring environments.
* Supports creation of transparent scorecards.

Disadvantages:

* May struggle to capture complex nonlinear relationships.
* Typically achieves lower predictive performance compared to advanced machine learning models.

### Gradient Boosting Models

Advantages:

* Often achieve higher predictive accuracy.
* Capture nonlinear patterns and feature interactions automatically.
* Better suited for complex behavioral datasets.

Disadvantages:

* Less interpretable and more difficult to explain.
* Harder to justify individual lending decisions.
* More challenging to validate, monitor, and document.
* May require additional explainability techniques such as SHAP values.

### Trade-off Summary

In regulated financial environments, predictive performance alone is not sufficient. Institutions must balance accuracy with interpretability, transparency, and regulatory compliance. Logistic Regression with WoE provides stronger explainability and governance, while Gradient Boosting may deliver superior predictive power. A practical approach is to compare both models and evaluate whether the performance improvement of the complex model justifies the additional complexity and regulatory burden.

## References

- Credit Scoring Statistical Analysis (Sinica)
- Alternative Credit Scoring (HKMA)
- Credit Scoring Approaches Guidelines (World Bank)
- How to Develop a Credit Risk Model and Scorecard
- Credit Risk Fundamentals (Corporate Finance Institute)

## Reproducing the Project

### Clone Repository

```bash
git clone <repository-url>
cd credit-risk-model
```

### Create Environment

```bash
python -m venv venv
source venv/bin/activate
```

Windows:

```bash
venv\Scripts\activate
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

### Run Exploratory Analysis

```bash
jupyter notebook notebooks/eda.ipynb
```

### Run Data Processing Pipeline

```bash
python src/data_processing.py
```

### Run Model Training

```bash
python src/train.py
```

### Run API

```bash
uvicorn src.api.main:app --reload
```
