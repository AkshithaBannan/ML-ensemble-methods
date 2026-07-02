# Corporate Sales Classification: Decision Trees & Ensemble Methods Pipeline

This repository features an end-to-end Machine Learning implementation focused on analyzing and predicting corporate sales performance utilizing a `sales.csv` dataset. The codebase transitions from raw data loading to exploratory transformations, baseline tree construction, hyperparameter optimization, and advanced ensemble configurations designed to optimize generalization accuracy.

---

## Pipeline Architecture & Workflow

The notebook processes features through a structured pipeline to classify sales performance targets:

1. **Environment Setup & Data Ingestion**: Seamless ingestion using Google Colab integration methods to parse independent predictors.
2. **Automated Feature Engineering**: Iterative verification loops that detect categorical object structures and apply `LabelEncoder` formatting to create clean numeric representations.
3. **Baseline Modeling Architecture**: Comparative evaluation of standalone `DecisionTreeClassifier` designs utilizing both **Gini Impurity** and **Information Gain Entropy** split criteria.
4. **Hierarchical Visualization**: Direct visual export of tree splits utilizing both the native `matplotlib` visualizers and high-fidelity vector representations via `graphviz`.
5. **Regularization & Pruning**: Constraining growth depth vectors ($max\_depth$) to limit variance errors.
6. **Ensemble Implementation Frameworks**: Scaling predictive capabilities by introducing robust Bagging, Random Forest, and Gradient Boosting workflows.

---

## Performance Comparison & Cross-Validation Matrix

The notebook monitors structural overfitting by tracking validation accuracy differences across single trees, parallel estimators, and sequential boosting classifiers:

| Model Type / Configuration | Training Accuracy | Testing Accuracy | Repeated CV Training Mean | Repeated CV Testing Mean |
| :--- | :---: | :---: | :---: | :---: |
| **Decision Tree (Unconstrained Gini)** | $1.00$ | $0.725$ | — | — |
| **Decision Tree (Unconstrained Entropy)** | $1.00$ | $0.750$ | — | — |
| **Decision Tree (Pruned to $max\_depth=7$)** | $0.925$ | $0.775$ | — | — |
| **Bagging Classifier ($n=100$)** | $0.997$ | $0.800$ | $0.99$ | **`0.79`** |
| **Random Forest Classifier ($n=100$)** | $0.997$ | $0.800$ | $0.94$ | **`0.78`** |
| **Gradient Boosting Classifier (Stump Base)**| $0.875$ | $0.775$ | $0.88$ | **`0.78`** |

### Key Analytical Takeaways
* **Overfitting Identification**: The unconstrained decision trees immediately maximize training performance ($1.00$) while degrading on unseen testing splits, confirming high-variance overfitting.
* **Pruning Benefits**: Restricting the decision structure to 77 internal nodes across a depth of 7 balances variance, lifting standalone testing accuracy to $0.775$.
* **Ensemble Domination**: Incorporating a `BaggingClassifier` over sub-sampled features stabilizes data partitioning issues, achieving the highest robust generalized performance profile (**$0.80$ testing accuracy**).

---

## Hyperparameter Optimization Grid

To unlock maximum prediction capabilities from sequential learners, the codebase designs a complete `GridSearchCV` parameter validation matrix targeting the `GradientBoostingClassifier`:

```python
param_grid = {
    'n_estimators': [10, 50, 100, 150, 200],
    'learning_rate': [0.01, 0.1, 0.3, 0.5, 0.7],
    'max_depth': [1, 3, 5, 7, 9],
    'subsample': [0.4, 0.7, 0.9, 1.0],
    'min_samples_split': [2, 3, 5, 7],
    'min_samples_leaf': [1, 3, 5],
    'max_features': [None, 'sqrt']
}
