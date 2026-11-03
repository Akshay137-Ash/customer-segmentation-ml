# Customer Segmentation Using Machine Learning

## Project Overview

This project explores customer segmentation using Python and machine learning. It was developed as part of a customer segmentation competition, with the objective of predicting customer segment labels from the provided dataset.

## Technologies Used

- **Python** — Core programming language
- **Pandas** — Data loading and preprocessing
- **Scikit-learn** — Machine learning and model evaluation
- **NumPy** — Numerical operations
- **Git & GitHub** — Version control

## Machine Learning Approach

The project explores different approaches to customer segmentation, including data preprocessing, feature scaling, model training, and prediction.

The final approach uses:

- StandardScaler for feature scaling
- NearestCentroid for classification
- Cross-validation for model evaluation

## Project Structure

```text
customer-segmentation-ml/
├── customer_segmentation.py
├── improve_model.py
├── model.py
├── simple_model.py
├── .gitignore
└── README.md
```

## Results

The final model achieved a **1.00000 public leaderboard score** in the competition.

This score reflects performance on the public evaluation set and does not necessarily indicate perfect generalization to unseen data.

## How to Run

1. Clone the repository.
2. Install the required Python packages:

   ```bash
   pip install pandas numpy scikit-learn
   ```

3. Obtain the competition dataset through the authorized source and place the required CSV files in the project directory.
4. Run the main script:

   ```bash
   python customer_segmentation.py
   ```

## Notes

Competition datasets and generated submission files are excluded from this repository using `.gitignore`.

## Author

**Akshay K.**

Developed as part of a machine learning competition.
