# House Price Prediction Using Boosting Technique

This repository contains a project for predicting house prices using multiple regression techniques and machine learning models, including **boosting** algorithms. The goal is to train several models on historical house price data and evaluate their performance using the R² score.

## Repository

[House-Price-Using-Boosting-Technique](https://github.com/alihassanml/House-Price-Using-Boosting-Technique.git)

## Models Used

The following models were used for training and testing:

- `LinearRegression`: Simple linear regression model.
- `LogisticRegression`: Logistic regression model, used incorrectly for regression tasks (should be used for classification).
- `DecisionTreeRegressor`: Tree-based regression model for making predictions by splitting the data based on features.
- `RandomForestRegressor`: Ensemble of decision trees that averages their predictions to improve accuracy and control overfitting.
- `GradientBoostingRegressor`: Gradient boosting model that sequentially improves the performance by focusing on errors made by previous models.

## Workflow

1. Split the dataset into training and testing sets.
2. Fit each model on the training data.
3. Make predictions on the test data.
4. Calculate the R² score to evaluate model performance.
5. Visualize the actual vs predicted values for each model.

## Code

Here is the code snippet for training and evaluating the models:

```python
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import r2_score

# Define models
models = {
    'LinearRegression': LinearRegression(),
    'LogisticRegression': LogisticRegression(),
    'DecisionTreeRegressor': DecisionTreeRegressor(),
    'RandomForestRegressor': RandomForestRegressor(),
    'GradientBoostingRegressor': GradientBoostingRegressor()
}

# Train and evaluate models
for name, model in models.items():
    model.fit(X_train, y_train)  # Fit the model
    y_pred = model.predict(X_test)  # Make predictions
    acc_score = r2_score(y_test, y_pred)  # Calculate R² score
    
    print(f'-------------Model {name}-------------')
    print('Accuracy --: ', acc_score)
    
    # Plot actual vs predicted values
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x=y_test, y=y_pred)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], color='red', linestyle='--')
    plt.xlabel('Actual Values')
    plt.ylabel('Predicted Values')
    plt.title(f'Actual vs Predicted for {model}')
    plt.show()
```

## Results

- **R² Score**: This metric is used to measure the accuracy of each model. The closer the value is to 1, the better the model fits the data.
- **Visualization**: Each model's performance is visualized using a scatter plot, showing the relationship between actual and predicted values.

## Prerequisites

To run this project, you need to have the following Python libraries installed:

```bash
pip install numpy pandas scikit-learn matplotlib seaborn
```

## How to Run

1. Clone the repository:

   ```bash
   git clone https://github.com/alihassanml/House-Price-Using-Boosting-Technique.git
   ```

2. Navigate to the project directory:

   ```bash
   cd House-Price-Using-Boosting-Technique
   ```

3. Run the Python script to train the models and visualize the results.

## Conclusion

This project implements and compares different machine learning models to predict house prices. You can easily modify the dataset or models to experiment with different techniques and improve performance.

## License

This project is licensed under the MIT License.
