
# Student Performance 

This project uses the Student Performance Dataset to examine the factors influencing academic student performance.


## About Data
The Student Performance Dataset is a dataset designed to examine the factors influencing academic student performance. The dataset consists of 10,000 student records, with each record containing information about various predictors and a performance index.

Variables:
- Hours Studied
- Previous Scores
- Extracurricular Activities
- Sleep Hours
- Sample Question Papers Practiced
Target Variable:

- Performance Index: A measure of the overall performance of each student.
## Models Evaluated
Several regression models were trained and evaluated to predict Performance Index. The models include:
- Linear Regression
- KNeighbors Regressor
- Decision Tree Regressor
- Gradient Boosting Regressor
- Random Forest Regressor
## Performance Metrics
The models were evaluated on the basis of:
- **Mean Squared Error (MSE):** the average squared difference between the value observed in a statistical study and the values predicted from a model
- **R-squared (R²) Score:** Indicates the proportion of the variance in the dependent variable that is predictable from the independent variables.
- **Mean Absolute Error (MAE):** the average of absolute errors for a group of predictions and observations as a measurement of the magnitude of errors for the entire group
## Conclusions
1. The **Linear Regression** and **Gradient Boosting** Regressor performed best based on MSE, MAE and R² Score
1. **KNeighbors Regressor**, **Decision Tree Regressor** and **Random Forest Regressor** while providing a basic understanding, might not be the most suitable models for this dataset.
1. **Decision Tree Regressor** performed worst for this dataset