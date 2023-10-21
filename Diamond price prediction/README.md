
# Diamond Price Prediction

This project uses the Diamond Dataset to examine the factors influencing Diamond Price.

# About Data
These dataset contain 10 features in which 'Price(in US dollars)' is dependent feature.The dataset consists of 53940 diamonds records, with each record containing information about various predictors and a price index.Those different features are:

* Carat(Weight of Daimond)
* Cut(Quality)
* Color
* Clarity
* Depth
* Table
* X(length)	
* Y(width)	
* Z(Depth)

# Models Evaluated
Several regression models were trained and evaluated to predict Performance Index. The models include:
* Linear Regression
* KNeighbors Regressor
* Decision Tree Regressor
* Gradient Boosting Regressor
* Random Forest Regressor

# Performance Metrics
The models were evaluated on the basis of:

* Mean Squared Error (MSE): the average squared difference between the value observed in a statistical study and the values predicted from a model.
* R-squared (R²) Score: Indicates the proportion of the variance in the dependent variable that is predictable from the independent variables.
* Mean Absolute Error (MAE): the average of absolute errors for a group of predictions and observations as a measurement of the magnitude of errors for the entire group

# Conclusions
1. The Linear Regression and Gradient Boosting Regressor performed best based on MSE, MAE and least based on R² Score.
2. KNeighbors Regressor, Decision Tree Regressor and Random Forest Regressor while providing a basic understanding, might not be the most suitable models for this dataset.
3. Decision Tree Regressor performed worst for this dataset