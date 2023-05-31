# Stock Prediction Neural Network
This project provides a neural network model and a data acquisition script for predicting the daily minimum and maximum stock prices of the NIS company based on a set of historical technical indicators. 
The data acquisition script loads the historical stock data, preprocesses it to calculate the technical indicators, and organizes the data into a format suitable for input to the neural network. The script also includes a normalization step to ensure that all input variables are on a comparable scale, which can improve the network's performance.
This project offers a practical application of machine learning for financial time series forecasting, and serves as a starting point for those interested in applying similar techniques to other stocks or financial instruments.

## Up to date stock information can be found on the following link
*Beogradska berza NIIS* [here](https://www.belex.rs/trgovanje/istorijski/NIIS) 

## Network Information

### Input and Output
- The input to the network is a 10-dimensional vector that represents different technical aspects of the stock's historical data including moving averages, momentum, stochastic oscillators, commodity channel index (CCI), Larry William's R%, and moving average convergence divergence (MACD).

- The output of the network is a 2-dimensional vector that represents the predicted minimum and maximum prices for a stock on the following day.

###  Performance Metrics 
- Performance of the network is assessed using three metrics: Mean Squared Error (MSE), Root Mean Squared Error (RMSE), and Mean Absolute Error (MAE). These metrics provide insight into the accuracy of the network's predictions, as well as the magnitude of the prediction errors.

### Network Structure
- The neural network used in this project is a Feedforward Backpropagation Neural Network. It consists of:
  - An input layer
  - Two hidden layers
  - An output layer
- Both hidden layers consist of 12 neurons each and the activation functions used are 'tansig' (Hyperbolic tangent sigmoid transfer function) for the first hidden layer and 'elliotsig' (Elliot sigmoid transfer function) for the second hidden layer.

### Network Training
- The network is trained using the Bayesian Regularization backpropagation function (trainbr), a powerful training function that adjusts the weights and biases of the network according to Levenberg-Marquardt optimization. It combines the speed of Newton's method when the weights are far from their optimal value and the accuracy of the Gradient Descent method when the weights are close to the optimal value.

- The training data is divided as follows:
  - 80% for training
  - 10% for validation
  - 10% for testing
- Regularization is used during the training to avoid overfitting, with a regularization parameter set at 0.5.

## Visualization
- The performance of the network during training can be visualized using various plots (uncomment the corresponding lines in the code to generate these plots). These plots help to understand the performance, training state, error distribution, and fitness of the network.
