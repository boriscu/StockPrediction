% Solve an Input-Output Fitting problem with a Neural Network

% This script assumes these variables are defined:
%
%   inputs - input data.
%   target - target data.

x = inputs;
t = target;

trainFcn = 'trainbr';  % Bayesian Regularization backpropagation.

% Create a Fitting Network
hiddenLayerSize = [12,12];
net = fitnet(hiddenLayerSize,trainFcn);

% Setup Division of Data for Training, Validation, Testing
net.divideParam.trainRatio = 80/100;
net.divideParam.valRatio = 10/100;
net.divideParam.testRatio = 10/100;

% Activation functions %
net.layers{1}.transferFcn = 'tansig';
net.layers{2}.transferFcn = 'elliotsig';


% Set the maximum number of epochs
net.trainParam.epochs = 1500;

% Set the maximum Mu value
net.trainParam.mu_max = 1e12;

% Train the Network
[net,tr] = train(net,x,t);

% Test the Network
y = net(x);
e = gsubtract(t,y);
performance = perform(net,t,y)
RMSE = sqrt(perform(net,t,y))
MAE = mae(gsubtract(t,y))

% View the Network
view(net)

% Predicted max and min for the next 5 days from the last day in the data
prediction = net(predict_value);
fprintf('Predicted Max and Min for the next 5 days from the last day in the data: Max = %f, Min = %f\n', prediction(1), prediction(2));
