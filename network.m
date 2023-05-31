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

% Regularization %
net.performParam.regularization = 0.5;

% Activation functions %
net.layers{1}.transferFcn = 'tansig';
net.layers{2}.transferFcn = 'elliotsig';

% Plots
%figure, plotperform(tr)
%figure, plottrainstate(tr)
%figure, ploterrhist(e)
%figure, plotregression(t,y)
%figure, plotfit(net,x,t)

