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


% Save the first 5 values of the first two rows of the output
output = y(1:2, 1:5);  % Takes the first 5 values of the first two rows

% Generate dates for the next five days
dates = datetime('tomorrow'):datetime('tomorrow')+4; 

% Open file for writing
fileID = fopen('5_day_prediction.txt', 'w');

% Write the dates
fprintf(fileID, '\t\t\t');
for i = 1:length(dates)
    fprintf(fileID, '%s\t', datestr(dates(i),'dd-mm-yyyy'));
end
fprintf(fileID, '\n');

% Write the first row with label
fprintf(fileID, 'Max Price:\t');
fprintf(fileID, '%f\t', output(1,:));
fprintf(fileID, '\n');

% Write the second row with label
fprintf(fileID, 'Min Price:\t');
fprintf(fileID, '%f\t', output(2,:));
fprintf(fileID, '\n');

% Close the file
fclose(fileID);

% View the Network
view(net)

% Plots
%figure, plotperform(tr)
%figure, plottrainstate(tr)
%figure, ploterrhist(e)
%figure, plotregression(t,y)
%figure, plotfit(net,x,t)

