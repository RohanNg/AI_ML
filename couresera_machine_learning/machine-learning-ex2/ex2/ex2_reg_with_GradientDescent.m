%% Machine Learning Online Class - Exercise 2: Logistic Regression

%% Initialization
clear ; close all; clc

%% Load Data
%  The first two columns contains the X values and the third column
%  contains the label (y).

data = load('ex2data1.txt');
X = data(:, [1, 2]); y = data(:, 3);
[X, mu, sigma] = featureNormalize(X);   %for data1 you must use feature
%normalize

plotData(X, y);

% Put some labels 
hold on;

% Labels and Legend
xlabel('Microchip Test 1')
ylabel('Microchip Test 2')

% Specified in plot order
legend('y = 1', 'y = 0')
hold off;

X = mapFeature(X(:,1), X(:,2));

% Initialize fitting parameters
initial_theta = zeros(size(X, 2), 1);

% Set regularization parameter lambda to 1
lambda = 0;

% Compute and display initial cost and gradient for regularized logistic
% regression
[cost, grad] = costFunctionReg(initial_theta, X, y, lambda);

fprintf('Cost at initial theta (zeros): %f\n', cost);

fprintf('\nProgram paused. Press enter to continue.\n');
pause;

fprintf('Running gradient descent ...\n');

% Set up gradient descent for regularized function
alpha = 0.5;
num_iters = 400;
lambda = 1;
initial_theta = zeros(size(X, 2), 1);
% Init Theta and Run Gradient Descent 
[theta, J_history] = gradienDescentMulti(X, y, initial_theta, alpha, num_iters, lambda);

%plot J_history
plot(1:num_iters,J_history);
pause;
plotDecisionBoundary(theta, X, y);

p = predict(theta, X);

fprintf('Train Accuracy: %f\n', mean(double(p == y)) * 100);



