function [theta, J_history] = gradientDescentMulti(X, y, theta, alpha, num_iters, lambda)
%GRADIENTDESCENTMULTI Performs gradient descent to learn theta
%   theta = GRADIENTDESCENTMULTI(x, y, theta, alpha, num_iters) updates theta by
%   taking num_iters gradient steps with learning rate alpha

% Initialize some useful values
m = length(y); % number of training examples
thetaLength = length(theta);
J_history = zeros(num_iters, 1);

for iter = 1:num_iters
    [J_history(iter), gradient] = costFunctionReg(theta, X, y, lambda);
    % ====================== YOUR CODE HERE ======================
    % Instructions: Perform a single gradient step on the parameter vector
    %               theta. 
    %
    % Hint: While debugging, it can be useful to print out the values
    %       of the cost function (computeCostMulti) and gradient here.
    %
    theta = theta - alpha*gradient;

    % ============================================================

    % Save the cost J in every iteration    

end

end
