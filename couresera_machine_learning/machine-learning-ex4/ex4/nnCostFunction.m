function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices. 
% 
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.
%

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

% Setup some useful variables
m = size(X, 1);
X = [ones(m,1),X];    
% You need to return the following variables correctly 
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the code by working through the
%               following parts.
%
% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by verifying the cost
%         computed in ex4.m
%
% Part 2: Implement the backpropagation algorithm to compute the gradients
%         Theta1_grad and Theta2_grad. You should return the partial derivatives of
%         the cost function with respect to Theta1 and Theta2 in Theta1_grad and
%         Theta2_grad, respectively. After implementing Part 2, you can check
%         that your implementation is correct by running checkNNGradients
%
%         Note: The vector y passed into the function is a vector of labels
%               containing values from 1..K. You need to map this vector into a 
%               binary vector of 1's and 0's to be used with the neural network
%               cost function.
%
%         Hint: We recommend implementing backpropagation using a for-loop
%               over the training examples if you are implementing it for the 
%               first time.
%
% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%

% recode label
    label = zeros(m, num_labels);
    for i = 1 : m;
        label(i,y(i)) = 1;   % return matric of size m*num_labels, each row is a vector row of the recoded labels
    end;
 
%% COST FUNCTION J

%output for all traning example by FORWARD propagation
        z2 = Theta1*X';
        a2 = sigmoid(z2);
        a2 = [ones(1,size(a2,2)) ; a2];
        z3 = Theta2*a2;
        a3 = sigmoid(z3)';  
% return matric of size m*num_labels, each row is a vector row of the 
% activation of K input units of traning example indicated by index

% cost without regularization term: 
for i = 1:m;
        J = J + sum(-label(i,:)'.*log(a3(i,:)')-(1-label(i,:)').*log(1-a3(i,:)'));
end;
    J = J/m;

% add regularization term to cost, not taking into account the terms that 
% correspond to bias unit
    J = J + lambda/(2*m)*sum(sum(Theta1(:,2:end).^2));
    J = J + lambda/(2*m)*sum(sum(Theta2(:,2:end).^2));
    
%% GRADIDENT CALCULATION
Grad1 = zeros(size(Theta1));
Grad2 = zeros(size(Theta2));
for i = 1:m;
    a1 = X(i,:)';       % take one training example at a time
    %perform feedforward pass
        z2 = Theta1*a1;
        a2 = sigmoid(z2);
        a2 = [ones(1,size(a2,2)) ; a2];
        z3 = Theta2*a2;
        a3 = sigmoid(z3); 
    %end of feedfordward pass
    
    %delta value of the last layer
    delta3 = a3 - label(i,:)';
    delta2 = Theta2'*delta3;
    delta2 = delta2(2:end);     %skip or remove delta2 of the bias term
    delta2 = delta2.*sigmoidGradient(z2);
    
    %accumulate gradient
    Grad1 = Grad1 + delta2*a1';
    Grad2 = Grad2 + delta3*a2';
end;
    Theta1_grad = Grad1/m;
    Theta2_grad = Grad2/m;

    % add regularization term to gradient, again not taking into account
    % the bias term
    adderGrad1 = Theta1(:,2:end);
    adderGrad1 = [zeros(size(adderGrad1, 1) ,1), adderGrad1];
    adderGrad1 = adderGrad1*lambda/m;
    
    adderGrad2 = Theta2(:,2:end);
    adderGrad2 = [zeros(size(adderGrad2, 1) ,1), adderGrad2];
    adderGrad2 = adderGrad2*lambda/m;
    
    Theta1_grad = Theta1_grad + adderGrad1;
    Theta2_grad = Theta2_grad + adderGrad2;
%%
% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];

end
