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
theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

% Setup some useful variables
m = size(X, 1);
         
% You need to return the following variables correctly 
J = 0;
theta1_grad = zeros(size(theta1));
theta2_grad = zeros(size(theta2));

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
% -------------------------------------------------------------
% =========================================================================

%Cost Function
X=[ones(m , 1)  X];%adding 1 coloumn to X
weight1=X;
a2=weight1*theta1';
weight2 = sigmoid(a2);
weight2=[ones(m,1) weight2];
weight3=sigmoid(weight2*theta2');

r=zeros(m,max(y));
for i = 1:m
    r(i,y(i,1)) = 1;
end  

cost_fun=r.*log(weight3)+(1-r).*log(1-weight3);

J=(-sum(sum(cost_fun,2))/m)+lambda/(2*m)*(sum(sum(theta1(:,2:end).^2))+sum(sum(theta2(:,2:end).^2)));


delta3=weight3-r;
delta2=(delta3*theta2(:,2:end)).*sigmoidGradient(a2);

Delta1=delta2'*weight1;
Delta2=delta3'*weight2;

theta1_grad=Delta1/m+lambda*[zeros(hidden_layer_size,1) theta1(:,2:end)]/m;
theta2_grad=Delta2/m+lambda*[zeros(num_labels,1) theta2(:,2:end)]/m;

% Unroll gradients
grad = [theta1_grad(:) ; theta2_grad(:)];


end