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
K = num_labels;        
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
%n=length(theta);
y_matrix= eye(num_labels)(y,:);
  
a1=[ones(m, 1) X];;
z2=a1*Theta1';

a2=sigmoid(z2);
a2=[ones(m,1) a2];
z3=a2*Theta2';
a3=sigmoid(z3);

sum1=0;

temp1= (-y_matrix.*(log(a3)));
temp2= (1-y_matrix).*(log(1-(a3)));
%  sum=sum+((-y(i)*log(sigmoid((theta')*X(i))))-((1-y(i))*log(1-sigmoid((theta')*X(i)))));
sum1=sum(sum((temp1)-(temp2)));
J=sum1/m;

n=rows(Theta1);
c=columns(Theta1);
n_t=rows(Theta2);
c_t=columns(Theta2);

sum2=0;
for j=1:n;
  for k=2:c;
sum2=sum2+(Theta1(j,k)^2);
end;
end;
sum2=((lambda*sum2)/(2*m));
disp(sum2);

sum3=0;
for j=1:n_t;
  for k=2:c_t;
sum3=sum3+(Theta2(j,k)^2);
end;
end;
sum3=((lambda*sum3)/(2*m));
disp(sum3);
J=J+sum2+sum3;

% Backprop Implementation.....
d3 = a3 - y_matrix;
Theta2_ex1 = Theta2(:,2:end);
d2=((d3*Theta2_ex1).*sigmoidGradient(z2));

D1= d2'*a1;
D2 = d3'*a2;
Theta1_grad=D1/m;
Theta2_grad=D2/m;

%Regularization ofthe computed Back propagation values
Theta1(:,1)=0;
Theta2(:,1)=0;
Theta1=(Theta1*lambda)/m;
Theta2=(Theta2*lambda)/m;
Theta1_grad+=Theta1;
Theta2_grad+=Theta2;


% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
