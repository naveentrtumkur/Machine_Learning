function [J, grad] = costFunction(theta, X, y)
%COSTFUNCTION Compute cost and gradient for logistic regression
%   J = COSTFUNCTION(theta, X, y) computes the cost of using theta as the
%   parameter for logistic regression and the gradient of the cost
%   w.r.t. to the parameters.

% Initialize some useful values
m = length(y); % number of training examples
n= length(theta);
% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));
r=rows(theta);
c=columns(theta);

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta
%
% Note: grad should have the same dimensions as theta
%

sum=0;
hyp = (theta')*(X');
for i=1:m ;
%  sum=sum+((-y(i)*log(sigmoid((theta')*X(i))))-((1-y(i))*log(1-sigmoid((theta')*X(i)))));
sum=sum+((-y(i)*log(sigmoid(hyp(i))))-((1-y(i))*log(1-sigmoid(hyp(i)))));
end;
J=sum/m;

kum=0,kum1=0,kum2=0;
%for j=1:n+1;
for i=1:m;
  kum=kum+((sigmoid(hyp(i))-(y(i)))*X(i,1));
 % disp(kum);
  kum1=kum1+((sigmoid(hyp(i))-(y(i)))*X(i,2));
  kum2=kum2+((sigmoid(hyp(i))-(y(i)))*X(i,3));
end;
%end;
%disp(kum);
%disp(kum1);
%disp(kum2);
%theta(1,1)=theta(1,1)-(kum/m);
%theta(2,1)=theta(2,1)-(kum1/m);
%theta(3,1)=theta(3,1)-(kum2/m);

grad(1,1)=kum/m;
grad(2,1)=kum1/m;
grad(3,1)=kum2/m;














% =============================================================

end
