function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
m = length(y); % number of training examples
n=length(theta);
% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta
sum=0;
hyp = (theta')*(X');
for i=1:m ;
%  sum=sum+((-y(i)*log(sigmoid((theta')*X(i))))-((1-y(i))*log(1-sigmoid((theta')*X(i)))));
sum=sum+((-y(i)*log(sigmoid(hyp(i))))-((1-y(i))*log(1-sigmoid(hyp(i)))));
end;

sum2=0;
for j=2:n;
  sum2=sum2+(theta(j,1)^2);
end;
sum2=((lambda*sum2)/(2*m));
J=sum/m+sum2;


%%%% Gradient calculation....

kum=zeros(size(theta));

for j=1:n;
for i=1:m;
  kum(j,1)=kum(j,1)+((sigmoid(hyp(i))-(y(i)))*X(i,j));
 % disp(kum);
  %kum1=kum1+((sigmoid(hyp(i))-(y(i)))*X(i,2));
  %kum2=kum2+((sigmoid(hyp(i))-(y(i)))*X(i,3));
end;
end;
for j=1:n
  if(j==1)
  grad(j,1)=kum(j,1)/m;
  else
  grad(j,1)=(kum(j,1)/m)+(lambda*theta(j)/m);
end;
end;

%grad(2,1)=kum1/m;
%grad(3,1)=kum2/m;


% =============================================================

end
