function [J, grad] = lrCostFunction(theta, X, y, lambda)
%LRCOSTFUNCTION Compute cost and gradient for logistic regression with 
%regularization
%   J = LRCOSTFUNCTION(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta
%
% Hint: The computation of the cost function and gradients can be
%       efficiently vectorized. For example, consider the computation
%
%           sigmoid(X * theta)
%
%       Each row of the resulting matrix will contain the value of the
%       prediction for that example. You can make use of this to vectorize
%       the cost function and gradient computations. 
%
% Hint: When computing the gradient of the regularized cost function, 
%       there're many possible vectorized solutions, but one solution
%       looks like:
%           grad = (unregularized gradient for logistic regression)
%           temp = theta; 
%           temp(1) = 0;   % because we don't add anything for j = 0  
%           grad = grad + YOUR_CODE_HERE (using the temp variable)
%
sum1=0;
temp1=0;
temp2=0;
sum3=0;
n=length(theta);
hyp = sigmoid(X*theta);
%hyp = (X.*(theta'));


%for i=1:m ;
%  sum=sum+((-y(i)*log(sigmoid((theta')*X(i))))-((1-y(i))*log(1-sigmoid((theta')*X(i)))));
%sum=sum+((-y(i)*log(sigmoid(hyp(i))))-((1-y(i))*log(1-sigmoid(hyp(i)))));
%end;
%J=sum/m;


%sum2=0;
%for j=2:n;
% sum2=sum2+(theta(j,1)^2);
%end;
%sum2=((lambda*sum2)/(2*m));
%J=sum/m+sum2;
temp1= (-y.*(log(hyp)));
temp2= (1-y).*(log(1-(hyp)));
%  sum=sum+((-y(i)*log(sigmoid((theta')*X(i))))-((1-y(i))*log(1-sigmoid((theta')*X(i)))));
sum1=sum(sum((temp1)-(temp2)));

sum3=sum1;

sum2=0;
for j=2:n;
sum2=sum2+(theta(j,1)^2);
end;
sum2=((lambda*sum2)/(2*m));
J=sum3/m+sum2;






% =============================================================
hyp=(theta')*(X');
%kum=zeros(size(theta));
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

%for j=1:n;
%for i=1:m;
 % kum(j,1)=kum(j,1)+((sigmoid(hyp(i))-(y(i)))*X(i,j));
 % disp(kum);
  %kum1=kum1+((sigmoid(hyp(i))-(y(i)))*X(i,2));
  %kum2=kum2+((sigmoid(hyp(i))-(y(i)))*X(i,3));
%end;
%end;
%kum= ((1/m)*(X')*(sigmoid(X)-y));

%for j=1:n;
%  if(j==1)
%  grad(j,1)=kum(j,1);
%  else
%  grad(j,1)=(kum(j,1))+((lambda*theta(j))/m);
%  end;
%end;
%          temp = theta; 
%          temp(1) = 0;   % because we don't add anything for j = 0  
          %grad = grad + YOUR_CODE_HERE (using the temp variable)
%grad = kum+((lambda*temp)/m);


end
