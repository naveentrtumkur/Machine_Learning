function g = sigmoidGradient(z)
%SIGMOIDGRADIENT returns the gradient of the sigmoid function
%evaluated at z
%   g = SIGMOIDGRADIENT(z) computes the gradient of the sigmoid function
%   evaluated at z. This should work regardless if z is a matrix or a
%   vector. In particular, if z is a vector or matrix, you should return
%   the gradient for each element.

g = zeros(size(z));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the gradient of the sigmoid function evaluated at
%               each value of z (z can be a matrix, vector or scalar).

r=rows(z);
c=columns(z);
for i=1:r;
  for j=1:c;
    g(i,j)= 1/(1+(exp(-z(i,j))));
    g(i,j)=g(i,j).*(1-g(i,j));
    
  end;
end;
z=g;













% =============================================================




end
