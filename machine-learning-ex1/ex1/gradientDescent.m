function [theta, J_history] = gradientDescent(X, y, theta, alpha, num_iters)
%GRADIENTDESCENT Performs gradient descent to learn theta
%   theta = GRADIENTDESCENT(X, y, theta, alpha, num_iters) updates theta by 
%   taking num_iters gradient steps with learning rate alpha

% Initialize some useful values
m = length(y); % number of training examples
%n=length(theta);
J_history = zeros(num_iters, 1);
%thet=zeros(num_iters,1);
for iter = 1:num_iters
    % ====================== YOUR CODE HERE ======================
    % Instructions: Perform a single gradient step on the parameter vector
    %               theta. 
    %
    % Hint: While debugging, it can be useful to print out the values
    %       of the cost function (computeCost) and gradient here.
    %

sum=0;
sum1=0;
J = zeros(2,1);
hyp = (theta')*(X');
for i=1:m ;
  sum=sum+((hyp(i)-y(i)));
  sum1=sum1+((hyp(i)-y(i))*X(i,2));
end;
J(1,1)=J(1,1)+alpha*(sum/(m));
J(2,1)=J(2,1)+alpha*(sum1/(m));
theta(1,1)=theta(1,1)-J(1,1);
theta(2,1)=theta(2,1)-J(2,1);
%disp(J(n,1));

%thet(iter,1)=J;
%if(iter<=num_iters)
%disp(theta(iter,1));
%thet(iter,1)=(theta(iter,1))-J;
%disp(theta(iter,1));
%end
    % ============================================================

    % Save the cost J in every iteration    
    J_history(iter) = computeCost(X, y, theta);
    %disp('The value of compute cost for theta:'),disp(iter),disp(J_history(iter));
end

end
