function J = computeCost(X, y, theta)
%COMPUTECOST Compute cost for linear regression
%   J = COMPUTECOST(X, y, theta) computes the cost of using theta as the
%   parameter for linear regression to fit the data points in X and y

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta
%               You should set J to the cost.

cost_sum = 0;

temp_cost= [];

for i= 1:m
  hypothesis = theta(1) + (X(i,2)*theta(2));
  cost_sum +=  ((hypothesis - y(i))^2);
endfor

J = cost_sum/(2*m);

%J = sum((X*transpose(theta) - y).^2) / (2 * m);

% =========================================================================

end
