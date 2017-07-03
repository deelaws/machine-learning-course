function [theta, J_history] = gradientDescent(X, y, theta, alpha, num_iters)
%GRADIENTDESCENT Performs gradient descent to learn theta
%   theta = GRADIENTDESCENT(X, y, theta, alpha, num_iters) updates theta by 
%   taking num_iters gradient steps with learning rate alpha

% Initialize some useful values
m = length(y); % number of training examples
J_history = zeros(num_iters, 1);


% cd 'C:\Users\deela\Documents\Machine Learning Course\week-2\machine-learning-ex1\ex1';
for iter = 1:num_iters

    % ====================== YOUR CODE HERE ======================
    % Instructions: Perform a single gradient step on the parameter vector
    %               theta. 
    %
    % Hint: While debugging, it can be useful to print out the values
    %       of the cost function (computeCost) and gradient here.
    %
    
    
    total_sum = 0;
    total_sum_2 = 0;
    for i = 1:m
      hypothesis = theta(1) + (theta(2) * X(i, 2));
      total_sum += (hypothesis - y(i));
      total_sum_2 += ((hypothesis - y(i)) * X(i, 2));
    endfor
    
    theta_1 =  (theta(1) - ( alpha * (total_sum /  m)));
    
   
    theta_2 =  (theta(2) - ( alpha * (total_sum_2 /m )));

    theta(1) = theta_1;
    theta(2) = theta_2;


    % ============================================================

    % Save the cost J in every iteration    
    J_history(iter) = computeCost(X, y, theta);
    
    

end

end
