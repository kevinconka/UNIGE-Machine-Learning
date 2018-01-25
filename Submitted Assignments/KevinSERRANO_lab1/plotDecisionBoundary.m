function plotDecisionBoundary(theta, X, y, varargin)
%PLOTDECISIONBOUNDARY Plots the data points X and y into a new figure with
%the decision boundary defined by theta
%   PLOTDECISIONBOUNDARY(theta, X,y) plots the data points with + for the 
%   positive examples and o for the negative examples. X is assumed to be 
%   a either 
%   1) Mx3 matrix, where the first column is an all-ones column for the 
%      intercept.
%   2) MxN, N>3 matrix, where the first column is all-ones

hold on

% Only need 2 points to define a line, so choose two endpoints
plot_x = [min(X(:,2)),  max(X(:,2))];
plot_x = plot_x + [-0.1 0.1] .* abs(plot_x);
% plot_x = [-10,  10];

% Calculate the decision boundary line
plot_y = (-1./theta(3)).*(theta(2).*plot_x + theta(1));

% Plot, and adjust axes for better viewing
plot(plot_x, plot_y, varargin{:})
hold off
    
end