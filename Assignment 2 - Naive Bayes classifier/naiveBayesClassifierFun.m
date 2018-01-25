function [y_hat, likelihood, prior] = naiveBayesClassifierFun(train, test)
% train: matrix containing the set of data to be used as the training set, 
% dimension n by d+1
% test: matrix containing the set of data for the testing, dimension m x c

[n, d] = size(train);
[m, c] = size(test);

% -------------------------------------------------------------------------
% Check that training and testing set dimensions match
if ((d - c) ~= 1 && (d - c) ~= 0)
    error('Test and training dimension mismatch.')
end

% -------------------------------------------------------------------------
% Check that no entry in any of the two data sets is < 1
if sum([train(:) < 1; test(:) < 1]); error('Zero-elements in matrix.'); end

% -------------------------------------------------------------------------
% Train a Naive Bayes classifier on the training set (first input
% argument), using its last column as the target
train_X = train(:, end-1);      % training features
train_y = train(:, end);        % target

max_val = max(train_X(:));   % max number of numerical labels   
k_classes = max(train_y);       % number of classes

% allocate space for prior and likelihood probs
prior = zeros(1, k_classes);
likelihood = zeros(k_classes, d - 1, max_val);  

% compute prior and likelihood probabilities matrix
for k = 1:k_classes                 % loop through classes
    temp = train_X( y == k, :);  % filter data according to k-th class
    prior(k) = sum( train_y == k ) ./ m;
    for i = 1:max_val
        likelihood(k, :, i) = sum( temp == i ) ./ m;
    end
end

g = repmat(prior(:), 1, c);
for i = 1:c
    for j = 1:(d - 1)
        g(:, i) = g(:, i) .* likelihood(:, j, trial(j));
    end
end

end