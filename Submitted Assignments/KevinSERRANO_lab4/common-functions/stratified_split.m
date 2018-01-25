function [train, test] = stratified_split(X, t, train_size)
% preserve percentage of samples for each class

% X = table2array(X);
[classes, ~, class_vec] = unique(t, 'rows');
[n_classes, d] = size(classes);
train_idx = [];
test_idx = [];

for class = 1:n_classes
    % filter observations per class
    idx = find( class_vec == class );
    
    % call random splitting on filtered X
    [train_k, test_k] = train_test_split(X(idx), t(idx), train_size, 'idx');
    
    % append indexes to the corresponding vector
    train_idx = [train_idx; idx(train_k)];
    test_idx = [test_idx; idx(test_k)];
end

if d > 1
    train = struct('X', X(train_idx, :), 't', t(train_idx, :));
    test  = struct('X', X(test_idx, :), 't', t(test_idx, :));
else
    train = struct('X', X(train_idx, :), 't', t(train_idx));
    test  = struct('X', X(test_idx, :), 't', t(test_idx));
end

end