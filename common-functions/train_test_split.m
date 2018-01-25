function [train, test] = train_test_split(X, t, train_size, varargin)

[m, ~] = size(X);
train_size = round(train_size * m);

idx = randperm(m)';
train = idx(1:train_size);
test = idx(train_size + 1:end);

if nargin <= 3
    train = struct('X', X(train, :), 't', t(train, :));
    test  = struct('X', X(test, :), 't', t(test, :));
else
    % Return indexes of subsets
end 

end