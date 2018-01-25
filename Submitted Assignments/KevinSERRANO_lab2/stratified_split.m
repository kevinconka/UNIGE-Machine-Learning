function [train_idx, test_idx] = stratified_split(data, train_size)
% preserve percentage of samples for each class
    
    data = table2array(data);
    [m, ~] = size(data);
    classes = unique(data(:, end));
    
    train_idx = [];
    test_idx = [];
    for k = 1:numel(classes)
        idx = find( data(:, end) == classes(k) );
        n_k = numel(idx);           % number of observations in class k
        idx = idx(randperm(n_k));
        delim = round(n_k * train_size);
        train_idx = [train_idx; idx(1:delim)];
        test_idx = [test_idx; idx(delim+1:end)];
    end

    train_idx = train_idx(:);
    test_idx = test_idx(:);

end