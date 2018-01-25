function [train_idx, test_idx] = stratified_split(data, train_size)
% preserve percentage of samples for each class
    
    data = table2array(data);
    classes = unique(data(:, end));    
    train_idx = [];
    test_idx = [];
    
    for k = 1:numel(classes)
        % filter observations per class
        idx = find( data(:, end) == classes(k) );
        % call random splitting on filtered data
        [train_k, test_k] = train_test_split(data(idx), train_size);
        % append indexes to the corresponding vector
        train_idx = [train_idx; idx(train_k)];
        test_idx = [test_idx; idx(test_k)];
    end

end