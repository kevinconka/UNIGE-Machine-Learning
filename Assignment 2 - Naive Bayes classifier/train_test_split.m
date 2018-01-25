function [train_idx, test_idx] = train_test_split(data, train_size)

    [m, ~] = size(data);
    train_size = round(train_size * m);
    
    idx = randperm(m)';
    train_idx = idx(1:train_size);
    test_idx = idx(train_size + 1:end);

end