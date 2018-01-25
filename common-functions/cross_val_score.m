function [scores, CMats, CVs] = cross_val_score(classifier, X, t, kFolds)

data = struct('X', X, 't', t);

%% get CV indexes

CVs = cell(1, kFolds);          % store CV train and test subsets

if kFolds == size(X, 1) % ------- leave-one-out
    
    for k = 1:kFolds
        CVs{k} = struct('X', data.X(k,:), 't', data.t(k,:));
    end
    
else % -------------------------- k-Fold 
    
    residual = data;            % temporal variable
    for k = 0:(kFolds - 1)
        split_pct = 1/(kFolds - k);
        [CVs{k+1}, residual] = stratified_split(residual.X, residual.t, split_pct);
        % [CVs{k+1}, residual] = train_test_split(residual.X, residual.t, split_pct);
    end
    
end

% reshape from struct to cell array
CVs_X = cellfun(@(x) x.X, CVs, 'UniformOutput', false);
CVs_t = cellfun(@(x) x.t, CVs, 'UniformOutput', false);

%% get all combinations train/validate for CV

combs = combnk(1:kFolds, max(kFolds-1, 1));
% Trained = cell(1, kFolds);
CMats = cell(1, kFolds);

for i = 1:kFolds
    
    train      = struct('X', cat(1, CVs_X{combs(i,:)}), ...
                        't', cat(1, CVs_t{combs(i,:)}) );
    validation = struct('X', cat(1, CVs_X{kFolds - (i-1)}),...
                        't', cat(1, CVs_t{kFolds - (i-1)}) );
                
    classifier.learn(train.X, train.t);             % training step
    y_pred = classifier.predict(validation.X);      % prediction step
    
    y = validation.t;
    % get confusion matrices for further performance metrics
    CMats{i} = confusionmat(y, y_pred);
    
    % store trained classifier
    % Trained{i} = classifier;
    
end

%% get scores of all CV outputs into single struct

scores = cellfun(@get_scores, CMats, 'UniformOutput', false);
scores = struct(...
    'accuracy', cell2mat( cellfun(@(x) x.accuracy, scores, 'UniformOutput', false) ),...
    'sensitivity', cell2mat( cellfun(@(x) x.sensitivity, scores, 'UniformOutput', false) ),...
    'specificity', cell2mat( cellfun(@(x) x.specificity, scores, 'UniformOutput', false) ),...
    'precision', cell2mat( cellfun(@(x) x.precision, scores, 'UniformOutput', false) ),...
    'recall', cell2mat( cellfun(@(x) x.recall, scores, 'UniformOutput', false) ),...
    'F1', cell2mat( cellfun(@(x) x.F1, scores, 'UniformOutput', false) ) ...
    );

end