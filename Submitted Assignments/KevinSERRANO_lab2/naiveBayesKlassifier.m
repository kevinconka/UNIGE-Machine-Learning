classdef naiveBayesKlassifier < handle
    % This class is a naive bayes classifier model working only with multivariate
    % multinomial distribution. In this kind of distribution the observations
    % are categorical.
    %
    % The steps performed by this classifier are:
    %   1. Record the distinct categories represented in the observations of
    %      the entire predictor and transfrom from categorical to numerical.
    %   2. Separate the observations by response class.
    %   3. For each response class, fit a multinomial model using the category
    %      relative frequencies and total number of observations.
    %
    % For each predictor you model with a multivariate multinomial distribution,
    % the naive Bayes classifier:
    %   - Records a separate set of distinct predictor levels for each predictor
    %   - Computes a separate set of probabilities for the set of predictor
    %     levels for each class.
    
    properties
        
        % data dependent properties
        predictorNames
        targetName
        classNames
        numObservations
        
        % numerical data
        categoricalLevels
        numericalLevels
        numericalPredictor
        numericalTarget
        
        % flag for smoothing (default = YES)
        smoothing = 'Yes'
        
        % Cell array containing conditional probabilities for each category
        priorProbs
        likelihood

        % some other stuf...
        
    end
    
    % --------------------------------------------------------------------
    
    methods
        
        %% Constructor
        function obj = naiveBayesKlassifier()
        end
        
        %% Extract information from the categorical data and transform it to numerical
        function cat2num(obj, Table)
            
            [m, n] = size(Table);
            obj.numObservations = m;
            obj.predictorNames = Table.Properties.VariableNames(1:n - 1);
            obj.targetName = Table.Properties.VariableNames(end);
            
            data = table2array(Table);      % table to categorical array
            obj.classNames = unique( data(:, end) )';
            obj.categoricalLevels = {};
            numData = zeros(m, n);          % allocate memory for numerical data
            
            for i = 1:n
                data_i = data(:, i);
                catLevels = cellstr( unique(data_i) );  % get categorical levels
                numLabels = zeros(m, 1);                % allocate memory
                for j = 1:numel( catLevels )
                    numLabels( data_i == catLevels(j) ) = j;    % categorical -> numerical
                end
                numData(:, i) = numLabels;              % add to pre-allocated matrix
                
                if i < n                    % if not part of the target
                    obj.categoricalLevels{i} = catLevels;
                    obj.numericalLevels{i} = unique(numLabels);
                end
            end
            
            obj.numericalPredictor = numData(:, 1:(n-1) );
            obj.numericalTarget = numData(:, end);
            
        end
        
        %% fit model and calculate mvmn probabilities
        function fit(obj, Table)
            
            % categorical -> numerical
            obj.cat2num(Table);
            
            k_classes = numel(obj.classNames);
            n_features = numel(obj.predictorNames);
            obj.likelihood = {};
            for k = 1:k_classes
                k_predictors = obj.numericalPredictor(k == obj.numericalTarget, :);
                n_k = size(k_predictors, 1);    % number of observations in class k
                % --------- compute prior probabilities ------------------
                obj.priorProbs(k) = n_k / obj.numObservations;
                % --------------------------------------------------------
                for f = 1:n_features
                    n_j = numel(obj.numericalLevels{f});
                    for l = 1:n_j
                        % ------ compute conditional probabilities -------
                        if ( strcmpi(obj.smoothing, 'yes') )
                            obj.likelihood{k, f}(l) = ...
                                (1 + sum(l == k_predictors(:, f))) / (n_j + n_k);
                        else
                            obj.likelihood{k, f}(l) = ...
                                sum(l == k_predictors(:, f)) / n_k;
                        end
                        % ------------------------------------------------
                    end
                end
            end
            
        end
        
        %% Predict class of data X
        function [y_pred, acc, g] = predict(obj, X)
            
            [m, n] = size(X);
            acc = NaN;
            
            % check for errors in matrix dimensions
            n_preds = numel(obj.predictorNames);
            if (n < n_preds || (n > n_preds + 1))
                error('Dimension missmatch');
            end
            
            data = table2array(X);
            g = repmat(obj.priorProbs, m, 1);   % store max-likelihood probs           
            for k = 1:numel(obj.classNames) 
                for i = 1:m                     % loop through observations in test set
                    for f = 1:n_preds
                        l = contains( obj.categoricalLevels{f}, string(data(i, f)) );
                        % --------- compute maximum likelihood -----------
                        try
                            g(i, k) = g(i, k) * obj.likelihood{k, f}(l);
                        catch
                            warning('Unknown attribute value. Skipping...');
                            g(i, k) = NaN;
                        end
                        % ------------------------------------------------
                    end
                end
            end
            [~, y_pred] = max(g, [], 2);        % get index of the highest probability per obs
            y_pred = obj.classNames(y_pred);    % numerical -> categorical
            y_pred = y_pred(:);                 % ensure column vector
            
            if (n == n_preds + 1)               % compute accuracy of prediction
                y = data(:, end);
                acc = mean( y == y_pred );
                g = [g, y == y_pred];
            end
            
        end
        
    end
end