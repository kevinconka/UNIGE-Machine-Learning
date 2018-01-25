classdef kNNKlassifier < handle
    % FILL THIS
    
    properties
        
        % data dependent properties
        numClasses
        numObservations
        classes
        
        % k-nn neighbors
        k
        
        % neighbor weight contribution
        weightfcn = 'equal'       % no weighting (default)
        
        % type of distance computation
        distance = 'Euclidean'  % by default
        
        % dataset
        raw_X
        raw_t
        num_t   % column-vector containing numerical representation of classes
        
        % some other stuf...
        
    end
    
    % --------------------------------------------------------------------
    
    methods
        
        %% Constructor
        function obj = kNNKlassifier(k)
            obj.k = k;
        end
        
        %% fit model and calculate mvmn probabilities
        function learn(obj, X, t, varargin)
            
            obj.numClasses = size(unique(t, 'rows'), 1);
            obj.numObservations = size(X, 1);
            obj.raw_X = X;
            obj.raw_t = t;
            [obj.classes, ~, obj.num_t] = unique(t, 'rows');
            
            if nargin > 3
                obj.distance = varargin{1};
            end
            
        end
        
        %% Predict class of data X
        function [y_pred, k_nn] = predict(obj, X)
            
            [m, ~] = size(X);
            y_pred = zeros(m, 1);
            
            for i = 1:m
                X_i = X(i,:);
                dist2 = sum( (X_i - obj.raw_X).^2, 2 ); % compute euclidean dist2
                [D, I] = sort(dist2);                   % sort from min dist to max dist
                
                % get the respective k-nn (class, distance, rank)
                k_nn = [obj.num_t( I(1:obj.k) ), D(1:obj.k), (1:obj.k)'];
                
                % get weighted contributions of each neighbor
                w = zeros(1, obj.numClasses);
                for c = 1:obj.numClasses
                    idx = (k_nn(:, 1) == c);
                    switch obj.weightfcn
                        case 'equal'     % default value
                            w(c) = sum( idx );
                        case 'invdist'
                            w(c) = sum( 1 ./ k_nn(idx, 2) );
                        case 'rank'
                            w(c) = sum( 1 ./ k_nn(idx, 3) );
                        otherwise
                            error('Unknown neighbor weight contribution');
                    end
                end
                
                % extract index corresponding to the most nn.
                [~, y_pred(i)] = max(w);           
            end
            
            y_pred = obj.classes(y_pred, :);    % mapping to original target form
            
        end
        
        %% Predict class of data X using 3D operations
        function [y_pred, dist] = predict3D(obj, X, k)
            
            [m, n] = size(X);
            
            X = reshape(X', [1 n m]);       % reshape to 3D matrix (1 obs in each dimension)
            Y = repmat(obj.raw_X, [1 1 m]); % repeat into 3D matrix
            
            switch(obj.distance)            % compute all distances and store in 3D mat
                case 'Manhattan'
                    dist = sum(abs(X - Y), 2); 
                otherwise % Euclidean
                    dist = sqrt(sum((X - Y).^2 , 2));
            end
      
            dist = reshape(dist, [obj.numObservations m]);    % 3D -> 2D
            [~, I] = sort(dist);            % sort from min to max dist and extract indexes            
            
            y_pred = zeros(m, 1);           % allocate memory
            for i = 1:m
                k_nn = obj.num_t( I(1:k, i) );          % get the respective k-nn
                [N, edges] = histcounts(k_nn);          % get nn counts
                [~, idx] = max(N);                      % extract index corresponding to the most nn.
                y_pred(i) = mean( edges(idx:idx+1) );   % get class corresponding to index
            end
            
            y_pred = obj.classes(y_pred, :);    % mapping to original target form
            
        end
        
    end
end