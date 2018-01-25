classdef perceptronKlassifier < handle
    % FILL THIS
    
    properties
        
        % data dependent properties
        numClasses
        numObservations
        classes
        
        % weight vector
        w
        
        % offline or online
        mode = 'online'
        
        % learning rate
        lr = 0.5
        
        % epochs
        max_epochs = 1000
        total_epochs
        % error
        
        % dataset
        raw_X
        raw_t
        % sign_t   % column-vector containing [-1, 1] representation of classes
        
        % some other stuf...
        
    end
    
    % --------------------------------------------------------------------
    
    methods
        
        %% Constructor
        function obj = perceptronKlassifier()
        end
        
        
        
        %% learn perceptron weights (multiclass enabled)
        function learn(obj, X, t, varargin)
            
            [m, n] = size(X);
            obj.numClasses = size(unique(t, 'rows'), 1);
            obj.numObservations = m;
            obj.raw_X = X;
            obj.raw_t = t;
            [obj.classes, ~, ~] = unique(t, 'rows');
            
            if numel(obj.classes) > 2   % if multiclass classification
                % one-vs-all approach (train 'k' perceptrons)
                for c = 1:numel(obj.classes)
                    bin_target = (t == obj.classes(c)); % binarize target [0, 1]
                    [obj.w(:,c), obj.total_epochs(c)] = compute_weights(obj, X, bin_target);
                end
            else
                bin_target = (t == obj.classes(2));     % assume second class is positive
                [obj.w, obj.total_epochs] = compute_weights(obj, X, bin_target);
            end
            
        end
        
        %% learn perceptron weights
        function learn2(obj, X, t, varargin)
            
            [m, n] = size(X);
            obj.numClasses = size(unique(t, 'rows'), 1);
            obj.numObservations = m;
            obj.raw_X = X;
            obj.raw_t = t;
            [obj.classes, ~, num_t] = unique(t, 'rows');
            obj.sign_t = sign(num_t - 1.5);     % [-1, 1]
            
            y = obj.sign_t;                     % [-1, 1]
            % randomly initialize weight vector [-0.01, 0.01]
            obj.w = 0.01 * (2*rand(n+1, 1) - 1);
            
            % data cannot be separated by a hyperplane passing through the
            % origin, therefore we introduce an extra "bias" row
            X = [ones(m, 1) X];
            
            if strcmpi(obj.mode, 'offline')
                % ----------------- OFFLINE LEARNING -----------------------
                % find hyperplane that linearly splits the data
                for epoch = 1:obj.max_epochs
                    
                    y_pred = sign(X * obj.w);               % y_pred = f(x*w)
                    dw = obj.lr * ( X' * (y - y_pred) );    % compute dw
                    obj.w = obj.w + dw;                     % update w
                    
                    if ( norm(y - y_pred) == 0 )
                        % linear separation found (no error)
                        break;
                    end
                    if ( sum( abs(y - y_pred) )/m < 1e-8 )
                        % linear separation not found, threshold reached
                        break;
                    end
                    
                end
                obj.total_epochs = epoch;
                % ---------------------------------------------------------
            else
                % ------------------ ONLINE LEARNING ---------------------
                % find hyperplane that linearly splits the data
                % observation by observation
                epoch = 0;
                l = 1;
                while(epoch < obj.max_epochs)
                    
                    a = sign( X(l,:) * obj.w );
                    delta = 0.5*( y(l) - a );       % desiredOutput - actualOutput
                    dw = obj.lr * delta * X(l,:)';  % compute dw
                    obj.w = obj.w + dw;             % update w
                    
                    l = l + 1;
                    if (l > m)      % whole train set scanned
                        epoch = epoch + 1;
                        y_pred = sign(X * obj.w);
                        obj.error(epoch) = 1 - mean(y == y_pred);
                        if ( y == y_pred ); break; end
                        l = 1;
                    end
                    
                end
                % an iteration over the whole training set is an epoch
                obj.total_epochs = epoch;
                % ---------------------------------------------------------
            end
            
        end
        
        %% Predict class of data X
        function [y_pred] = predict(obj, X)
            
            [m, ~] = size(X);
            X = [ones(m, 1) X]; % extend for bias weight
            
            if numel(obj.classes) > 2
                y_pred = X * obj.w;                 % get weighted sum
                [~, y_pred] = max(y_pred, [], 2);   % choose max value (most confident prediction)
            else
                y_pred = sign(X * obj.w + eps);
            end
            
            % map back to original binary format stored in obj.classes
            [~, ~, y_pred] = unique(y_pred);
            y_pred = obj.classes(y_pred);
            
        end
        
    end
end % end of classdef

function [w, total_epochs] = compute_weights(obj, X, t)
[m, n] = size(X);

y = sign(t - 0.5);     % [-1, 1]
% randomly initialize weight vector [-0.01, 0.01]
w = 0.01 * (2*rand(n+1, 1) - 1);

% data cannot be separated by a hyperplane passing through the
% origin, therefore we introduce an extra "bias" row
X = [ones(m, 1) X];

if strcmpi(obj.mode, 'offline')
    % ----------------- OFFLINE LEARNING -----------------------
    % find hyperplane that linearly splits the data
    for epoch = 1:obj.max_epochs
        
        y_pred = sign(X * w);                   % y_pred = f(x*w)
        dw = obj.lr * ( X' * (y - y_pred) );    % compute dw
        w = w + dw;                             % update w
        
        % error(epoch) = 1 - mean(y == y_pred);
        if ( norm(y - y_pred) == 0 )
            % linear separation found (no error)
            break;
        end
        if ( sum( abs(y - y_pred) )/m < 1e-8 )
            % linear separation not found, threshold reached
            break;
        end
        
    end
    total_epochs = epoch;
    % ---------------------------------------------------------
else
    % ------------------ ONLINE LEARNING ---------------------
    % find hyperplane that linearly splits the data
    % observation by observation
    epoch = 0;
    l = 1;
    while(epoch < obj.max_epochs)
        
        a = sign( X(l,:) * w );
        delta = 0.5*( y(l) - a );       % desiredOutput - actualOutput
        dw = obj.lr * delta * X(l,:)';  % compute dw
        w = w + dw;                     % update w
        
        l = l + 1;
        if (l > m)      % whole train set scanned
            epoch = epoch + 1;
            y_pred = sign(X * w);
            % error(epoch) = 1 - mean(y == y_pred);
            if ( y == y_pred ); break; end
            l = 1;      % start again with obs #1
        end
        
    end
    % an iteration over the whole training set is an epoch
    total_epochs = epoch;
    % ---------------------------------------------------------
end


end

