%% Lab assignment 4: perceptron classifier and evaluation

clear; clc; close all;

addpath(genpath('../datasets'))
addpath(genpath('../common-functions'))
addpath(genpath('../Klassifiers'))

%% test classifier with simple 2D data

data = load('dataset1.txt');
% data = load('dataset2.txt');

X = data(:, 1:end-1);
t = data(:, end);
[m, n] = size(X);       % [samples, features]
fig_n = 1;

figure(fig_n)
plotData(X, t, 'LineWidth', 1.5, 'MarkerSize', 7);
axis([0 1 0 1]);

% rng default         % for reproducibility

percep = perceptronKlassifier();    % create perceptron classifier
% percep.mode = 'offline';
percep.learn(X, t);                 % train classifier
percep.total_epochs

% plot evolution of weight vector
figure(fig_n)
plotDecisionBoundary(percep.w, X, t, 'k-', 'LineWidth', 2.5);

percep.learn(X, t);                 % train classifier

% plot evolution of weight vector
figure(fig_n)
plotDecisionBoundary(percep.w, X, t, 'k-', 'LineWidth', 2.5);
axis square

%% Load semeion digits database

% adapt to MATLAB indexing, class 1 = 1, 2 = 2, ..., 0 = 10
[X, t] = readdigits('semeion.data');

% convert data to [0, 1, 2, 3, ..., 9] form
[t, ~] = find(t'==1); t = mod(t, 10);

% Randomly select 100 data points to display
sel = randperm(size(X, 1));
sel = sel(1:100);

displayData(X(sel, :));

%% Test perceptron classifier (check if all data is linearly separable)

rng default         % for reproducibility
classes = unique(t);
epochs_per_class = zeros(1, 10);

for c = 1:numel(classes)
    
    percep = perceptronKlassifier();
    percep.lr = 0.5;            % learning rate
    percep.max_epochs = 200;    % 1000 by default    
    % percep.mode = 'offline';
    % tic
    percep.learn(X, (t == classes(c)) );
    % toc
    
    y_pred = percep.predict(X);
    
    acc = mean( (t == classes(c)) == y_pred );
    epochs_per_class(c) = percep.total_epochs;
    
    fprintf('[%d epochs] Accuracy for class %d = %.2f\n', ...
        percep.total_epochs, classes(c), acc);
    
end

%% Test k-NN algorithm

rng default         % for reproducibility
[train, test] = stratified_split(X, t, 0.5);
% [train, test] = train_test_split(X, t, 0.70);

knn = kNNKlassifier(10);    % k = 10
knn.weightfcn = 'rank';     % neighbor weight contribution

knn.learn(train.X, train.t);

y_pred = knn.predict(test.X);
acc = mean( test.t == y_pred );

fprintf('Accuracy for %d-nn = %.2f\n', 10, 100*acc);
[M, order] = confusionmat(test.t, y_pred);

figure(2)
plot_confMat(M, order, @F1Score);
axis square
fig = gcf;
fig.Position = fig.Position .* [1 1 0 1] + [0 0 500 0];

%% Tests with CV (varying k-folds)

[m, ~] = size(X);
rng default         % for reproducibility
% kFolds = 2;
kFolds = [2 3 4 5 7 10 15 20 25 30 40 50 75 100 250 531 m];
iters = numel(kFolds);

c = 1;              % binarize with respect to digit '1'
scores = cell(2, iters);

for i = 1:iters
    
    knn = kNNKlassifier(5);             % create knnKlassifier object with k=5
    percep = perceptronKlassifier();    % create perceptronKlassifier object
    
    [train, ~] = stratified_split(X, t, 1);
    
    disp(i);
    [scores{1, i}, ~] = cross_val_score(knn,    train.X, train.t == c, kFolds(i));
    [scores{2, i}, ~] = cross_val_score(percep, train.X, train.t == c, kFolds(i));
    
end

%% Fine tune knn hyper-parameters using CV grid search

rng default             % for reproducibility

% ----------------- MODOFY HERE ----------------------
% initialize hyper-parameters values and CV gridsearch values
weightfcn = {'equal', 'invdist', 'rank'};
K = [1 2 3 4 5 7 10 15 20 25 50 100 200];
iters = 5; kFolds = 5;                          % 5x5CV
% ----------------- ----------- ----------------------

gridsearch = cell(numel(weightfcn), numel(K), iters);    % store results
[train, test] = stratified_split(X, t, 0.5);    % split data

i = 1; j = 1;
for w = weightfcn       % loop through hyper-parameter 'weightfcn' (1st dim)
    for k = K           % loop through hyper-pramater of k's (2nd dim)
        knn = kNNKlassifier(k);
        knn.weightfcn = w{:};
        for l = 1:iters     % run cross validation 'iters' times (3rd dim)
            gridsearch{i, j, l} = cross_val_score(knn, train.X, train.t, kFolds);
        end
        j = j + 1;
    end
    j = 1;
    i = i + 1;
end

%% Extract best hyper-parameters depending on accuracy (grid search)

load CV_gridsearch.mat

% get mean accuracy over the k-folds into matrix format
perf = cellfun(@(x) mean(x.accuracy), gridsearch);
% ** For other metrics such as recall, precision, F1 use template below **
% perf = cellfun(@(x) mean( nanmean(x.F1) ), gridsearch);

perf = mean(perf, 3);                   % avg across 5 ierations (3rd dim)
[~, max_idx] = max(perf(:));            % extract index of maximum performance metric
[w, k] = ind2sub(size(perf), max_idx);  % map index to 2D indexes

fprintf('Best hyperparameters: [%d, %s]\n', K(k), weightfcn{w});

% --------------------- Validation score -----------------------
knn = kNNKlassifier(K(k));
knn.weightfcn = weightfcn{w};

knn.learn(train.X, train.t);
y_pred = knn.predict(test.X);

fprintf('Accuracy on validation set: %.2f\n', mean(test.t == y_pred));

%% k-nn vs perceptron algorithm (binary classification)

[m, ~] = size(X);
classes = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9];

rng default         % for reproducibility
iters = 5;
kFolds = 10;        % 5x10CV
[train, test] = stratified_split(X, t, 1);  % use all data for k-fold CV

scores = cell(2, numel(classes), iters);    % store scores
CMats = cell(2, numel(classes), iters);     % store confusion matrices
for c = 1:numel(classes)        % loop over all classes (2nd dim)
    for i = 1:iters             % loop over X iterations (3rd dim)
        
        knn = kNNKlassifier(10);            % create knnKlassifier object with
        knn.weightfcn = 'rank';             % ... hyper-parameters tuned
        percep = perceptronKlassifier();    % create perceptronKlassifier object
        
        % Extract scores and confusion matrices of each k-fold CV
        [scores{1, c, i}, CMats{1, c, i}] = ...
            cross_val_score(knn,    train.X, train.t == classes(c), kFolds);
        [scores{2, c, i}, CMats{2, c, i}] = ...
            cross_val_score(percep, train.X, train.t == classes(c), kFolds);
    end
    disp(c)
end

save bin_5x10CV_results.mat scores CMats classes

%% Analyze results

load bin_5x10CV_results.mat
% ** to get info on other metrics, change the 'mean(x.metric)' field **

% get average over the k-folds into matrix format (2nd dim)
perf = cellfun(@(x) mean(x.accuracy), scores);

% get average over iterations (3rd dim)
perf = mean(perf, 3);

fprintf('knn score - digit %d-vs-all = %.3f\n', [classes; perf(1,:)]);
fprintf('perceptron score - digit %d-vs-all = %.3f\n', [classes; perf(2,:)]);

%% Test multiclass perceptron

% rng default         % for reproducibility
[train, test] = stratified_split(X, t, 0.5);

percep = perceptronKlassifier();
% percep.mode = 'offline';
percep.learn(train.X, train.t);

y_pred = percep.predict(test.X);
acc = mean( test.t == y_pred );

fprintf('Accuracy = %.2f\n', 100*acc);
[M, order] = confusionmat(test.t, y_pred);

figure(2)
plot_confMat(M, order, @F1Score); axis square
fig = gcf;
fig.Position = fig.Position .* [1 1 0 1] + [0 0 500 0];

%% k-nn vs perceptron algorithm (multiclass classification)

[m, ~] = size(X);

rng default         % for reproducibility
iters = 5;
kFolds = 10;        % 5x10CV
[train, test] = stratified_split(X, t, 1);  % use all data for k-fold CV

scores = cell(2, iters);    % store scores
CMats = cell(2, iters);     % store confusion matrices
for i = 1:iters
    
    knn = kNNKlassifier(10);            % create knnKlassifier object with
    knn.weightfcn = 'rank';             % ... hyper-parameters tuned
    percep = perceptronKlassifier();    % create perceptronKlassifier object
    
    % Extract scores and confusion matrices of each k-fold CV
    [scores{1, i}, CMats{1, i}] = cross_val_score(knn,    train.X, train.t, kFolds);
    [scores{2, i}, CMats{2, i}] = cross_val_score(percep, train.X, train.t, kFolds);
    disp(i);
    
end

save mult_5x10CV_results.mat scores CMats

%% Analyze results

load mult_5x10CV_results.mat

% ** Some metrics contain one value per class, so double average is
% needed [precision, recall, F1, sensitivity, specificity] **
perf = cellfun(@(x) mean( nanmean(x.F1) ), scores);

% get average over iterations (2nd dim)
perf = mean(perf, 2);

fprintf('knn avg score = %.3f\n', perf(1));
fprintf('perceptron avg score = %.3f\n', perf(2));
