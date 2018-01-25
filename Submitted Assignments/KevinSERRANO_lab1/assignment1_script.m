%% ============== Assignment 1 - linear-threshold classifier ==============
clear; clc; clf;

%% Plot data

% data = load('dataset1.txt');
% data = load('dataset2.txt');
data = load('iris-2class.txt');
X = data(:, 1:end-1);
y = data(:, end);
[m, n] = size(X);       % [samples, features]

% optional feature scaling (pre-processing)
% [X, ~, ~] = featureNormalize(X);

figure(1)
plotData(X, y, 'LineWidth', 1.5, 'MarkerSize', 7); 
% axis([0 1 0 1]);

% data cannot be separated by a line passing through the origin, therefore
% we introduce an extra "bias" row
X = [ones(size(X, 1), 1) X];

%% Choosing line and plotting

% w = [-0.91 0.75 1];     % dataset1
% w = [-1 0.73 1];        % dataset2
w = [-2 -1 8];          % iris2

w = w / norm(w);
y_hat = linclass(X, w);

figure(1)
plotData(X(:, 2:end), y, 'LineWidth', 1.5, 'MarkerSize', 7); 
plotDecisionBoundary(w, X, y, 'k-');
% axis([0 1 0 1]);

%% Random perturbations

rng default                     % For reproducibility
P = linspace(0.01, 0.1, 10);    % vector of perturbation coeffs
num_iters = 200;
results = zeros(size(P, 2), num_iters);

i = 0;
for p = P 
    plotData(X(:, 2:end), y, 'LineWidth', 1.5, 'MarkerSize', 6);
    plotDecisionBoundary(w, X, y, '-');
    % axis([0 1 0 1]);
    i = i + 1;
    for j = 1:num_iters     
        r = rand(1, 3)*2 - 1;   % random numbers from [-1 to 1]
        r = r / norm(r);        % normalize
        w_p = w + p*r;          % add noise to hyperplane
        plotDecisionBoundary(w_p, X, y , '--'); 
        y_hat = linclass(X, w_p) ;   % predict class
        y_hat = -2*y_hat + 1;       % map to match the classes -1 and 1
        results(i, j) = sum(y ~= y_hat);    % get the errors
        % pause
    end
    % pause
end

%% Plot results for analysis

figure(2) % display results info in image form
imagesc(results'); colorbar; colormap hot;
xlabel('perturbation size $p$', 'Interpreter', 'latex');
ylabel('random trial', 'Interpreter', 'latex');
set(gca,...clc
        'FontSize',12,...
        'TickLabelInterpreter', 'latex',...
        'XTick',[1 2 3 4 5 6 7 8 9 10],...
        'XTickLabel', {'0.01', '0.02', '0.03', '0.04', '0.05', '0.06',...
                        '0.07', '0.08', '0.09', '0.1'});
                    
figure(3)   % average vs perturbation size
plot(P, mean(results, 2)); grid on;
xlabel('perturbation size $p$', 'Interpreter', 'latex');
ylabel('average number of errors', 'Interpreter', 'latex');
set(gca,...
        'FontSize',12,...
        'TickLabelInterpreter', 'latex',...
        'XTick', P, 'XLim', [min(P), max(P)]);

figure(4)   % box plot
% http://onlinestatbook.com/2/graphing_distributions/boxplots.html
boxplot(results', 'Labels', cellfun(@num2str, num2cell(P), 'UniformOutput', false));
hold on;
plot([1:numel(P)], mean(results, 2), 'dk'); 
hold off;
legend('average');
grid on;
xlabel('perturbation size $p$', 'Interpreter', 'latex');
ylabel('number of errors', 'Interpreter', 'latex');
set(gca,...
        'FontSize',12,...
        'TickLabelInterpreter', 'latex');
M = findobj('Tag', 'Median');
for median = M'
    median.LineWidth = 2.5;
end

%% Plot average number of errors vs perturbation size p for all datasets

load('avgs.mat');

figure(5)   % average vs perturbation size
plot(P, avgs, 'LineWidth', 2); grid on;
legend({'dataset1', 'dataset2', 'iris-2class'}, 'Interpreter', 'latex');
xlabel('perturbation size $p$', 'Interpreter', 'latex');
ylabel('average number of errors', 'Interpreter', 'latex');
set(gca,...
        'FontSize',12,...
        'TickLabelInterpreter', 'latex',...
        'XTick', P, 'XLim', [min(P), max(P)]);
    
figure(6)   % average vs perturbation size
plot(P, avgs ./ [7 7 150], 'LineWidth', 2); grid on;
legend({'dataset1', 'dataset2', 'iris-2class'}, 'Interpreter', 'latex');
xlabel('perturbation size $p$', 'Interpreter', 'latex');
ylabel('average percentage of errors', 'Interpreter', 'latex');
set(gca,...
        'FontSize',12,...
        'TickLabelInterpreter', 'latex',...
        'XTick', P, 'XLim', [min(P), max(P)]);
                    
