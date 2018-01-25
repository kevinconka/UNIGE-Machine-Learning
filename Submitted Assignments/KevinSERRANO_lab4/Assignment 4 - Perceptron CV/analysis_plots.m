%% Plot results

clear; clc; close all;

addpath(genpath('../datasets'))
addpath(genpath('../common-functions'))

%%  Plot results of CV performances

load k-foldCV_scores.mat

% ----------------- MODIFY HERE ----------------------
% k-folds -> [2 3 4 5 7 10 15 20 25 30 40 50 75 100 250 531 1593] 
ks = logical([1 1 0 0 1 1  1  1  1  1  1  1  1  1   1   1   0]);     
s = 1;           % 1 = knn, 2 = perceptron
% ----------------- ----------- ----------------------

figure(4)

score_avg = cellfun(@(x) nanmean(x.accuracy), scores(:, ks));
score_std = cellfun(@(x) nanstd(x.accuracy), scores(:, ks));
yyaxis left
plot(1:sum(ks), score_avg(1,:), 1:sum(ks), score_avg(2,:), 'LineWidth', 2);
ylim([0.88 1])
ylabel('accuracy', 'Interpreter', 'latex')

yyaxis right
plot(1:sum(ks), score_std(1,:), 1:sum(ks), score_std(2,:), 'LineWidth', 2);
ylabel('standard deviation ($\sigma$)', 'Interpreter', 'latex')

title('CV results', 'Interpreter', 'latex');
xlabel('$k$-Folds', 'Interpreter', 'latex');
% ylabel('F1 score', 'Interpreter', 'latex');
legend({'$k$-NN', 'perceptron'}, 'Interpreter', 'latex', 'Location', 'SouthWest', 'FontSize', 12);
grid on
set(gca,...
    'FontSize',12,...
    'TickLabelInterpreter', 'latex',... %);
    'XLim', [1 sum(ks)],...
    'XTick', 1:sum(ks),...
    'XTickLabelRotation', 60,...
    'XTickLabel', kFolds(ks));

%% Percentage of outliers based on LOOCV (k = 1593)

load k-foldCV_scores.mat
LOOCV_avg = cellfun(@(x) nanmean(x.accuracy), scores(:, end));
LOOCV_std = cellfun(@(x) nanstd(x.accuracy), scores(:, end));

T = table(LOOCV_avg, LOOCV_std, 'VariableNames',{'Accuracy','std'},...
    'RowNames',{'k-NN','perceptron'})

%% Fine-tuning k-nn results

load CV_gridsearch.mat

acc = cellfun(@(x) mean( x.accuracy ), gridsearch);
% The following metrics contain a value per class, so double average is needed
pre = cellfun(@(x) mean( nanmean(x.precision) ), gridsearch);
rec = cellfun(@(x) mean( nanmean(x.recall) ), gridsearch);
f1s = cellfun(@(x) mean( nanmean(x.F1) ), gridsearch);

% average accross iterations (3rd dim)
acc = mean(acc, 3); pre = mean(pre, 3); rec = mean(rec, 3); f1s = mean(f1s, 3);

% [a, b] = max(cat(3, acc, pre, rec, f1s), [], 2);

%% Graph plot
% ----------------- MODOFY HERE ----------------------
res2plot = 100*acc;
% ----------------- ----------- ----------------------

[max_y, max_x] = max(res2plot, [], 2);
[m, n] = size(res2plot);

figure(9);
h = plot(1:n, res2plot, 'LineWidth', 2);
hold on
for i = 1:m
    plot(max_x(i), max_y(i), 'x', 'LineWidth', 2, 'Color', h(i).Color, 'MarkerSize', 10)
end
text(max_x(i), max_y(i), num2str(max_y(i)),...
    'Interpreter', 'latex', 'FontSize', 12, 'VerticalAlignment', 'bottom')
hold off
grid on

legend({'\texttt{equal}', '\texttt{invdist}', '\texttt{rank}'}, ...
    'Interpreter', 'latex', 'Location', 'SouthWest', 'FontSize', 12);
title('CV gridsearch', 'Interpreter', 'latex');
ylabel('accuracy score (\%)', 'Interpreter', 'latex'); 
xlabel('$k$-nn', 'Interpreter', 'latex');
set(gca,...
    'XLim', [1 n],...
    'XTick', 1:n, ...
    'XTickLabel', K, ...
    'FontSize', 12, ...
    'TickLabelInterpreter', 'latex');

%% Table-like plot
% ----------------- MODOFY HERE ----------------------
res2plot = 100*acc;
% ----------------- ----------- ----------------------

[~, max_idx] = max(res2plot(:));
figure(10); imagesc(res2plot')
[m, n] = size(res2plot');
colormap autumn

% add 'grid'
hold on
plot([0; n]+0.5, [1;1]*[1:m-1]+0.5, '-k')   % horizontal lines
plot([1;1]*[1:n-1]+0.5, [0; m]+0.5, '-k')   % vertical lines
hold off;

% Create strings from the matrix values and remove spaces
% textStrings = num2str([confpercent(:), confmat(:)], '%.1f%%\n%d\n');
textStrings = num2str(res2plot(:), '%.1f');
textStrings = strtrim(cellstr(textStrings));
textStrings{max_idx} = sprintf(['\\textbf{%s}'], textStrings{max_idx});

% Create x and y coordinates for the strings and plot them
[x,y] = meshgrid(1:n, 1:m); x = x'; y = y';
hStrings = text(x(:), y(:), textStrings(:), ...
    'HorizontalAlignment','center', 'Interpreter', 'latex', 'FontSize', 12);

title('Accuracy \%', 'Interpreter', 'latex');
ylabel('$k$', 'Interpreter', 'latex'); 
xlabel('\texttt{weights}', 'Interpreter', 'latex');
% Setting the axis labels
set(gca,...
    'XTick', 1:n, ...
    'XTickLabel', weightfcn, ...
    'YTick', 1:m, ...
    'YTickLabel', K, ...
    'TickLength',[0 0], ...
    'FontSize', 12, ...
    'TickLabelInterpreter', 'latex');

%% K-nn vs perceptron binary classification

load bin_5x10CV_results.mat

% get average over the k-folds into matrix format (2nd dim)
acc = cellfun(@(x) mean(x.accuracy), scores);
pre = cellfun(@(x) mean(x.precision), scores);
rec = cellfun(@(x) mean(x.recall), scores);
f1s = cellfun(@(x) mean(x.F1), scores);

acc = mean(acc, 3); pre = mean(pre, 3); rec = mean(rec, 3); f1s = mean(f1s, 3);%

% [a, b] = max(cat(3, acc, pre, rec, f1s), [], 2);
% [a, b]

% ----------------- MODOFY HERE ----------------------
model = 1;     % 1 = knn, 2 = perceptron
res2plot = [acc(model, :); pre(model,:); rec(model,:); f1s(model,:)];
res2plot = [res2plot mean(res2plot, 2)];
% ----------------- ----------- ----------------------

% [~, max_idx] = max(res2plot(:));
figure(11); imagesc(res2plot')
[m, n] = size(res2plot');

% set the colormap
if model == 1
lowRGB = [255 102 102]; medRGB = [255 178 102]; highRGB = [255 255 255]; % redish
else
lowRGB = [102 198 255]; medRGB = [102 255 255]; highRGB = [255 255 255]; % blueish
end
cmap = interp1([0 0.5 1], [lowRGB; medRGB; highRGB], linspace(0,1,64)) ./ 255;
colormap(cmap);

% add 'grid'
hold on
plot([0; n]+0.5, [1;1]*[1:m-1]+0.5, '-k')   % horizontal lines
plot([1;1]*[1:n-1]+0.5, [0; m]+0.5, '-k')   % vertical lines
plot([0; n]+0.5, [1;1]*[m-1]+0.5, '-k', 'LineWidth', 3)   % bold line
hold off;

% Create strings from the matrix values and remove spaces
% textStrings = num2str([confpercent(:), confmat(:)], '%.1f%%\n%d\n');
textStrings = num2str(res2plot(:), '%.3f');
textStrings = strtrim(cellstr(textStrings));
% textStrings{max_idx} = sprintf(['\\textbf{%s}'], textStrings{max_idx});

% Create x and y coordinates for the strings and plot them
[x,y] = meshgrid(1:n, 1:m); x = x'; y = y';
hStrings = text(x(:), y(:), textStrings(:), ...
    'HorizontalAlignment','center', 'Interpreter', 'latex', 'FontSize', 12);

title('$k$-nn performance', 'Interpreter', 'latex');
ylabel('digit (one-vs-all)', 'Interpreter', 'latex'); 
xlabel('metric', 'Interpreter', 'latex');
% Setting the axis labels
set(gca,...
    'XTick', 1:n, ...
    'XTickLabel', {'accuracy', 'precision', 'recall', 'F1'}, ...
    'YTick', 1:m, ...
    'YTickLabel', {'0','1','2','3','4','5','6','7','8','9','avg.'}, ...
    'TickLength',[0 0], ...
    'FontSize', 12, ...
    'TickLabelInterpreter', 'latex');

%% K-nn vs perceptron multiclass classification

load mult_5x10CV_results.mat
m = numel(classes);

% avg confusion matrices over CV k-folds
avgCMats = cellfun(@(x) mean( reshape(cell2mat(x), [m m kFolds]), 3), CMats, 'un', 0);

avgCMats = cell2mat( cat(3, avgCMats(1,:), avgCMats(2,:)) );    % unfold matrices, 3rd dim = model
avgCMats = mean( reshape(avgCMats, [m m iters 2]), 3);          % reshape and average over # iters

knn_CMat = avgCMats(:,:,:,1);
prc_CMat = avgCMats(:,:,:,2);

figure(20)
plot_confMat(knn_CMat, classes, @F1Score)
title(['$k$-nn avg. accuracy: ', sprintf('%.2f', 100*accuracy(knn_CMat)), '\%'], ...
    'Interpreter', 'latex'); axis square
fig = gcf;
fig.Position = fig.Position .* [1 1 0 1] + [0 0 500 0];

figure(21) 
plot_confMat(avgCMats(:,:,2), classes, @F1Score)
title(['Perceptron avg. accuracy: ', sprintf('%.2f', 100*accuracy(prc_CMat)), '\%'], ...
   'Interpreter', 'latex'); axis square
fig = gcf;
fig.Position = fig.Position .* [1 1 0 1] + [0 0 500 0];

%% ---------------- EXTRA MATERIAL -------------------

% To use in analysis of CV performance when varying k-folds
% ------------------- BOX PLOT ----------------------
% % create grouping indexes
% grp = [];
% for i = find(ks)
%     grp = [grp, kFolds(i) * ones(1, kFolds(i))];
% end
% merged_scores = merge_scores(scores(s, ks), 'print', 'no');
% boxplot(merged_scores.accuracy, grp); %, 'LabelOrientation', 'inline');
% M = findobj('Tag', 'Median');
% for median = M'
%     median.LineWidth = 2.5;
% end
% hold on
% plot(1:sum(ks), cellfun(@(x) nanmean(x.accuracy), scores(s, ks)), '-og');
% hold off
% ------------------- -------- ----------------------
