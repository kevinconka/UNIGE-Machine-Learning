function plot_confMat(varargin)
%PLOTCONFMAT plots the confusion matrix with colorscale, absolute numbers
%   and precision normalized percentages
%
%   usage: 
%   PLOTCONFMAT(confmat) plots the confmat with integers 1 to n as class labels
%   PLOTCONFMAT(confmat, labels) plots the confmat with the specified labels
%
%   Vahe Tshitoyan
%   20/08/2017
%
%   Arguments
%   confmat:            a square confusion matrix
%   labels (optional):  vector of class labels

% number of arguments
switch (nargin)
    case 0
       confmat = 1;
       labels = {'1'};
    case 1
       confmat = varargin{1};
       labels = 1:size(confmat, 1);
    case 2
       confmat = varargin{1};
       labels = varargin{2};
    otherwise
       confmat = varargin{1};
       labels = varargin{2};
       fcnHandle = varargin{3};
end

confmat(isnan(confmat)) = 0;    % in case there are NaN elements
numlabels = size(confmat, 1);   % number of labels

% calculate accuracy
accuracy = 100 * trace(confmat)/sum(confmat(:));

% perform extra performance metrics on confmat (recall, precision, F1)
if nargin >= 3
    [~, confmat] = fcnHandle(confmat);
    confmat = 100 * confmat;
end

% plotting the colors
imagesc(confmat);

% add titles and axis-labels
title(['Accuracy: ', sprintf('%.2f', accuracy), '\%'], 'Interpreter', 'latex');
ylabel('Target Class', 'Interpreter', 'latex'); 
xlabel('Predicted Class', 'Interpreter', 'latex');

% set the colormap
lowRGB = [255 255 255]; midRGB = [178 255 102]; highRGB = [0 204 0];
cmap = interp1([0 0.1 1], [lowRGB; midRGB; highRGB], linspace(0,1,100)) ./ 255;
colormap(cmap);
% MAP = colormap(flipud(hot));

% Create strings from the matrix values and remove spaces
% textStrings = num2str([confpercent(:), confmat(:)], '%.1f%%\n%d\n');
if nargin >= 3
    textStrings = num2str(confmat(:), '%.1f');
else
    textStrings = num2str(confmat(:), '%d');
end
textStrings = strtrim(cellstr(textStrings));

% Create x and y coordinates for the strings and plot them
[x,y] = meshgrid(1:numlabels);
hStrings = text(x(:), y(:), textStrings(:), ...
    'HorizontalAlignment','center', 'Interpreter', 'latex');

% % Get the middle value of the color range
% midValue = mean(get(gca, 'CLim'));
% 
% % Choose white or black for the text color of the strings so
% % they can be easily seen over the background color
% textColors = repmat(cmap(end, :), numlabels*numlabels, 1);
% textColors(confpercent(:) > midValue, :) = repmat(cmap(1, :), sum(confpercent(:) > midValue), 1);
% % textColors = repmat(confpercent(:) > midValue, 1, 3);
% set(hStrings, {'Color'}, num2cell(textColors, 2));

% Setting the axis labels
set(gca,...
    'XTick', 1:numlabels, ...
    'XTickLabel', labels, ...
    'YTick', 1:numlabels, ...
    'YTickLabel', labels, ...
    'TickLength',[0 0], ...
    'FontSize', 12, ...
    'TickLabelInterpreter', 'latex');

end