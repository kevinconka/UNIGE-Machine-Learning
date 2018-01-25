function [] = plotData(X, l, varargin)

plot(X((l == 1), 1), X((l == 1), 2), 'r+', varargin{:}); hold on
plot(X((l == -1), 1), X((l == -1), 2), 'bo', varargin{:});  hold off
grid on

xlabel('$x_1$', 'Interpreter', 'latex');
ylabel('$x_2$', 'Interpreter', 'latex');
set(gca,...clc
        'FontSize',12,...
        'TickLabelInterpreter', 'latex');

end