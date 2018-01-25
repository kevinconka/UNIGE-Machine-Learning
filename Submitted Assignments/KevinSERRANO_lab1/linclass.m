function y = linclass(x,w)
% linear threshold classifier
%   x: n x m data set to classify
%   w: 1 x m coefficient vector of hyperplane
  y = x * w' > 0;
end

% this source code MUST be saved in a separate file named linclass.m

% typing 'help linclass' at the prompt will print the three lines of comments
% that follow the function header