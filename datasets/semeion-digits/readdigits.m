function [X, t] = readdigits(filename)
% adapt to MATLAB indexing, class 1 = 1, 2 = 2,..., 0 = 10

rawdata = load(filename);
X = rawdata(:,1:256);
% X = -X + 1;
t = rawdata(:,end-9:end);
clear('rawdata');
t = sign(t(:,[2:end 1])-.5);

end
