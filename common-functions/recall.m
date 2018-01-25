function [y, Y] = recall(M)
% received comfusion matrix as input (cols as predictions)

Y = M ./ repmat(sum(M, 2), 1, size(M, 1));
y = diag(Y);

if size(M, 1) == 2; y = y(2); end

end