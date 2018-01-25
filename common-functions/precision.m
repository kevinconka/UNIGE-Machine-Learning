function [y, Y] = precision(M)
% received comfusion matrix as input (cols as predictions)

Y = M ./ repmat(sum(M, 1), size(M, 1), 1);
y = diag(Y);

if size(M, 1) == 2; y = y(2); end

end