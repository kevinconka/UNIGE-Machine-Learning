function [y, Y] = F1Score(M)

[~, P] = precision(M);
[~, R] = recall(M);

Y = 2 * (P .* R) ./ (P + R); Y(isnan(Y)) = 0;
y = diag(Y);

if size(M, 1) == 2; y = y(2); end

end