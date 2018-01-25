function y = specificity(M)

tn = trace(M) - diag(M);
fn = sum(M, 1)' - diag(M);
y = tn ./ (tn + fn);

if size(M, 1) == 2; y = y(2); end

end