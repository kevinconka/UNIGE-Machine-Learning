function y = accuracy(M)

y = trace(M) / sum(M(:));

end