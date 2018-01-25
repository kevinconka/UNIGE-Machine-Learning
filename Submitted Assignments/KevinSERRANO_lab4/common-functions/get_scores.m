function scores = get_scores(M)

scores = struct(...
    'accuracy', trace(M) / sum(M(:)),...
    'sensitivity', recall(M), ...
    'specificity', specificity(M), ...
    'precision', precision(M), ...
    'recall', recall(M),...
    'F1', F1Score(M));

end

% for i = 1:m
%     for j = 1:n
%         accuracy(i,j)      = trace(CMats{i,j}) / sum(CMats{i,j}(:));
%         binCMat = binCMats{i,j};
%         sensitivity(i,j,:) = binCMat(2,2,:) ./ ( binCMat(2,2,:) + binCMat(2,1,:) );
%         specificity(i,j,:) = binCMat(1,1,:) ./ ( binCMat(1,1,:) + binCMat(1,2,:) );
%         precision(i,j,:)   = binCMat(2,2,:) ./ ( binCMat(2,2,:) + binCMat(1,2,:) );
%         recall(i,j,:)      = sensitivity(i,j,:);
%         F1                 = 2 * (precision(i,j,:) .* recall(i,j,:)) ./ ...
%                                   (precision(i,j,:) + recall(i,j,:));
%     end
% end