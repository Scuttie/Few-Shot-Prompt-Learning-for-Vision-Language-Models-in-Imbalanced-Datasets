function [confusion, results] = evaluation(datasetPath, split, images, labels, scores)
% EVALUATION Evaluate classification results
%   CONFUSION = EVALUATION(DATASETPATH, SPLIT, IMAGES, LABELS, SCORES)
%   evaluate the classification results IMAGES,LABELS,SCORES on the
%   data split SPLIT.
%
%   IMAGES, LABELS and SCORES are one-dimensional arrays of the same length,
%   specifying a number of (image,lable,score) triplets. IMAGES and
%   LABELS can be either cell arrays of strings, with the name of
%   images and airplane models respectively, or indexes in the list of
%   images (images.txt) and evaluated airplane models
%   (models_evaluated.txt) -- the latter option is generally more
%   efficient. SCORES is a numeric array containing the score of the
%   corresponding predictions.
%
%   CONFUSION is confusion matrix, with one row
%   per ground truth class and one column per estimated class. The
%   average accuracy is simply the average of the diagonal of the confusion.
%
%   [~,RESULTS] = EVALUATION() returns an additional struct array with
%   one entry for each evaluated class. It has the following
%   fields:
%
%   RESULTS.RC - Recall
%   RESULTS.PR - Precision
%   RESULTS.TP - True positive rate
%   RESULTS.TN - True negative rate
%   RESULTS.AP - Average precision
%   RESULTS.ROCEER - ROC Equal Error Rate

% Author: Andrea Vedaldi

% Copyright (C) 2013 Andrea Vedaldi
% This code is released in the public domain.

% Get the ground truth image list and labels for the set.

[images0, labels0] = textread(fullfile(datasetPath, ['images_' split '.txt']), '%7s%*1s%s', 'delimiter', '\n', 'whitespace', '') ;
[classes0, ~, y0] = unique(labels0) ;

% Convert character labels to indexes. Images and ground truth classes
% are assigned a number in the same order as the training data.

ok = true(size(labels)) ;
if isnumeric(labels)
  y = labels ;
else
  [~,y] = ismember(labels, classes0) ;
  if any(y == 0)
    for i = find(y == 0)
      warning('Class %s not found in set of ground truth classes\n', labels{i}) ;
      ok(i) = false ;
    end
  end
end

if isnumeric(images)
  x = images ;
else
  [~, x] = ismember(images, images0) ;
  if any(y == 0)
    for i = find(y == 0)
      warning('Image %s was not found in set of ground truth images\n', images{i}) ;
      ok(i) = false ;
    end
  end
end
y0 = y0' ;
y = y(ok)' ;
x = x(ok)' ;

numImages = numel(images0) ;
numClasses = numel(classes0) ;

fprintf('%s: %s split, %d classes, %d images\n', mfilename, split, numClasses, numImages) ;

% Iterate over predicted classes. For each, initialize all prediction
% scores for all images to -infinity. Then, replace the score for
% those image-label pairs that appear in the input.

scorem = -inf(numClasses, numImages) ;
for y1 = 1:numClasses
  scorem(y1, x(y == y1)) = scores(y == y1) ;

  [rc,pr,info] = vl_pr(2 * (y0 == y1) - 1, scorem(y1, :), 'IncludeInf', false) ;
  results(y1).rc = rc ;
  results(y1).pr = pr ;
  results(y1).ap = info.ap ;

  [tp,tn,info] = vl_roc(2 * (y0 == y1) - 1, scorem(y1, :), 'IncludeInf', false) ;
  results(y1).tp = tp ;
  results(y1).tn = tn ;

  results(y1).roceer = info.eer ;
  results(y1).name = classes0{y1} ;
  results(y1).numGtSamples = sum(y0 == y1) ;
  results(y1).numCandidates = sum(y == y1) ;

  fprintf('%s: %25s [%5d gt,%5d cands] AP %5.2f%%, ROC-EER %5.2f%%\n', ...
    mfilename, ...
    results(y1).name, ...
    results(y1).numGtSamples, ...
    results(y1).numCandidates, ...
    results(y1).ap * 100, ...
    results(y1).roceer * 100) ;
end

confusion = zeros(numClasses) ;
[~, preds] = max([-inf(1, numImages) ; scorem]) ;
preds = preds - 1 ;

for y1 = 1:numClasses
  z = accumarray(preds(preds > 0 & y0 == y1)', 1, [numClasses 1])' ;
  z = z/results(y1).numGtSamples ;
  confusion(y1,:) = z ;
end

fprintf('%s: mean accuracy: %.2f %%\n', mfilename, mean(diag(confusion))*100) ;
