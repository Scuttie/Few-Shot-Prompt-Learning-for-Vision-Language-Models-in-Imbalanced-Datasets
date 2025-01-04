% Demonstrates the use of the EVALUATION() functions.

% choose a task-set combination
split = 'variant_test' ;
%split = 'variant_trainval' ;
%split = 'family_test' ;
%split = 'manufacturer_test' ;

switch 1
  case 1
    % Example 1: the evaluation set contains exactly one image-label pair
    images = {'0900914'} ;
    labels = {'747-400'} ;
    scores = 1 ;
  case 2
    % Example 2: the evaluation set contains exactly all the ground truth image-label pairs (perfect
    % performance).
    [images, labels] = textread(fullfile('data', ['images_' split '.txt']), '%7s%*1s%s', 'delimiter', '\n', 'whitespace', '') ;
    scores = ones(size(labels)) ;
  case 3
    % Example 3: the evaluation set contains all the possible
    % image-label pair and random scores. Numeric inputs are used
    % for efficiency.
    [images0, labels0] = textread(fullfile('data', ['images_' split '.txt']), '%7s%*1s%s', 'delimiter', '\n', 'whitespace', '') ;
    n = numel(images0) ;
    clear images labels scores ;
    for ci = 1:100
      images{ci} = 1:n ;
      labels{ci} = repmat(ci,1,n) ;
      scores{ci} = randn(1,n) ;
    end
    images = [images{:}] ;
    labels = [labels{:}] ;
    scores = [scores{:}] ;
end

[confusion, results] = evaluation('data', split, images, labels, scores) ;

figure(1) ; clf ;
imagesc(confusion) ; axis tight equal ;
xlabel('predicted') ;
ylabel('ground truth') ;
title(sprintf('mean accuracy: %.2f %%\n', mean(diag(confusion))*100)) ;
