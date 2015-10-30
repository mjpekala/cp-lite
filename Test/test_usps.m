% Before running this script, be sure to download the USPS data set
% to pwd (getUSPS.sh will do this for you).

% oct 2015, mjp

addpath('..');
rng(9999);

load zip.train;
train.y = zip(:,1);
train.X = zip(:,2:end);
clear zip;

load zip.test;
test.y = zip(:,1);
test.X = zip(:,2:end);
clear zip;

% There's a known bias in the USPS data set.  We can remove this bias
% by reshuffling train and test.
%
% Disable this if you wish to retain the original train/test split.
fixBias = 1;
if fixBias
    fprintf('[%s] reshuffling train and test to restore exchangeability...\n', mfilename);
    nTrain = length(train.y);
    Z = [train.y train.X ; test.y test.X];
    Z = Z(randperm(size(Z,1)), :);
    
    train.y = Z(1:nTrain, 1);
    train.X = Z(1:nTrain, 2:end);
    test.y = Z(nTrain+1:end, 1);
    test.X = Z(nTrain+1:end, 2:end);
end


fprintf('[%s]: creating CP bag...\n', mfilename);
bag = make_bag(train.X, train.y);

pVals = zeros(size(test.X,1), 10);
for ii = 1:size(test.X,1)
    if mod(ii, 100) == 1
        fprintf('[%s]: computing p-values for example %d (of %d)\n', mfilename, ii, size(test.X,1));
    end
    
    pVals(ii,:) = knn_cp(bag, test.X(ii,:), 'k', 3, 'smoothed', 1, 'classConditional', 1);
end


% some diagnostics
[pMax, yHat] = max(pVals, [], 2);
yHat = yHat-1;  % map [1,10] -> [0,9]
acc = sum(yHat == test.y) / length(test.y);
fprintf('[%s]: "accuracy" on test set: %0.2f%%\n', mfilename, 100*acc);

for ii = 1:10 , yi = ii - 1;
    figure('Position', [100 100 600 300]);
    subplot(1,2,1);
    hist(pVals(test.y==yi, ii));
    title(sprintf('objects of class %d', yi));
    
    subplot(1,2,2);
    hist(pVals(test.y~=yi, ii));
    title(sprintf('objects of class other than %d', yi));
    suptitle(sprintf('p-values for hypothesis y=%d', yi));
end


