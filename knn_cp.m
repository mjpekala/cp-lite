function pVals = knn_cp(bag, x, varargin)
% KNN_CP  Computes conformal prediction p-values using a
%         kNN-ratio nonconformity score.
%
%   REQUIRED PARAMETERS:
%     bag :  A structure created by calling make_bag()
%
%     x  : a dx1 vector corresponding to an out-of-bag example
%          to compute p-values for
%
%   OPTIONAL PARAMETERS:
%      'k'       :=  (positive integer) The number of nearest neighbors to 
%                    use in the kNN ratio
%      'verbose' :=  If true/1, prints debugging information
%      'smoothed' := If true/1, uses a smoothed CP (recommended)
%      'classConditional' := If true/1 uses a 'Mondrian' CP variant
%                    that ensures per-class validity

% oct 2015, mjp

%~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
% parse parameters
%~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
ip = inputParser;
ip.addRequired('bag');
ip.addRequired('x', @isvector);
ip.addParamValue('k', 1, @(x) isscalar(x) && x > 0);
ip.addParamValue('verbose', 0, @isscalar);
ip.addParamValue('smoothed', 1, @isscalar);
ip.addParamValue('classConditional', 1, @isscalar);
ip.addParamValue('online', 1, @isscalar);
ip.parse(bag, x, varargin{:});

k = ip.Results.k;
verbose = ip.Results.verbose;
doSmoothing = ip.Results.smoothed;
classConditional = ip.Results.classConditional;


% initialize variables and check parameters
[n,d] = size(bag.X);
assert(all(size(bag.D) == n));
assert(length(x) == d);

yAll = sort(unique(bag.y));
pVals = zeros(length(yAll),1);
distx = euclidean_distance(bag.X, x(:)');

%~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

% calculate a nonconformity score for all items in the bag
netDistSame = sum(bag.D_same(1:k,:),1);     % note restriction to k NN here
netDistDiff = sum(bag.D_diff(1:k,:),1);     %  "   "
alpha = netDistSame ./ netDistDiff;

% address some corner cases in the calculation above
alpha(netDistSame == 0 & netDistDiff == 0) = 0;  % avoid NaNs
alpha(netDistSame > 0 & netDistDiff == 0) = Inf;
assert(~any(isnan(alpha)));


% need to test each possible hypothesis (i.e. class label in yAll)
for kk=1:length(yAll) 
    yk = yAll(kk);                % null hypothesis is x is class y_k

    % compute nonconformity score / test statistic for this hypothesis
    dSameK = distx(bag.y == yk);
    dSameK = sort(dSameK, 'ascend');
    
    dDiffK = distx(bag.y ~= yk);
    dDiffK = sort(dDiffK, 'ascend');
    
    alphaK = sum(dSameK(1:k)) / sum(dDiffK(1:k));

    % if class conditional, only compare nonconformity scores with
    % objects of the same class.
    if classConditional
        bits = (bag.y == yk);
    else
        bits = logical(ones(size(bag.y)));
    end
       
    % Note: the comparison of alpha(ii) with itself should be included in
    % this calculation (hence the +1).
    if doSmoothing
        nEq = sum(double(alpha(bits) == alphaK) .* bag.tau(bits)') + rand;
    else
        nEq = sum(alpha(bits) == alphaK) + 1;
    end
    nGt = sum(alpha(bits) > alphaK);
    pVals(kk) = (nEq+nGt) / (sum(bits)+1);
    
end % loop over labels

