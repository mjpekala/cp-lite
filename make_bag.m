function bag = make_bag(X, y)
% MAKE_BAG  Creates a CP data structure from examples Z = (X,y).
%
%   This data structure contains:
%       X : an n x d matrix of n objects each having d features
%       y : an n x 1 vector of class labels
%       D : an n x n matrix of all-pairs object distances
%     tau : an n x 1 vector of values in [0,1] used for smoothing
%  D_same : sorted distances to objects with the same label
%  D_diff : sorted distances to objects with a different label
%
%   This is not suitable for large scale problems (where some
%   smarter data structures and algorithms should be used for
%   computing distances and nearest neighbors).

% oct 2015, mjp

[n,d] = size(X);
assert(n == length(y));

bag.X = X;
bag.y = y(:);
bag.tau = rand(n,1);
bag.D = euclidean_distance(X, X);

%~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
% computing the k nearest neighbors is somewhat costly
% (in part because we are brute-force sorting rather
%  than using a selection algorithm).  To avoid doing
% this operation multiple times, do it once up front
% when creating the bag.  
%~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
IsSame = bsxfun(@eq, bag.y(:), bag.y(:)');

D_same = bag.D;
D_same(1:(n+1):(n*n)) = Inf;            % ignore self-distances
D_same(~IsSame) = Inf;                  % ignore distances to objects w/ different labels
bag.D_same = sort(D_same,1,'ascend');   % sorts columns indepdendently

D_diff = bag.D;  
D_diff(IsSame) = Inf;                   
bag.D_diff = sort(D_diff,1,'ascend'); 
