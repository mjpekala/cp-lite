function [D] = euclidean_distance(X, Y, varargin)
% EUCLIDEAN_DISTANCE  Finds all pairs distances between objects in
%                     X and Y.
%
%   PARAMETERS:
%    X : an nxd matrix of n objects each with d dimensions (objects-as-rows)
%    Y : an mxd matrix of m objects each with d dimensions (objects-as-rows).
%        Note the matrix Y can be equal to the matrix X.
%
%   RETURNS:
%    D : all-pairs distances
%
%   Costs of this brute-force approach:
%      O(d*n*m)  to compute all pairs distances (n*m inner products
%                of dimension d)

% oct 2015, mjp

ip = inputParser;
ip.addRequired('X', @ismatrix);
ip.addRequired('Y', @ismatrix);
ip.parse(X, Y);

assert(size(X,2) == size(Y,2));  % X and Y should have same dimension
[n,d] = size(X);
m     = size(Y,1);

X = X';  % more convenient to use objects-as-columns internally
Y = Y';


% Use the following equivalence (for efficiency):
%  ||x-y||^2  = ||x||^2 + ||y||^2 - 2xy
x2 = sum(X.*X, 1);
y2 = sum(Y.*Y, 1);
XtY = X'*Y;
D = bsxfun(@plus, x2', y2) - 2*XtY;
D = max(D,0);
D = sqrt(D);

D = D';
