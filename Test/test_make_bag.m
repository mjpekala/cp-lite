% This also serves as a test of our euclidean_distance() function.

% oct 2015, mjp

addpath('..');

load fisheriris

y = strcmp(species, 'versicolor') + 2*strcmp(species, 'virginica');

X1 = meas(:,1);
X = meas;

bag1 = make_bag(X1,y);
bag = make_bag(X,y);

% Make sure the bag distances are consistent with matlab's pdist()
% (it used to be the case that our euclidean distance function was
%  much faster; should check to see if this is still the case with
%  newer version of Matlab).
D1 = squareform(pdist(X1, 'euclidean'));
D = squareform(pdist(X, 'euclidean'));

assert(norm(D1 - bag1.D, 'fro') < 1e-5);
assert(norm(D - bag.D, 'fro') < 1e-5);

% make sure vector version of euclidean_distance is reasonable
for ii = 1:size(X,1)
    di = euclidean_distance(X, X(ii,:));
    assert(norm(di - D(:,ii)', 2) <= 1e-5);
end



fprintf('[%s] All tests passed.\n', mfilename);
