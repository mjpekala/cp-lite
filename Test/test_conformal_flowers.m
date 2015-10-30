% Tests the nonconformity score calculation using an example from [1].
%
%  References:
%    [1] Shafer & Vovk "A Tutorial on Conformal Prediction," 2008

% oct 2015, mjp

addpath(genpath('..'));


%~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
% example from Shafer & Vovk "A Tutorial on Conformal
% Prediction" 2008

Z  = [5.0   1 ;
      4.4   1 ;
      4.9   1 ;
      4.4   1 ;
      5.1   1 ;
      5.9  -1 ;
      5.0   1 ;
      6.4  -1 ;
      6.7  -1 ;
      6.2  -1 ;
      5.1   1 ;
      4.6   1 ;
      5.0   1 ;
      5.4   1 ;
      5.0  -1 ;
      6.7  -1 ;
      5.8  -1 ;
      5.5   1 ;
      5.8  -1 ;
      5.4   1 ;
      5.1   1 ;
      5.7  -1 ;
      4.6   1 ;
      4.6   1];
X = Z(:,1);  y = Z(:,2);

xNew = 6.8;    % features of the unknown object


% expected nonconformity scores, assuming k=1 
expectedAlphaS = [0 0 1 0 0 ...
                  0.25 0 0.5 0 1/3 ...
                  0 0 0 0 Inf ...
                  0 0 0.5 0 0 ...
                  0 .5 0 0 13]';

expectedAlphaV = [0 0 1 0 0 ...
                  0.25 0 0.22 0 0.29 ...
                  0 0 0 0 Inf ...
                  0 0 0.5 0 0 ...
                  0 .5 0 0 0.077]';

%~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

bag = make_bag(X, y);
pVals = knn_cp(bag, xNew, 'k', 1, 'smoothed', 0, 'classConditional', 0);

% make sure calculation matches expected result.
% * assumes no smoothing *
assert(abs(pVals(1) - 0.32) < 1e-5);
assert(abs(pVals(2) - 0.08) < 1e-5);

fprintf('[%s]: All tests passed!!\n', mfilename);
