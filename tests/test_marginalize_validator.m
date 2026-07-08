function test_marginalize_validator
%TEST_MARGINALIZE_VALIDATOR Tests for the mustMarginalize arguments validator.
%
%   Exercises @RealTimeCOIN/private/mustMarginalize, which enforces that a
%   (max_contexts+1)-by-K per-context mixing-weight matrix has columns summing
%   to one (within a 1e-6 tolerance). The private folder is placed on the path
%   for the duration of the test so the validator can be called directly.

rng(11);

% Make the private class helpers callable directly from this test.
thisDir = fileparts(mfilename('fullpath'));
privateDir = fullfile(thisDir, '..', '@RealTimeCOIN', 'private');
addpath(privateDir);
restorePath = onCleanup(@() rmpath(privateDir));

% Small model: max_contexts + 1 == 4 context slots (rows).
coin = RealTimeCOIN('num_particles', 5, 'max_contexts', 3);
numRows = coin.max_contexts + 1;
assert(numRows == 4, 'Expected 4 context rows for max_contexts == 3');

% (a) A normalized (max_contexts+1)-by-K weight matrix passes.
good = rand(numRows, 6);
good = good ./ sum(good, 1);
mustMarginalize(coin, good);   % must not error

% (a') Deviations within the 1e-6 tolerance are absorbed (round-off is safe).
nearlyGood = good;
nearlyGood(1, 1) = nearlyGood(1, 1) + 5e-7;   % column sum off by 5e-7 < 1e-6
mustMarginalize(coin, nearlyGood);            % must not error

% (b) A clearly non-normalized (max_contexts+1)-by-K matrix throws.
bad = ones(numRows, 2);   % each column sums to 4
testutil.mustError('non-normalized weights', ...
    @() mustMarginalize(coin, bad), 'RealTimeCOIN:MustMarginalize');

% (c) A matrix whose row count ~= max_contexts+1 is not checked and passes,
%     even though its columns do not sum to one.
offShape = ones(numRows - 1, 2);   % 3 rows, column sums == 3
mustMarginalize(coin, offShape);   % must not error

fprintf('test_marginalize_validator passed.\n');
end
