function [mapped, accuracy, mapping] = validation_best_label_map(trueLabels, inferredLabels)
%VALIDATION_BEST_LABEL_MAP Match inferred context labels to true labels.
%
%   Context labels in a mixture model are exchangeable: inferred label "2"
%   can represent the synthetic context called "1" without changing the
%   statistical model.  This helper finds the relabelling of inferred
%   labels that maximizes agreement with the known synthetic labels.
%
%   The optimal relabelling is a maximum-weight bipartite matching between
%   inferred labels and target labels 1:K, where the weight of mapping an
%   inferred label to a target is the number of trials on which that pairing
%   agrees with the true labels (the confusion matrix).  This is solved as a
%   linear assignment problem, which is exact for any number of contexts K.
%
%   The private class solver @RealTimeCOIN/private/linearAssignment is not
%   reachable from the validation folder, so the built-in MATLAB solver
%   MATCHPAIRS (R2019a+) is used.  When MATCHPAIRS is unavailable the code
%   falls back to an exhaustive permutation search for K <= 8; for K > 8
%   without MATCHPAIRS it warns and returns the identity mapping (the only
%   case in which the result may be suboptimal).

trueLabels = trueLabels(:)';
inferredLabels = inferredLabels(:)';
valid = isfinite(trueLabels) & isfinite(inferredLabels);
trueValid = trueLabels(valid);
inferredValid = inferredLabels(valid);

uTrue = unique(trueValid);
uInf = unique(inferredValid);
K = max(numel(uTrue), numel(uInf));
if K == 0
    mapped = inferredLabels;
    accuracy = NaN;
    mapping = containers.Map('KeyType', 'double', 'ValueType', 'double');
    return;
end

% Confusion counts: agreement(i, g) is the number of trials where inferred
% label uInf(i) coincides with true label g, for target labels g = 1:K.
nInf = numel(uInf);
agreement = zeros(nInf, K);
for i = 1:nInf
    inferredMask = inferredValid == uInf(i);
    for g = 1:K
        agreement(i, g) = sum(inferredMask & trueValid == g);
    end
end

if exist('matchpairs', 'file') == 2 || exist('matchpairs', 'builtin') == 5
    bestMapVector = hungarian_label_map(agreement, K);
elseif K <= 8
    bestMapVector = exhaustive_label_map(agreement, K);
else
    warning('validation_best_label_map:NoAssignmentSolver', ...
        ['matchpairs is unavailable and K = %d > 8; returning the identity ', ...
         'label map, which may understate the relabelled accuracy.'], K);
    bestMapVector = 1:K;
end

mapped = inferredLabels;
mapping = containers.Map('KeyType', 'double', 'ValueType', 'double');
matchedCount = 0;
for i = 1:nInf
    target = bestMapVector(i);
    mapping(uInf(i)) = target;
    mapped(inferredLabels == uInf(i)) = target;
    matchedCount = matchedCount + agreement(i, target);
end
accuracy = matchedCount / numel(trueValid);
end

function mapVector = hungarian_label_map(agreement, K)
%HUNGARIAN_LABEL_MAP Optimal inferred->target assignment via matchpairs.
%
%   Maximises total agreement by minimising the negated confusion matrix.
%   Returns MAPVECTOR(i) = target label in 1:K for inferred label i; the
%   result is guaranteed to be an injective (permutation-valid) mapping.
nInf = size(agreement, 1);
% costUnmatched must exceed the benefit of any single match so that every
% inferred row is matched (there are K >= nInf target columns available).
costUnmatched = 1;
M = matchpairs(-agreement, costUnmatched);   % rows = inferred, cols = target
mapVector = zeros(1, nInf);
if ~isempty(M)
    mapVector(M(:, 1)) = M(:, 2);
end
% Assign any inferred label matchpairs left unmatched to a leftover target
% so the mapping is always a valid injection into 1:K.
unmatched = find(mapVector == 0);
if ~isempty(unmatched)
    leftover = setdiff(1:K, mapVector(mapVector > 0), 'stable');
    mapVector(unmatched) = leftover(1:numel(unmatched));
end
end

function bestMapVector = exhaustive_label_map(agreement, K)
%EXHAUSTIVE_LABEL_MAP Brute-force optimal assignment for small K.
nInf = size(agreement, 1);
candidates = perms(1:K);
bestScore = -Inf;
bestMapVector = 1:K;
for row = 1:size(candidates, 1)
    candidate = candidates(row, :);
    score = 0;
    for i = 1:nInf
        score = score + agreement(i, candidate(i));
    end
    if score > bestScore
        bestScore = score;
        bestMapVector = candidate(1:nInf);
    end
end
end
