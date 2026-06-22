function [mapped, accuracy, mapping] = validation_best_label_map(trueLabels, inferredLabels)
%VALIDATION_BEST_LABEL_MAP Match inferred context labels to true labels.
%
%   Context labels in a mixture model are exchangeable: inferred label "2"
%   can represent the synthetic context called "1" without changing the
%   statistical model.  This helper finds the relabelling of inferred
%   labels that maximizes agreement with the known synthetic labels.

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

labels = 1:K;
if K <= 8
    candidates = perms(labels);
else
    candidates = labels;
end

bestAccuracy = -Inf;
bestMapVector = labels;
for row = 1:size(candidates, 1)
    candidate = candidates(row, :);
    trialMapped = inferredValid;
    for i = 1:numel(uInf)
        targetIdx = min(i, K);
        trialMapped(inferredValid == uInf(i)) = candidate(targetIdx);
    end
    score = mean(trialMapped == trueValid);
    if score > bestAccuracy
        bestAccuracy = score;
        bestMapVector = candidate;
    end
end

mapped = inferredLabels;
mapping = containers.Map('KeyType', 'double', 'ValueType', 'double');
for i = 1:numel(uInf)
    targetIdx = min(i, K);
    mapping(uInf(i)) = bestMapVector(targetIdx);
    mapped(inferredLabels == uInf(i)) = bestMapVector(targetIdx);
end
accuracy = bestAccuracy;
end
