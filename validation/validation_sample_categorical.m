function idx = validation_sample_categorical(p)
%VALIDATION_SAMPLE_CATEGORICAL Draw one index from unnormalised probabilities.

p = p(:)';
p(~isfinite(p) | p < 0) = 0;
if sum(p) <= 0
    p = ones(size(p)) ./ numel(p);
else
    p = p ./ sum(p);
end
idx = find(rand <= cumsum(p), 1, 'first');
end
