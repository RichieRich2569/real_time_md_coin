function out = averageAcrossRuns(vals)
%AVERAGEACROSSRUNS NaN-aware equal-weight mean of per-run arrays.
%   out = averageAcrossRuns(vals) takes a 1-by-R cell array whose elements are
%   identically-sized numeric arrays (one per run) and returns their elementwise
%   mean. The average is NaN-aware in the sense of COIN.m's
%   weighted_sum_along_dimension: each output entry is the mean over the runs
%   for which that entry is finite; an entry that is non-finite for EVERY run is
%   returned as NaN.
    R = numel(vals);
    if R == 0
        out = [];
        return;
    end
    runDim = ndims(vals{1}) + 1;
    stacked = cat(runDim, vals{:});
    finiteMask = isfinite(stacked);
    n = sum(finiteMask, runDim);
    s = sum(stacked .* finiteMask, runDim);   % treat non-finite as absent
    out = s ./ n;
    out(n == 0) = NaN;
end
