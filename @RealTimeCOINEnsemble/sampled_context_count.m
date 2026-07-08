function n = sampled_context_count(obj)
%SAMPLED_CONTEXT_COUNT Cross-run averaged sampled-context occupancy.
%
%   *** PHASE 2 STUB: returns a NaN placeholder of the correct shape. ***
%   The real implementation aligns members onto a common reference frame
%   (docs/SPEC_ensemble.md Part 10) and zero-fill-averages the per-run sampled
%   context-count vectors so the result sums to 1.
    arguments
        obj (1, 1) RealTimeCOINEnsemble
    end
    n = nan(1, obj.members{1}.max_contexts + 1);
end
