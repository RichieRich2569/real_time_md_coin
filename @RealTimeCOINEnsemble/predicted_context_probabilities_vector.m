function p = predicted_context_probabilities_vector(obj)
%PREDICTED_CONTEXT_PROBABILITIES_VECTOR Cross-run averaged prior context probs.
%
%   *** PHASE 2 STUB: returns a NaN placeholder of the correct shape. ***
%   The real implementation aligns members onto a common reference frame
%   (docs/SPEC_ensemble.md Part 10) and zero-fill-averages the per-run predicted
%   context probability vectors so the result sums to 1.
    arguments
        obj (1, 1) RealTimeCOINEnsemble
    end
    p = nan(1, obj.members{1}.max_contexts + 1);
end
