function p = responsibilities_vector(obj)
%RESPONSIBILITIES_VECTOR Cross-run averaged posterior context probabilities.
%   p = responsibilities_vector(obj) returns the 1-by-(max_contexts+1) posterior
%   context probability row, averaged across runs in the common reference frame
%   (docs/SPEC_ensemble.md Part 10). Real contexts occupy slots 1..Kref, the
%   novel-context probability sits in slot Kref+1, and the row sums to 1
%   (zero-fill averaging conserves probability). Read-only.
%
%   See also PREDICTED_CONTEXT_PROBABILITIES_VECTOR, SAMPLED_CONTEXT_COUNT.
    arguments
        obj (1, 1) RealTimeCOINEnsemble
    end
    p = ensembleContextVector(obj, @(m) m.responsibilities_vector());
end
