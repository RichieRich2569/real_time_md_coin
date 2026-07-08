function n = sampled_context_count(obj)
%SAMPLED_CONTEXT_COUNT Cross-run averaged sampled-context occupancy.
%   n = sampled_context_count(obj) returns the 1-by-(max_contexts+1) sampled
%   context occupancy row (fraction of particles per context), averaged across
%   runs in the common reference frame (docs/SPEC_ensemble.md Part 10). Real
%   contexts occupy slots 1..Kref, the novel context sits in slot Kref+1, and the
%   row sums to 1. Read-only.
%
%   See also RESPONSIBILITIES_VECTOR, PREDICTED_CONTEXT_PROBABILITIES_VECTOR.
    arguments
        obj (1, 1) RealTimeCOINEnsemble
    end
    n = ensembleContextVector(obj, @(m) m.sampled_context_count());
end
