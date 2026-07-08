function p = responsibilities_vector(obj)
%RESPONSIBILITIES_VECTOR Cross-run averaged posterior context probabilities.
%
%   *** PHASE 2 STUB: returns a NaN placeholder of the correct shape. ***
%   The real implementation aligns each member's contexts onto a common
%   reference frame (docs/SPEC_ensemble.md Part 10) and zero-fill-averages the
%   per-run responsibility vectors so the result sums to 1. Author tests against
%   the spec, not this stub.
    arguments
        obj (1, 1) RealTimeCOINEnsemble
    end
    p = nan(1, obj.members{1}.max_contexts + 1);
end
