function m = state_feedback_given_context_probability(obj, values)
%STATE_FEEDBACK_GIVEN_CONTEXT_PROBABILITY Cross-run averaged per-context feedback density.
%
%   *** PHASE 2 STUB: returns an empty containers.Map. ***
%   The real implementation aligns members onto a common reference frame
%   (docs/SPEC_ensemble.md Part 10) and NaN-omit-averages each reference
%   context's per-run predictive feedback density, returning a containers.Map
%   keyed by reference context label 1..Kref.
    arguments
        obj (1, 1) RealTimeCOINEnsemble
        values (:, :) double {mustBeFinite, mustBeReal}
    end
    m = containers.Map('KeyType', 'double', 'ValueType', 'any');
end
