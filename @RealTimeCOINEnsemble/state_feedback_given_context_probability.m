function m = state_feedback_given_context_probability(obj, values)
%STATE_FEEDBACK_GIVEN_CONTEXT_PROBABILITY Cross-run averaged per-context feedback density.
%   m = state_feedback_given_context_probability(obj, values) returns a
%   containers.Map keyed by reference context label 1..Kref
%   (docs/SPEC_ensemble.md Part 10). Each value is the per-context predictive
%   feedback density averaged across runs with the NaN-omit rule (mean over only
%   the runs whose matched context has a density for that reference label).
%   values follows the single-model shape rules (scalar model: length-K vector;
%   MD: N-by-K columns), each density a 1-by-K row.
%
%   See also RealTimeCOIN/state_feedback_given_context_probability,
%   STATE_GIVEN_CONTEXT_PROBABILITY.
    arguments
        obj (1, 1) RealTimeCOINEnsemble
        values (:, :) double {mustBeFinite, mustBeReal}
    end
    m = ensembleContextDensity(obj, @(mem) mem.state_feedback_given_context_probability(values));
end
