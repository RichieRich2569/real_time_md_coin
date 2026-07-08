function m = state_given_context_probability(obj, values)
%STATE_GIVEN_CONTEXT_PROBABILITY Cross-run averaged per-context state density.
%   m = state_given_context_probability(obj, values) returns a containers.Map
%   keyed by reference context label 1..Kref (docs/SPEC_ensemble.md Part 10). Each
%   value is the per-context posterior state density averaged across runs with the
%   NaN-omit rule: the density of reference context j is the mean over only the
%   runs whose matched context has a density for j; contexts held by no run are
%   absent from the map. values follows the single-model shape rules (scalar
%   model: length-K vector; MD: N-by-K columns), each density a 1-by-K row.
%
%   See also RealTimeCOIN/state_given_context_probability,
%   STATE_FEEDBACK_GIVEN_CONTEXT_PROBABILITY.
    arguments
        obj (1, 1) RealTimeCOINEnsemble
        values (:, :) double {mustBeFinite, mustBeReal}
    end
    m = ensembleContextDensity(obj, @(mem) mem.state_given_context_probability(values));
end
