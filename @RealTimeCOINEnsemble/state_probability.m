function d = state_probability(obj, values)
%STATE_PROBABILITY Run-averaged posterior latent-state density on a grid.
%   d = state_probability(obj, values) returns the equal-weight, NaN-aware
%   average across the R members of RealTimeCOIN/state_probability(values) at the
%   current trial. Averaging the per-run densities gives the density of the
%   pooled 1/R-weighted mixture. values and the returned row d follow the same
%   shape rules as the single-model method (scalar model: length-K vector in,
%   1-by-K row out; multi-dimensional: N-by-K columns in, 1-by-K row out).
%
%   See also RealTimeCOIN/state_probability, STATE_FEEDBACK_PROBABILITY.
    arguments
        obj (1, 1) RealTimeCOINEnsemble
        values (:, :) double {mustBeFinite, mustBeReal}
    end
    vals = cell(1, obj.runs);
    for k = 1:obj.runs
        vals{k} = obj.members{k}.state_probability(values);
    end
    d = averageAcrossRuns(vals);
end
