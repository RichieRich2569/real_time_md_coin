function d = state_feedback_probability(obj, values)
%STATE_FEEDBACK_PROBABILITY Run-averaged predictive feedback density on a grid.
%   d = state_feedback_probability(obj, values) returns the equal-weight,
%   NaN-aware average across the R members of
%   RealTimeCOIN/state_feedback_probability(values) at the current trial (the
%   density of the pooled 1/R-weighted mixture). values and the returned row d
%   follow the same shape rules as the single-model method.
%
%   See also RealTimeCOIN/state_feedback_probability, STATE_PROBABILITY.
    arguments
        obj (1, 1) RealTimeCOINEnsemble
        values (:, :) double {mustBeFinite, mustBeReal}
    end
    vals = cell(1, obj.runs);
    for k = 1:obj.runs
        vals{k} = obj.members{k}.state_feedback_probability(values);
    end
    d = averageAcrossRuns(vals);
end
