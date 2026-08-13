function d = novel_state_feedback_probability(obj, values)
%NOVEL_STATE_FEEDBACK_PROBABILITY Run-averaged novel-context feedback density.
%   d = novel_state_feedback_probability(obj, values) returns the equal-weight,
%   NaN-aware average across the R members of
%   RealTimeCOIN/novel_state_feedback_probability(values) at the current trial. A
%   member whose novel-context density is all zeros (context budget exhausted)
%   contributes zeros to the average. values and the returned row d follow the
%   same shape rules as the single-model method.
%
%   See also RealTimeCOIN/novel_state_feedback_probability, NOVEL_STATE_PROBABILITY.
    arguments
        obj (1, 1) RealTimeCOINEnsemble
        values (:, :) double {mustBeFinite, mustBeReal}
    end
    vals = cell(1, obj.runs);
    for k = 1:obj.runs
        vals{k} = obj.members{k}.novel_state_feedback_probability(values);
    end
    d = averageAcrossRuns(vals);
end
