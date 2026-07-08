function u = motor_output(obj)
%MOTOR_OUTPUT Run-averaged predicted state feedback.
%   u = motor_output(obj) returns the equal-weight, NaN-aware average across the
%   R members of RealTimeCOIN/motor_output evaluated at the current trial. It is
%   a scalar for the scalar model (state_dim == 1) and an N-by-1 vector for the
%   multi-dimensional model. Read-only: draws no randomness.
%
%   See also RealTimeCOIN/motor_output, STATE_MOMENTS.
    arguments
        obj (1, 1) RealTimeCOINEnsemble
    end
    vals = cell(1, obj.runs);
    for k = 1:obj.runs
        vals{k} = obj.members{k}.motor_output();
    end
    u = averageAcrossRuns(vals);
end
