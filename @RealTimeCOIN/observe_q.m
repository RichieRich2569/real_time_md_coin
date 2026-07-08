function observe_q(obj, q)
%OBSERVE_Q Stage the sensory cue for the upcoming trial.
%   observe_q(obj, q) records the cue index/value q for the next trial into
%   obj.pending_q, without advancing the trial counter. The staged cue is
%   consumed by the following observe_y call, which runs the inference
%   pipeline. Call observe_q before each observe_y when the paradigm provides
%   an explicit contextual cue.
%
%   q is a scalar cue identifier (any numeric value; distinct values are
%   mapped to consecutive cue columns internally). Passing [] or NaN clears
%   any pending cue, so the next trial is treated as cue-free.
%
%   See also OBSERVE_Y.
    arguments
        obj (1, 1) RealTimeCOIN
        q double {mustBeScalarOrEmpty} = []
    end
    if isempty(q) || isnan(q)
        obj.pending_q = [];
        return;
    end
    obj.pending_q = q;
end
