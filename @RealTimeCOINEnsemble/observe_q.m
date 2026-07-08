function observe_q(obj, q)
%OBSERVE_Q Stage the cue q for the upcoming trial on every member.
%   observe_q(obj, q) forwards the identical cue q to each member RealTimeCOIN
%   (member.observe_q(q)). It draws no randomness and does not advance the
%   trial. q is a scalar cue id, or [] / NaN for a cue-free trial.
%
%   See also OBSERVE_Y, RealTimeCOIN/observe_q.
    arguments
        obj (1, 1) RealTimeCOINEnsemble
        q double {mustBeScalarOrEmpty} = []
    end
    for k = 1:obj.runs
        obj.members{k}.observe_q(q);
    end
end
