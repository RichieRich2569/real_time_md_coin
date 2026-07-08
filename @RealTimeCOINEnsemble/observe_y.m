function observe_y(obj, y)
%OBSERVE_Y Feed feedback y to every member, advancing the ensemble one trial.
%   observe_y(obj, y) forwards the identical feedback y to each member
%   RealTimeCOIN, running each member's update under that member's dedicated RNG
%   substream so the members stay independent and reproducible. The caller's
%   global RNG stream is saved on entry and restored on exit (even on error), so
%   the ensemble has no side effect on the caller's stream.
%
%   y is a scalar (scalar model) or an N-by-1 column (multi-dimensional model);
%   [] or NaN marks a missing observation, and NaN entries of a column mark
%   partially-observed dimensions. All members receive the identical y.
%
%   See also OBSERVE_Q, RealTimeCOIN/observe_y.
    arguments
        obj (1, 1) RealTimeCOINEnsemble
        y (:, 1) double = []
    end
    prev = RandStream.getGlobalStream();
    restore = onCleanup(@() RandStream.setGlobalStream(prev));
    for k = 1:obj.runs
        RandStream.setGlobalStream(obj.streams{k});
        obj.members{k}.observe_y(y);
    end
    obj.trial_ = obj.trial_ + 1;
end
