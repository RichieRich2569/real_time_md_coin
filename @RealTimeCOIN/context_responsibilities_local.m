function p = context_responsibilities_local(obj)
%CONTEXT_RESPONSIBILITIES_LOCAL Fast local-label posterior context weights.
%
%   Returns a row vector in the modal particles' local label frame. This is
%   intended for live plots/logging and deliberately avoids global relabelling.

    p = obj.localContextProbabilityVector("responsibilities");
end
