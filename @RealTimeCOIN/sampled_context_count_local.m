function p = sampled_context_count_local(obj)
%SAMPLED_CONTEXT_COUNT_LOCAL Fast local-label sampled-context occupancy.
%
%   Returns a row vector in the modal particles' local label frame. This is
%   intended for live plots/logging and deliberately avoids global relabelling.

    p = obj.localContextProbabilityVector("count");
end
