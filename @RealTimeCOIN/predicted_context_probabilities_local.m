function p = predicted_context_probabilities_local(obj)
%PREDICTED_CONTEXT_PROBABILITIES_LOCAL Fast local-label context prediction.
%
%   Returns a row vector in the modal particles' local label frame. This is
%   intended for live plots/logging and deliberately avoids global relabelling.

    p = obj.localContextProbabilityVector("predicted");
end
