function probs = context_predicted_probabilities(obj)
%CONTEXT_PREDICTED_PROBABILITIES (deprecated) Alias for the *_MAP form.
%
%   Kept for backward compatibility. Forwards to
%   predicted_context_probabilities_map and emits a one-time deprecation warning
%   per session. The old name is confusing because the near-identically named
%   PREDICTED_CONTEXT_PROBABILITIES returned a vector; prefer the explicit
%   predicted_context_probabilities_map / predicted_context_probabilities_vector.
%
%   See also PREDICTED_CONTEXT_PROBABILITIES_MAP,
%   PREDICTED_CONTEXT_PROBABILITIES_VECTOR.
    arguments
        obj (1, 1) RealTimeCOIN
    end
    persistent warned
    if isempty(warned)
        warning("RealTimeCOIN:DeprecatedMethod", ...
            "context_predicted_probabilities is deprecated; use " + ...
            "predicted_context_probabilities_map (containers.Map) or " + ...
            "predicted_context_probabilities_vector (row vector).");
        warned = true;
    end
    probs = obj.predicted_context_probabilities_map();
end
