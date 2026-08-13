function p = predicted_context_probabilities(obj)
%PREDICTED_CONTEXT_PROBABILITIES (deprecated) Alias for the *_VECTOR form.
%
%   Kept for backward compatibility. Forwards to
%   predicted_context_probabilities_vector and emits a one-time deprecation
%   warning per session. Prefer the explicit *_vector (row vector) or *_map
%   (containers.Map) name to disambiguate the return type at the call site.
%
%   See also PREDICTED_CONTEXT_PROBABILITIES_VECTOR,
%   PREDICTED_CONTEXT_PROBABILITIES_MAP.
    arguments
        obj (1, 1) RealTimeCOIN
    end
    persistent warned
    if isempty(warned)
        warning("RealTimeCOIN:DeprecatedMethod", ...
            "predicted_context_probabilities is deprecated; use " + ...
            "predicted_context_probabilities_vector (row vector) or " + ...
            "predicted_context_probabilities_map (containers.Map).");
        warned = true;
    end
    p = obj.predicted_context_probabilities_vector();
end
