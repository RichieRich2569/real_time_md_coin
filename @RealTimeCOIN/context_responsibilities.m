function probs = context_responsibilities(obj)
%CONTEXT_RESPONSIBILITIES (deprecated) Alias for RESPONSIBILITIES_MAP.
%
%   Kept for backward compatibility. Forwards to responsibilities_map and emits a
%   one-time deprecation warning per session. The old name is confusing because
%   the near-identically named RESPONSIBILITIES returned a vector; prefer the
%   explicit responsibilities_map / responsibilities_vector.
%
%   See also RESPONSIBILITIES_MAP, RESPONSIBILITIES_VECTOR.
    arguments
        obj (1, 1) RealTimeCOIN
    end
    persistent warned
    if isempty(warned)
        warning("RealTimeCOIN:DeprecatedMethod", ...
            "context_responsibilities is deprecated; use responsibilities_map " + ...
            "(containers.Map) or responsibilities_vector (row vector).");
        warned = true;
    end
    probs = obj.responsibilities_map();
end
