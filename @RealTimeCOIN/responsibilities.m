function p = responsibilities(obj)
%RESPONSIBILITIES (deprecated) Alias for RESPONSIBILITIES_VECTOR.
%
%   Kept for backward compatibility. Forwards to responsibilities_vector and
%   emits a one-time deprecation warning per session. Use the explicit
%   responsibilities_vector (row vector) or responsibilities_map (containers.Map)
%   to make the return type clear at the call site.
%
%   See also RESPONSIBILITIES_VECTOR, RESPONSIBILITIES_MAP.
    arguments
        obj (1, 1) RealTimeCOIN
    end
    persistent warned
    if isempty(warned)
        warning("RealTimeCOIN:DeprecatedMethod", ...
            "responsibilities is deprecated; use responsibilities_vector " + ...
            "(row vector) or responsibilities_map (containers.Map).");
        warned = true;
    end
    p = obj.responsibilities_vector();
end
