function invalidateContextAlignment(obj)
%INVALIDATECONTEXTALIGNMENT Mark the cached global alignment stale.
%   invalidateContextAlignment(obj) bumps the monotone obj.state_version counter
%   and clears obj.alignment_cache. Called at the end of every observe_y so that
%   the next context-facing query recomputes the alignment against the updated
%   particle state (see ensureContextAlignment.m). The warm-start seed in
%   obj.alignment_seed is deliberately left intact so the recompute can reuse it.

    % Bump the version so any existing cache is detected as stale, then drop it.
    obj.state_version = obj.state_version + 1;
    obj.alignment_cache = [];
end
