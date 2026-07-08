function alignment = ensureContextAlignment(obj)
%ENSURECONTEXTALIGNMENT Return the cached global alignment, recomputing if stale.
%   alignment = ensureContextAlignment(obj) is the caching front-end for
%   computeContextAlignment.m. It returns obj.alignment_cache when that cache was
%   built for the current obj.state_version; otherwise it recomputes and stores
%   the fresh alignment. state_version is bumped by invalidateContextAlignment
%   after every observe_y, so the cache is reused within a trial and rebuilt once
%   per new observation. Called by all context-facing query methods.

    % Cache hit: same state_version as when the alignment was computed.
    if ~isempty(obj.alignment_cache) && ...
            isfield(obj.alignment_cache, 'cache_state_version') && ...
            obj.alignment_cache.cache_state_version == obj.state_version
        alignment = obj.alignment_cache;
        return;
    end
    % Cache miss / stale: recompute and store for reuse until the next mutation.
    alignment = obj.computeContextAlignment();
    obj.alignment_cache = alignment;
end
