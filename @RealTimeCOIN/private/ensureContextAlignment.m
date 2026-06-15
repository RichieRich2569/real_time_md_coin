function alignment = ensureContextAlignment(obj)
    if ~isempty(obj.alignment_cache) && ...
            isfield(obj.alignment_cache, 'cache_state_version') && ...
            obj.alignment_cache.cache_state_version == obj.state_version
        alignment = obj.alignment_cache;
        return;
    end
    alignment = obj.computeContextAlignment();
    obj.alignment_cache = alignment;
end
