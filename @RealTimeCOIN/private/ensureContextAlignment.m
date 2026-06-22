function alignment = ensureContextAlignment(obj)
    % Context-alignment-based diagnostics/queries are not yet generalised to
    % the multi-dimensional state (Phase 2). Fail with a clear, actionable
    % message rather than an obscure missing-field error from the scalar
    % alignment code. Multi-dimensional summaries are available via
    % state_moments() and predictive_feedback_moments().
    if obj.state_dim > 1
        error('RealTimeCOIN:MultiDimQueryUnsupported', ...
            ['Context-alignment queries (diagnostics, sampled_context_count, ', ...
             'context_alignment, predicted/responsibility context probabilities) ', ...
             'are not yet supported for state_dim > 1. Use state_moments() and ', ...
             'predictive_feedback_moments() for multi-dimensional summaries.']);
    end
    if ~isempty(obj.alignment_cache) && ...
            isfield(obj.alignment_cache, 'cache_state_version') && ...
            obj.alignment_cache.cache_state_version == obj.state_version
        alignment = obj.alignment_cache;
        return;
    end
    alignment = obj.computeContextAlignment();
    obj.alignment_cache = alignment;
end
