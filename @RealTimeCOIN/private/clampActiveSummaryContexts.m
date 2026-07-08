function active = clampActiveSummaryContexts(obj, alignment)
%CLAMPACTIVESUMMARYCONTEXTS Active summary contexts clamped to the aligned set.
%
%   active = clampActiveSummaryContexts(obj, alignment) returns the list of
%   active summary contexts (activeSummaryContexts) restricted to those with a
%   valid global label in the supplied alignment (index <= alignment.K). If the
%   restriction empties the list but the alignment does hold contexts
%   (alignment.K > 0), it falls back to context 1, matching COIN's convention of
%   always summarising at least the first global context.
%
%   Shared by the per-context parameter-density query methods
%   (retention_given_context_probability, drift_given_context_probability,
%   bias_given_context_probability) so the clamp/fallback boilerplate lives in
%   one place. Pure read of obj and alignment; does not mutate model state.
    active = activeSummaryContexts(obj);
    active = active(active <= alignment.K);
    if isempty(active) && alignment.K > 0
        active = 1;
    end
end
