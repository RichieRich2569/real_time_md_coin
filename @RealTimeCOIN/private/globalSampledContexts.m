function c = globalSampledContexts(obj, alignment)
%GLOBALSAMPLEDCONTEXTS Sampled context of each modal particle in global labels.
%   c = globalSampledContexts(obj, alignment) returns the currently sampled
%   context (obj.D.context) of every modal particle, translated from its local
%   per-particle label into the shared global-context label via
%   alignment.assignment.
%
%   Inputs:
%     alignment  alignment struct (see ensureContextAlignment). Optional;
%                recomputed lazily when omitted.
%
%   Output:
%     c   1-by-nModal row of global context labels. Entries are NaN where the
%         sampled local context has no global assignment (unmatched slot).
    if nargin < 2
        alignment = obj.ensureContextAlignment();
    end
    modalIdx = alignment.modal_particle_indices; % particles at the modal cardinality
    c = NaN(1, numel(modalIdx));
    for idx = 1:numel(modalIdx)
        p = modalIdx(idx);
        local = obj.D.context(p);               % this particle's sampled local slot
        if local <= size(alignment.assignment, 1)
            target = alignment.assignment(local, p); % corresponding global slot
            if target > 0
                c(idx) = target;
            end
        end
    end
end
