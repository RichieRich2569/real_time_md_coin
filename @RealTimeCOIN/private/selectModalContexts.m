function [Km, modalMask, modalIdx, weights] = selectModalContexts(obj)
%SELECTMODALCONTEXTS Pick the modal context cardinality and its particle subset.
%   [Km, modalMask, modalIdx, weights] = selectModalContexts(obj) chooses the
%   number of contexts Km used for the global alignment, and the particles that
%   participate in it. Aligning only particles that agree on the number of
%   contexts (the modal cardinality) keeps the assignment problem square and the
%   global labels well defined.
%
%   Outputs:
%     Km        modal number of active contexts across particles.
%     modalMask 1-by-P logical, true for particles with exactly Km contexts.
%     modalIdx  indices of the modal particles (find(modalMask)).
%     weights   1-by-numel(modalIdx) uniform weights summing to 1.
%
%   Fallback: if no particle carries the modal cardinality (should not normally
%   happen), every particle is treated as modal and Km falls back to the largest
%   observed cardinality, clipped to max_contexts.

    P = obj.num_particles;
    cards = obj.D.C(:)';                 % per-particle active-context counts
    Km = obj.modalCardinality(cards);    % most common cardinality
    modalMask = cards == Km;
    modalIdx = find(modalMask);
    if isempty(modalIdx)
        % Degenerate case: fall back to using all particles.
        modalIdx = 1:P;
        modalMask = true(1, P);
        Km = min(max(cards), obj.max_contexts);
    end
    % Equal weight per modal particle (weights sum to 1).
    weights = ones(1, numel(modalIdx)) ./ max(numel(modalIdx), 1);
end
