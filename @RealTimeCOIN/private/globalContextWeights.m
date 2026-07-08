function Wg = globalContextWeights(obj, W, alignment)
%GLOBALCONTEXTWEIGHTS Accumulate per-particle context weights into global slots.
%   Wg = globalContextWeights(obj, W, alignment) reindexes a per-particle
%   context-weight matrix W (e.g. predicted probabilities or responsibilities)
%   into the shared global-context frame, summing the mass of any local slots
%   that map to the same global slot for a particle.
%
%   Unlike globalContextMatrix (which overwrites), weights are additive so that
%   probability mass is conserved when several local slots align to one global
%   context. Only the modal particles are summarised (one output column each).
%
%   Inputs:
%     W          (max_contexts+1)-by-P matrix of per-particle context weights.
%     alignment  alignment struct (see ensureContextAlignment). Optional;
%                recomputed lazily when omitted.
%
%   Output:
%     Wg   (max_contexts+1)-by-nModal matrix of globally aligned weights. When
%          the modal cardinality Km has reached max_contexts every particle has
%          instantiated all contexts, so the novel slot carries no mass and any
%          slots beyond Km are zeroed.
    if nargin < 3
        alignment = obj.ensureContextAlignment();
    end
    Cmax = obj.max_contexts + 1;                 % context slots incl. novel (+1)
    Km = alignment.K;                            % modal context cardinality
    modalIdx = alignment.modal_particle_indices; % particles at the modal cardinality
    Wg = zeros(Cmax, numel(modalIdx));
    for idx = 1:numel(modalIdx)
        p = modalIdx(idx);
        for local = 1:Cmax
            target = alignment.assignment(local, p); % global slot for this local slot
            if target > 0 && target <= Cmax
                Wg(target, idx) = Wg(target, idx) + W(local, p);
            end
        end
        if Km >= obj.max_contexts
            % All contexts instantiated: no novel slot, drop trailing slots.
            Wg(Km+1:end, idx) = 0;
        end
    end
end
