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
    Wg = obj.scatterToGlobal(W, alignment, "add");
end
