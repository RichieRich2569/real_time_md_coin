function Tg = globalTransitionTensor(obj, T, alignment)
%GLOBALTRANSITIONTENSOR Accumulate per-particle transition mass into global slots.
%   Tg = globalTransitionTensor(obj, T, alignment) reindexes a per-particle
%   context-transition tensor T (from-by-to-by-particle) into the shared
%   global-context frame, realigning BOTH the from and to context axes and
%   summing the mass of local (from,to) pairs that map to the same global pair.
%
%   This is the two-context-axis analogue of globalContextWeights. Only the
%   modal particles are summarised (one output page each).
%
%   Inputs:
%     T          (max_contexts+1)-by-(max_contexts+1)-by-P transition tensor,
%                indexed by local from/to context slots.
%     alignment  alignment struct (see ensureContextAlignment). Optional;
%                recomputed lazily when omitted.
%
%   Output:
%     Tg   (max_contexts+1)-by-(max_contexts+1)-by-nModal transition tensor,
%          indexed by global from/to context slots.
    if nargin < 3
        alignment = obj.ensureContextAlignment();
    end
    Tg = obj.scatterToGlobal(T, alignment, "transition");
end
