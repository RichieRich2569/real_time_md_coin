function Lg = globalCueTensor(obj, L, alignment)
%GLOBALCUETENSOR Accumulate per-particle cue emissions into global slots.
%   Lg = globalCueTensor(obj, L, alignment) reindexes a per-particle cue tensor
%   L (context-by-cue-by-particle) into the shared global-context frame, summing
%   the emission mass of local slots that map to the same global context.
%
%   This is the context-by-cue analogue of globalContextWeights: the cue axis
%   (size Q) is carried through unchanged while the context axis is realigned.
%   Only the modal particles are summarised (one output page each).
%
%   Inputs:
%     L          (max_contexts+1)-by-Q-by-P cue tensor indexed by local context.
%     alignment  alignment struct (see ensureContextAlignment). Optional;
%                recomputed lazily when omitted.
%
%   Output:
%     Lg   (max_contexts+1)-by-Q-by-nModal cue tensor indexed by global context.
    if nargin < 3
        alignment = obj.ensureContextAlignment();
    end
    Lg = obj.scatterToGlobal(L, alignment, "cue");
end
