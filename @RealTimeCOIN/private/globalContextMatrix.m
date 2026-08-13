function Xg = globalContextMatrix(obj, X, alignment)
%GLOBALCONTEXTMATRIX Scatter a per-particle local matrix into the global frame.
%   Xg = globalContextMatrix(obj, X, alignment) reindexes a per-particle,
%   locally-labelled matrix X into the shared global-context frame so that the
%   same physical context occupies the same row across particles.
%
%   Context labels are per-particle and arbitrary; alignment.assignment maps
%   each local slot to its global slot for a given particle (0 = unmatched).
%   Only the modal particles (those with the modal context cardinality) are
%   summarised, so the output has one column per modal particle.
%
%   Inputs:
%     X          (max_contexts+1)-by-P matrix indexed by local context slot.
%     alignment  alignment struct (see ensureContextAlignment). Optional;
%                recomputed lazily when omitted.
%
%   Output:
%     Xg   (max_contexts+1)-by-nModal matrix indexed by global context slot.
%          Unmatched local slots are dropped. Assignment overwrites (last local
%          slot mapped to a global slot wins); see globalContextWeights for the
%          accumulating variant used for probability mass.
    if nargin < 3
        alignment = obj.ensureContextAlignment();
    end
    Xg = obj.scatterToGlobal(X, alignment, "overwrite");
end
