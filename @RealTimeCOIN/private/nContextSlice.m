function A = nContextSlice(obj, p)
%NCONTEXTSLICE Context-transition count matrix for one particle.
%   A = nContextSlice(obj, p) returns the per-particle context-transition count
%   matrix obj.D.n_context(:, :, p), i.e. the running counts of observed
%   from->to context transitions for particle `p`.
%
%   Inputs:
%     p  particle index.
%
%   Output:
%     A  (max_contexts+1)-by-(max_contexts+1) transition count matrix.
    A = obj.D.n_context(:, :, p);
end
