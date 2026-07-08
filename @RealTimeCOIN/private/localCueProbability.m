function q = localCueProbability(obj, local, p)
%LOCALCUEPROBABILITY Cue emission distribution for one context/particle.
%   q = localCueProbability(obj, local, p) returns the normalised cue-emission
%   probability row for local context slot `local` of particle `p`, read from
%   obj.D.local_cue_matrix and renormalised to sum to 1.
%
%   Inputs:
%     local  local context slot index.
%     p      particle index.
%
%   Output:
%     q  1-by-Q row of cue probabilities summing to 1.
    q = squeeze(obj.D.local_cue_matrix(local, :, p));
    q = obj.normalizeProbability(q(:)');
end
