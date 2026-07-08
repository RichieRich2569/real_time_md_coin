function [L, ok] = choljitter(~, S)
%CHOLJITTER Lower-triangular Cholesky factor with PSD-failure fallback.
%
%   [L, ok] = choljitter(obj, S) returns a lower-triangular L with S = L*L'.
%   Particle-filter covariances occasionally drift very slightly non-positive
%   definite through accumulated round-off; rather than erroring, we
%   symmetrise S and add escalating diagonal jitter until the factorisation
%   succeeds. If even that fails (ok == false) we fall back to the diagonal
%   square root so downstream code degrades gracefully instead of throwing.

    % Delegate to the shared PD-repair utility; the "jitter" tactic is a verbatim
    % copy of the escalating-diagonal-jitter body this function used to inline.
    [L, ok] = ensurePD("jitter", S);
end
