function [L, ok] = choljitter(~, S)
%CHOLJITTER Lower-triangular Cholesky factor with PSD-failure fallback.
%
%   [L, ok] = choljitter(obj, S) returns a lower-triangular L with S = L*L'.
%   Particle-filter covariances occasionally drift very slightly non-positive
%   definite through accumulated round-off; rather than erroring, we
%   symmetrise S and add escalating diagonal jitter until the factorisation
%   succeeds. If even that fails (ok == false) we fall back to the diagonal
%   square root so downstream code degrades gracefully instead of throwing.

    S = (S + S') ./ 2;
    [L, flag] = chol(S, 'lower');
    if flag == 0
        ok = true;
        return;
    end

    scale = mean(diag(S));
    if ~isfinite(scale) || scale <= 0
        scale = 1;
    end
    jit = 1e-12 * scale;   % base jitter: 1e-12 relative to the mean diagonal
    for k = 1:8
        [L, flag] = chol(S + jit * eye(size(S)), 'lower');
        if flag == 0
            ok = true;
            return;
        end
        jit = jit * 10;
    end

    % Last-resort diagonal fallback (independent dimensions).
    d = max(diag(S), eps);
    L = diag(sqrt(d));
    ok = false;
end
