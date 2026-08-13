function [out, ok] = ensurePD(mode, M)
%ENSUREPD Positive-definite repair backing the three particle-filter tactics.
%
%   [out, ok] = ensurePD(mode, M) applies one of three numerically distinct
%   positive-definite-repair tactics to the matrix M, selected by mode:
%
%     "jitter"  - lower-triangular Cholesky factor with an escalating diagonal
%                 jitter fallback (backs choljitter). out is the factor L and
%                 ok reports whether a genuine Cholesky succeeded.
%     "load"    - zero non-finite entries, symmetrise, and diagonally load to a
%                 workable condition number (backs regularizeCovariance). out is
%                 the conditioned covariance.
%     "eigclip" - symmetrise and clip negative eigenvalues to zero, projecting
%                 to the nearest PSD matrix (backs stationaryStateCovMD). out is
%                 the projected covariance.
%
%   Each branch is a verbatim copy of the original caller's body, so the three
%   tactics remain distinct and every caller's numeric output is unchanged. ok
%   is only meaningful for the "jitter" mode; it defaults to true otherwise.

    ok = true;
    switch mode
        case "jitter"
            % --- verbatim choljitter body (operates on M == S) ---
            S = (M + M') ./ 2;
            [L, flag] = chol(S, 'lower');
            if flag == 0
                out = L;
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
                    out = L;
                    ok = true;
                    return;
                end
                jit = jit * 10;
            end

            % Last-resort diagonal fallback (independent dimensions).
            d = max(diag(S), eps);
            out = diag(sqrt(d));
            ok = false;
        case "load"
            % --- verbatim regularizeCovariance body (operates on M == covar) ---
            covar = M;
            covar(~isfinite(covar)) = 0;
            covar = (covar + covar') ./ 2;   % enforce exact symmetry
            if isempty(covar)
                out = eps;                   % empty -> smallest positive scalar variance
                return;
            end
            % Add eps on the diagonal (eps = smallest resolvable spacing near 1) so a
            % zero/rank-deficient covariance becomes strictly positive definite.
            covar = covar + eps .* eye(size(covar));
            % rcond < 1e-12 flags a near-singular matrix (reciprocal condition number
            % threshold); bump the diagonal by 1e-9 to restore a usable condition.
            if rcond(covar) < 1e-12
                covar = covar + 1e-9 .* eye(size(covar));
            end
            out = covar;
        case "eigclip"
            % --- verbatim stationaryStateCovMD PSD-projection epilogue ---
            P = (M + M') ./ 2;
            % Project to the nearest PSD matrix (clip negative eigenvalues) so a new
            % context is always seeded with a usable covariance.
            [Vc, Dc] = eig(P);
            dvals = max(real(diag(Dc)), 0);
            P = Vc * diag(dvals) * Vc';
            out = (P + P') ./ 2;
        otherwise
            error("RealTimeCOIN:ensurePD:UnknownMode", ...
                "Unknown PD-repair mode '%s'.", mode);
    end
end
