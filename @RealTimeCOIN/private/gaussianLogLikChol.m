function logLik = gaussianLogLikChol(obj, yTilde, S)
%GAUSSIANLOGLIKCHOL Stable multivariate Gaussian log-likelihood of a residual.
%   logLik = gaussianLogLikChol(obj, yTilde, S) returns log N(yTilde | 0, S),
%   the Gaussian log-density of innovation yTilde under innovation covariance S,
%   evaluated stably via S's Cholesky factor (no explicit inverse). See the
%   derivation below. At M == 1 it reduces to log(RealTimeCOIN.normal_pdf).
%
% =========================================================================
% MATHEMATICAL PROOF: MULTIVARIATE LIKELIHOOD VIA CHOLESKY FACTORISATION
% =========================================================================
% Given the innovation (measurement residual) yTilde = y - yhat and the
% innovation covariance S (S = P_pred + R for the identity-observation
% model), the Gaussian density of the observation is
%     p(y) = (2*pi)^(-M/2) * det(S)^(-1/2) * exp(-0.5 * yTilde' * S^-1 * yTilde).
%
% To evaluate this stably and without forming S^-1 explicitly, factor S by
% its lower Cholesky factor L (S = L*L'):
% 1. Determinant:
%        det(S) = det(L)*det(L') = prod(diag(L))^2
%        log(det(S)) = 2 * sum(log(diag(L))).
% 2. Mahalanobis distance:
%        yTilde' * S^-1 * yTilde = yTilde' * (L*L')^-1 * yTilde
%                                = (L^-1 * yTilde)' * (L^-1 * yTilde)
%                                = foo' * foo,    foo = L \ yTilde.
% This is an O(M^3) computation that guarantees positive definiteness and
% avoids the numerical hazards of an explicit inverse. At M == 1 it reduces
% to the scalar log of RealTimeCOIN.normal_pdf used in the scalar pipeline.
% =========================================================================

    M = numel(yTilde);
    yTilde = yTilde(:);
    [L, ok] = obj.choljitter(S);
    if ~ok
        % choljitter returned a diagonal fallback; treat dimensions as
        % independent, which still yields a finite, well-ordered likelihood.
        L = diag(max(diag(L), sqrt(eps)));
    end
    logDetS = 2 * sum(log(diag(L)));
    foo = L \ yTilde;
    mahalanobis = foo' * foo;
    logLik = -0.5 * (M * log(2*pi) + logDetS + mahalanobis);
end
