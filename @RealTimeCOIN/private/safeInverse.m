function Ainv = safeInverse(~, A)
%SAFEINVERSE Invert a square matrix, falling back to the pseudo-inverse.
%   Ainv = safeInverse(obj, A) returns inv(A) for well-conditioned A and
%   pinv(A) when A is numerically singular (reciprocal condition number
%   rcond(A) < 1e-12). This guards the multi-dimensional Kalman/precision
%   updates against blow-up when a covariance becomes degenerate. The leading
%   obj argument is ignored (private RealTimeCOIN method).
    if rcond(A) < 1e-12
        Ainv = pinv(A);
    else
        Ainv = inv(A);
    end
end
