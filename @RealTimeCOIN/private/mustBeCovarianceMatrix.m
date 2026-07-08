function mustBeCovarianceMatrix(V)
%MUSTBECOVARIANCEMATRIX Argument validator: pages of V are covariance matrices.
%
%   mustBeCovarianceMatrix(V) is an arguments-block validator shared by the
%   multi-dimensional state_probability / state_feedback_probability density
%   mixtures. V is an N-by-N-by-C-by-P stack of per-(context, particle)
%   covariance pages V(:,:,c,p). Each page must be real, square, symmetric,
%   have nonnegative variances on its diagonal, and (for N > 1) satisfy the
%   Cauchy-Schwarz bound |cov(i,j)| <= sqrt(var(i) * var(j)). Empty V passes
%   trivially. The check is vectorised page-wise so it stays cheap on the full
%   particle stack.
%
%   Tolerances scale with the largest entry of V so the screen is invariant to
%   the units of the state space rather than tied to an absolute epsilon.
%
%   See also state_probability, state_feedback_probability.
    if ~isreal(V)
        error("RealTimeCOIN:InvalidCovarianceMatrix", ...
            "Covariance matrices must be real.");
    end

    N = size(V, 1);
    if size(V, 2) ~= N
        error("RealTimeCOIN:InvalidCovarianceMatrix", ...
            "Each covariance matrix must be square; received %d-by-%d pages.", ...
            size(V, 1), size(V, 2));
    end

    if isempty(V)
        return;
    end

    tol = 1e-10 * max(1, max(abs(V), [], 'all'));
    if any(abs(V - permute(V, [2 1 3 4])) > tol, 'all')
        error("RealTimeCOIN:InvalidCovarianceMatrix", ...
            "Each covariance matrix must be symmetric.");
    end

    pages = reshape(V, N * N, []);
    variances = pages(1:(N + 1):end, :);
    if any(variances < -tol, 'all')
        error("RealTimeCOIN:InvalidCovarianceMatrix", ...
            "Covariance matrix variances must be nonnegative.");
    end

    if N > 1
        sigma = sqrt(max(variances, 0));
        covLimit = reshape(sigma, N, 1, []) .* reshape(sigma, 1, N, []);
        if any(abs(reshape(V, N, N, [])) - covLimit > tol, 'all')
            error("RealTimeCOIN:InvalidCovarianceMatrix", ...
                "Covariances must satisfy |cov(i,j)| <= sqrt(var(i)*var(j)).");
        end
    end
end
