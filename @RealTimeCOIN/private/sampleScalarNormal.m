function X = sampleScalarNormal(obj, mu, variance, sz, low, high)
%SAMPLESCALARNORMAL Draw truncated univariate normals element-wise.
%   X = sampleScalarNormal(obj, mu, variance, sz, low, high) returns an array of
%   size SZ whose entries are independent draws from N(mu, variance) truncated
%   to the interval [low, high]. MU, VARIANCE, LOW and HIGH may each be given as
%   a scalar (broadcast to SZ) or as an array already of size SZ.
%
%   Each stochastic entry is drawn as mu + sigma * trandn(l, u), where sigma =
%   sqrt(variance) and (l, u) are the standardised truncation limits. Entries
%   with variance <= 0 or a non-finite variance are treated as deterministic and
%   returned at mu (then clamped to the bounds), so degenerate priors and
%   fully-observed states pass through unchanged.
%
%   Bounds clamp. The result is finally clamped with
%       X = min(max(X, low), high - eps)
%   The -eps keeps every draw STRICTLY below the upper bound. This matters for
%   the retention parameter, which must satisfy a in [0, 1): a value of exactly
%   1 would make the latent AR(1) process non-stationary, so the open upper
%   bound is enforced numerically by subtracting one machine epsilon.
%
%   Inputs:
%     mu        Mean(s): scalar or array of size SZ.
%     variance  Variance(s): scalar or array of size SZ (<=0 => deterministic).
%     sz        Size vector for the output array.
%     low       Lower truncation bound(s): scalar or array of size SZ.
%     high      Upper truncation bound(s): scalar or array of size SZ.
%
%   Output:
%     X         Array of size SZ of truncated normal draws.
%
%   See also trandn, sampleBivariateTruncated.

    if ~isnumeric(mu) || ~isreal(mu) || ~isnumeric(variance) || ~isreal(variance)
        error("RealTimeCOIN:sampleScalarNormal:invalidMoments", ...
            "mu and variance must be real numeric arrays.");
    end
    if ~isnumeric(low) || ~isreal(low) || ~isnumeric(high) || ~isreal(high)
        error("RealTimeCOIN:sampleScalarNormal:invalidBounds", ...
            "low and high must be real numeric arrays.");
    end
    if ~isnumeric(sz) || ~isvector(sz) || any(sz < 0) || any(mod(sz, 1) ~= 0)
        error("RealTimeCOIN:sampleScalarNormal:invalidSize", ...
            "sz must be a vector of non-negative integers.");
    end

    if isscalar(mu)
        mu = mu .* ones(sz);
    end
    if isscalar(variance)
        variance = variance .* ones(sz);
    end
    if isscalar(low)
        low = low .* ones(sz);
    end
    if isscalar(high)
        high = high .* ones(sz);
    end

    X = mu;
    stochastic = variance > 0 & isfinite(variance);
    if any(stochastic, 'all')
        sigma = sqrt(variance(stochastic));
        l = (low(stochastic) - mu(stochastic)) ./ sigma;
        u = (high(stochastic) - mu(stochastic)) ./ sigma;
        X(stochastic) = mu(stochastic) + sigma .* obj.trandn(l, u);
    end
    X = min(max(X, low), high - eps);
end
