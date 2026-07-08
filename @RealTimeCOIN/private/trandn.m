function x = trandn(~, l, u)
%TRANDN Sample the standard normal truncated to [l, u], element-wise.
%   x = trandn(obj, l, u) returns a column vector the length of L (= length of
%   U) whose i-th entry is drawn from the standard normal N(0, 1) conditioned on
%   l(i) < X < u(i). Infinite entries of L and U are allowed (one-sided or
%   unbounded truncation). To sample the non-standard N(m, s^2) truncated to
%   (a, b), draw X = trandn((a - m)/s, (b - m)/s) and set Z = m + s*X.
%
%   The OBJ argument is unused (stateless helper exposed as a class method).
%
%   Algorithm (Botev 2016): each element is routed to one of three exact
%   acceptance-rejection schemes chosen by where the interval sits relative to a
%   threshold a = 0.66:
%     * l > a            : right tail, sampled by ntail (Rayleigh proposal);
%     * u < -a           : left tail, sampled by -ntail on the mirrored limits;
%     * otherwise        : interval straddling 0, sampled by tn.
%   The threshold 0.66 is Botev's tuned switch-over between the tail sampler and
%   the central sampler; it balances the acceptance rates of the two proposals.
%
%   Inputs:
%     l  Real vector of lower truncation limits (-Inf allowed).
%     u  Real vector of upper truncation limits (+Inf allowed), length(l).
%
%   Output:
%     x  length(l)-by-1 vector of truncated standard-normal draws.
%
%   Reference:
%     Botev, Z. I. (2016). "The normal law under linear restrictions:
%     simulation and estimation via minimax tilting". Journal of the Royal
%     Statistical Society: Series B (Statistical Methodology).
%     doi:10.1111/rssb.12162
%
%   See also sampleScalarNormal, sampleBivariateTruncated.

    if ~isnumeric(l) || ~isreal(l) || ~isnumeric(u) || ~isreal(u)
        error("RealTimeCOIN:trandn:invalidLimits", ...
            "Truncation limits l and u must be real numeric vectors.");
    end
    l = l(:);
    u = u(:);
    if length(l) ~= length(u)
        error("RealTimeCOIN:trandn:limitLength", ...
            "Truncation limits have to be vectors of the same length.");
    end
    if any(l > u)
        error("RealTimeCOIN:trandn:emptyInterval", ...
            "Each lower limit must not exceed its upper limit.");
    end

    x = nan(size(l));
    a = 0.66;   % Botev's tuned threshold for switching between the tail and
                % central samplers (see help).

    I = l > a;
    if any(I)
        tl = l(I);
        tu = u(I);
        x(I) = ntail(tl, tu);
    end

    J = u < -a;
    if any(J)
        tl = -u(J);
        tu = -l(J);
        x(J) = -ntail(tl, tu);
    end

    I = ~(I | J);
    if any(I)
        tl = l(I);
        tu = u(I);
        x(I) = tn(tl, tu);
    end
end

function x = ntail(l, u)
% Samples a column vector from the standard normal truncated to [l, u], where
% l > 0, using acceptance-rejection from a Rayleigh distribution.
    c = l.^2 / 2;
    n = length(l);
    f = expm1(c - u.^2 / 2);
    x = c - reallog(1 + rand(n, 1) .* f);
    I = find(rand(n, 1).^2 .* x > c);
    d = length(I);
    while d > 0
        cy = c(I);
        y = cy - reallog(1 + rand(d, 1) .* f(I));
        idx = rand(d, 1).^2 .* y < cy;
        x(I(idx)) = y(idx);
        I = I(~idx);
        d = length(I);
    end
    x = sqrt(2 * x);
end

function x = tn(l, u)
% Samples a column vector from the standard normal truncated to [l, u] using
% acceptance-rejection for wide intervals and inverse transform otherwise. The
% width threshold tol = 2 is Botev's switch-over: intervals wider than 2 use
% plain normal rejection (high acceptance), narrower ones use the numerically
% stable erfc/erfcinv inverse-transform draw.
    tol = 2;
    I = abs(u - l) > tol;
    x = l;
    if any(I)
        tl = l(I);
        tu = u(I);
        x(I) = trndReject(tl, tu);
    end

    I = ~I;
    if any(I)
        tl = l(I);
        tu = u(I);
        pl = erfc(tl / sqrt(2)) / 2;
        pu = erfc(tu / sqrt(2)) / 2;
        x(I) = sqrt(2) * erfcinv(2 * (pl - (pl - pu) .* rand(size(tl))));
    end
end

function x = trndReject(l, u)
% Uses acceptance-rejection from the untruncated standard normal to simulate
% from the truncated normal on wide intervals. (Named trndReject rather than
% trnd to avoid shadowing the Statistics Toolbox trnd.)
    x = randn(size(l));
    I = find(x < l | x > u);
    d = length(I);
    while d > 0
        ly = l(I);
        uy = u(I);
        y = randn(size(ly));
        idx = y > ly & y < uy;
        x(I(idx)) = y(idx);
        I = I(~idx);
        d = length(I);
    end
end
