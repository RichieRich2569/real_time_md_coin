function x = trandn(~, l, u)
%% truncated normal generator
% * efficient generator of a vector of length(l)=length(u)
% from the standard multivariate normal distribution,
% truncated over the region [l,u];
% infinite values for 'u' and 'l' are accepted;
% * Remark:
% If you wish to simulate a random variable
% 'Z' from the non-standard Gaussian N(m,s^2)
% conditional on l<Z<u, then first simulate
% X=trandn((l-m)/s,(u-m)/s) and set Z=m+s*X;

% Reference:
% Botev, Z. I. (2016). "The normal law under linear restrictions:
% simulation and estimation via minimax tilting". Journal of the
% Royal Statistical Society: Series B (Statistical Methodology).
% doi:10.1111/rssb.12162

l = l(:);
u = u(:);
if length(l) ~= length(u)
    error('RealTimeCOIN:TruncationLimits', ...
        'Truncation limits have to be vectors of the same length');
end
x = nan(size(l));
a = .66; % threshold for switching between methods

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
% samples a column vector from the standard normal truncated to [l,u],
% where l > 0; uses acceptance-rejection from a Rayleigh distribution.
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
% samples a column vector from the standard normal truncated to [l,u] using
% acceptance-rejection for wide intervals and inverse transform otherwise.
tol = 2;
I = abs(u - l) > tol;
x = l;
if any(I)
    tl = l(I);
    tu = u(I);
    x(I) = trnd(tl, tu);
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

function x = trnd(l, u)
% uses acceptance rejection to simulate from truncated normal.
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
