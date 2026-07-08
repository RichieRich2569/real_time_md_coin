function m = stationaryStateMean(~, a, d)
%STATIONARYSTATEMEAN Stationary mean of the scalar latent AR(1) process.
%
%   m = stationaryStateMean(obj, a, d) returns the stationary mean of the
%   AR(1) state s_i = a*s_{i-1} + d + w, which solves m = a*m + d, i.e.
%       m = d / (1 - a).
%   Inputs a (retention) and d (drift) may be arrays; the result is element-
%   wise. This is the scalar counterpart of stationaryStateMeanMD.m.

    denom = 1 - a;
    m = zeros(size(a));
    % Only divide where |1 - a| exceeds eps (the smallest resolvable spacing);
    % a retention at/above 1 has no finite stationary mean, so leave those at 0.
    good = abs(denom) > eps;
    m(good) = d(good) ./ denom(good);
end
