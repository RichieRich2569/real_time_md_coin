function v = stationaryStateVar(obj, a)
%STATIONARYSTATEVAR Stationary variance of the scalar latent AR(1) process.
%
%   v = stationaryStateVar(obj, a) returns the stationary variance of the
%   AR(1) state s_i = a*s_{i-1} + w with w ~ N(0, sigma_process_noise^2),
%   which solves v = a^2*v + sigma_process_noise^2, i.e.
%       v = sigma_process_noise^2 / (1 - a^2).
%   Input a (retention) may be an array; the result is element-wise. This is
%   the scalar counterpart of stationaryStateCovMD.m.

    denom = 1 - a.^2;
    v = zeros(size(a));
    % Only divide where 1 - a^2 exceeds eps (smallest resolvable spacing); an
    % |a| at/above 1 has no finite stationary variance, so leave those at 0.
    good = denom > eps;
    v(good) = obj.sigma_process_noise^2 ./ denom(good);
end
