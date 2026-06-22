function m = stationaryStateMeanMD(obj, A, d)
%STATIONARYSTATEMEANMD Stationary mean of the MD latent AR(1) process.
%
%   For s_i = A s_{i-1} + d + w_i the stationary mean solves m = A m + d, so
%       m = (I - A) \ d.
%   This is the multi-dimensional generalisation of stationaryStateMean.m
%   (scalar d/(1-a)). If (I - A) is near-singular (a unit eigenvalue) we fall
%   back to the pseudo-inverse so a new context is still seeded with a finite
%   mean rather than Inf/NaN.

    N = obj.state_dim;
    ImA = eye(N) - A;
    if rcond(ImA) < 1e-12
        m = pinv(ImA) * d(:);
    else
        m = ImA \ d(:);
    end
    m = m(:);
end
