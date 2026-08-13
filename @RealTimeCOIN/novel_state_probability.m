function densities = novel_state_probability(obj, values)
%NOVEL_STATE_PROBABILITY Posterior state density of the novel (unseen) context.
%
%   Companion to state_given_context_probability. Returns the state density of
%   the single not-yet-instantiated ("novel") context, i.e. the belief about a
%   context the model has not observed yet. For the scalar model (state_dim ==
%   1) values is a vector and the density is a row vector of equal length; for
%   the multi-dimensional model values is an N-by-K matrix of column query
%   points and the density is a 1-by-K row.
%
%   Each particle contributes its novel slot's stationary state distribution
%   (the same re-seeding used by predictStates/predictStatesMD): mean d/(1-a)
%   and variance Q/(1-a^2) in the scalar case, and the multivariate stationary
%   moments otherwise. The result is the equal-weight mixture over the particles
%   that still have an available novel slot. If every particle has saturated its
%   context budget the density is all zeros.
    arguments
        obj (1, 1) RealTimeCOIN
        values (:, :) double {mustBeFinite, mustBeReal}
    end

    P = obj.num_particles;

    if obj.state_dim == 1
        used = 0;
        W = zeros(1, P);
        M = zeros(1, P);
        V = zeros(1, P);
        for p = 1:P
            if obj.D.C(p) >= obj.max_contexts
                continue;                      % no novel slot left in this particle
            end
            novel = obj.D.C(p) + 1;
            used = used + 1;
            W(used) = 1;
            M(used) = stationaryStateMean(obj, obj.D.retention(novel, p), obj.D.drift(novel, p));
            V(used) = stationaryStateVar(obj, obj.D.retention(novel, p));
        end
        densities = mixtureDensityOnGrid(obj, values, W(1:used), M(1:used), V(1:used), ...
            used, "novel_state_probability");
        return;
    end

    N = obj.state_dim;
    Q = processNoiseCov(obj);
    used = 0;
    W = zeros(1, P);
    M = zeros(N, 1, P);
    V = zeros(N, N, 1, P);
    for p = 1:P
        if obj.D.C(p) >= obj.max_contexts
            continue;                          % no novel slot left in this particle
        end
        novel = obj.D.C(p) + 1;
        A = obj.D.Theta(:, 1:N, novel, p);
        d = obj.D.Theta(:, N+1, novel, p);
        used = used + 1;
        W(used) = 1;
        M(:, 1, used) = stationaryStateMeanMD(obj, A, d);
        V(:, :, 1, used) = stationaryStateCovMD(obj, A, Q);
    end
    densities = mixtureDensityOnGrid(obj, values, W(1:used), M(:, 1, 1:used), ...
        V(:, :, 1, 1:used), used, "novel_state_probability");
end
