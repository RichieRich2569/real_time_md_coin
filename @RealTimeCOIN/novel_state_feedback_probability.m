function densities = novel_state_feedback_probability(obj, values)
%NOVEL_STATE_FEEDBACK_PROBABILITY Feedback density of the novel (unseen) context.
%
%   Feedback (observation-space) counterpart of novel_state_probability, and the
%   novel-context analogue of state_feedback_given_context_probability. Returns
%   the predictive feedback density y = state + bias + noise of the single
%   not-yet-instantiated ("novel") context. For the scalar model (state_dim ==
%   1) values is a vector and the density is a row vector; for the
%   multi-dimensional model values is an N-by-K matrix of column query points
%   and the density is a 1-by-K row.
%
%   Each particle contributes its novel slot's stationary state distribution
%   shifted by that slot's sampled bias and inflated by the observation noise R.
%   The result is the equal-weight mixture over the particles that still have an
%   available novel slot; if every particle has saturated its context budget the
%   density is all zeros.
    arguments
        obj (1, 1) RealTimeCOIN
        values (:, :) double {mustBeFinite, mustBeReal}
    end

    P = obj.num_particles;

    if obj.state_dim == 1
        R = obj.sigma_sensory_noise^2;
        used = 0;
        W = zeros(1, P);
        M = zeros(1, P);
        V = zeros(1, P);
        for p = 1:P
            if obj.D.C(p) >= obj.max_contexts
                continue;                      % no novel slot left in this particle
            end
            novel = obj.D.C(p) + 1;
            m = stationaryStateMean(obj, obj.D.retention(novel, p), obj.D.drift(novel, p));
            v = stationaryStateVar(obj, obj.D.retention(novel, p));
            used = used + 1;
            W(used) = 1;
            [M(used), V(used)] = feedbackTransform(m, v, obj.D.bias(novel, p), R);
        end
        densities = mixtureDensityOnGrid(obj, values, W(1:used), M(1:used), V(1:used), ...
            used, "novel_state_feedback_probability");
        return;
    end

    N = obj.state_dim;
    Q = processNoiseCov(obj);
    R = observationNoiseCov(obj);
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
        m = stationaryStateMeanMD(obj, A, d);
        Vc = stationaryStateCovMD(obj, A, Q);
        used = used + 1;
        W(used) = 1;
        [M(:, 1, used), V(:, :, 1, used)] = feedbackTransform(m, Vc, obj.D.bias(:, novel, p), R);
    end
    densities = mixtureDensityOnGrid(obj, values, W(1:used), M(:, 1, 1:used), ...
        V(:, :, 1, 1:used), used, "novel_state_feedback_probability");
end
