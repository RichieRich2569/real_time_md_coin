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
    used = 0;

    if obj.state_dim == 1
        values = values(:)';
        R = obj.sigma_sensory_noise^2;
        densities = zeros(size(values));
        for p = 1:P
            if obj.D.C(p) >= obj.max_contexts
                continue;                      % no novel slot left in this particle
            end
            novel = obj.D.C(p) + 1;
            m = stationaryStateMean(obj, obj.D.retention(novel, p), obj.D.drift(novel, p));
            v = stationaryStateVar(obj, obj.D.retention(novel, p));
            densities = densities + obj.normal_pdf(values, m + obj.D.bias(novel, p), v + R);
            used = used + 1;
        end
        if used > 0
            densities = densities ./ used;
        end
        return;
    end

    N = obj.state_dim;
    if size(values, 1) ~= N
        error('RealTimeCOIN:GridDimensionMismatch', ...
            ['novel_state_feedback_probability expects an %d-by-K grid (each ', ...
             'column a query point) for state_dim == %d; received a %d-by-%d array.'], ...
            N, N, size(values, 1), size(values, 2));
    end
    Q = processNoiseCov(obj);
    R = observationNoiseCov(obj);
    densities = zeros(1, size(values, 2));
    for p = 1:P
        if obj.D.C(p) >= obj.max_contexts
            continue;                          % no novel slot left in this particle
        end
        novel = obj.D.C(p) + 1;
        A = obj.D.Theta(:, 1:N, novel, p);
        d = obj.D.Theta(:, N+1, novel, p);
        m = stationaryStateMeanMD(obj, A, d) + obj.D.bias(:, novel, p);
        V = stationaryStateCovMD(obj, A, Q) + R;
        densities = densities + gaussianPdfColumnsMD(obj, values, m, V);
        used = used + 1;
    end
    if used > 0
        densities = densities ./ used;
    end
end
