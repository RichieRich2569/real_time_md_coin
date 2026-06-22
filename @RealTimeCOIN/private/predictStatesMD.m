function predictStatesMD(obj)
%PREDICTSTATESMD Multivariate Kalman prediction step.
%
%   Multi-dimensional counterpart of predictStates.m. For each context j of
%   each particle p, propagate the latent state mean and covariance one step
%   under the augmented dynamics Theta = [A | d]:
%
%       s_{i|i-1} = A s_{i-1|i-1} + d
%       P_{i|i-1} = A P_{i-1|i-1} A' + Q
%
%   The (single) novel context slot C+1 is (re)seeded to the stationary
%   distribution of its sampled dynamics, mirroring the scalar code which
%   resets the novel context to d/(1-a), Q/(1-a^2).

    N = obj.state_dim;
    Cmax = obj.max_contexts + 1;
    P = obj.num_particles;
    Q = obj.processNoiseCov();

    obj.D.state_mean = zeros(N, Cmax, P);
    obj.D.state_cov = zeros(N, N, Cmax, P);

    for p = 1:P
        for c = 1:Cmax
            A = obj.D.Theta(:, 1:N, c, p);
            d = obj.D.Theta(:, N+1, c, p);
            sf = obj.D.state_filtered_mean(:, c, p);
            Pf = obj.D.state_filtered_cov(:, :, c, p);

            obj.D.state_mean(:, c, p) = A * sf + d;
            Pp = A * Pf * A' + Q;
            obj.D.state_cov(:, :, c, p) = (Pp + Pp') ./ 2;
        end

        % Re-seed the novel context to its stationary distribution.
        novel = min(obj.D.C(p) + 1, Cmax);
        if obj.D.C(p) < obj.max_contexts
            A = obj.D.Theta(:, 1:N, novel, p);
            d = obj.D.Theta(:, N+1, novel, p);
            obj.D.state_mean(:, novel, p) = obj.stationaryStateMeanMD(A, d);
            obj.D.state_cov(:, :, novel, p) = obj.stationaryStateCovMD(A, Q);
        end
    end
end
