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

    % Batched Kalman time update over all (context, particle) pages. pagemtimes
    % applies the same per-page GEMM as the original block-at-a-time loop:
    %   s_{i|i-1} = A s_f + d ,   P_{i|i-1} = A P_f A' + Q  (then symmetrised).
    A  = obj.D.Theta(:, 1:N, :, :);                       % N x N x Cmax x P
    d  = obj.D.Theta(:, N+1, :, :);                       % N x 1 x Cmax x P
    sf = reshape(obj.D.state_filtered_mean, N, 1, Cmax, P);
    Pf = obj.D.state_filtered_cov;                        % N x N x Cmax x P

    sMean = pagemtimes(A, sf) + d;                        % N x 1 x Cmax x P
    obj.D.state_mean = reshape(sMean, N, Cmax, P);
    Pp = pagemtimes(pagemtimes(A, Pf), 'none', A, 'transpose') + Q;
    obj.D.state_cov = (Pp + permute(Pp, [2 1 3 4])) ./ 2;  % symmetrise vs round-off

    % Re-seed each particle's novel context slot to its stationary distribution
    % (overwrites the generic propagation above for that slot, as before).
    for p = 1:P
        if obj.D.C(p) < obj.max_contexts
            novel = obj.D.C(p) + 1;
            An = obj.D.Theta(:, 1:N, novel, p);
            dn = obj.D.Theta(:, N+1, novel, p);
            obj.D.state_mean(:, novel, p) = obj.stationaryStateMeanMD(An, dn);
            obj.D.state_cov(:, :, novel, p) = obj.stationaryStateCovMD(An, Q);
        end
    end
end
