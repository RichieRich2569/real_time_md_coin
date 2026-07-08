function [mu, v] = state_moments(obj)
%STATE_MOMENTS Predictive latent-state mean and (co)variance.
%
%   [mu, v] = state_moments(obj) returns the mean and variance of the
%   predictive latent state, marginalised over contexts and particles using
%   the predicted context probabilities. For the scalar model (state_dim==1)
%   mu and v are scalars. For the multi-dimensional model mu is an N-by-1
%   mean vector and v is the N-by-N covariance matrix, computed from the
%   per-context Gaussian mixture moments
%       E[s]   = sum_k w_k m_k
%       Cov[s] = sum_k w_k (V_k + m_k m_k') - E[s] E[s]'.
%
%   This is a read-only query: it draws no random numbers and does not mutate
%   particle state.
    arguments
        obj (1, 1) RealTimeCOIN
    end

    W = obj.D.predicted_probabilities;

    if obj.state_dim == 1
        mu = sum(W .* obj.D.state_mean, 'all') ./ obj.num_particles;
        second = sum(W .* (obj.D.state_var + obj.D.state_mean.^2), 'all') ./ obj.num_particles;
        v = max(second - mu.^2, 0);
        return;
    end

    % Multi-dimensional Gaussian-mixture moments. Kept as an explicit nested
    % loop (rather than vectorised) so the accumulation order is preserved
    % exactly; see "Deferred optimizations" in the quality review.
    N = obj.state_dim;
    Cmax = obj.max_contexts + 1;
    P = obj.num_particles;
    mu = zeros(N, 1);
    second = zeros(N, N);
    for p = 1:P
        for c = 1:Cmax
            w = W(c, p) / P;
            if w == 0
                continue;
            end
            m = obj.D.state_mean(:, c, p);
            mu = mu + w .* m;
            second = second + w .* (obj.D.state_cov(:, :, c, p) + (m * m'));
        end
    end
    v = second - (mu * mu');
    v = (v + v') ./ 2;
end
