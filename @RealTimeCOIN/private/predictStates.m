function predictStates(obj)
%PREDICTSTATES Scalar Kalman prediction (time update) of the latent state.
%   predictStates(obj) propagates each context's filtered state one step under
%   the scalar linear-Gaussian dynamics s_i = a * s_{i-1} + d + w,
%   w ~ N(0, sigma_process_noise^2):
%
%       state_mean = a * state_filtered_mean + d
%       state_var  = a^2 * state_filtered_var + sigma_process_noise^2
%
%   Every already-instantiated context is propagated by the vectorised update
%   above; the (single) novel-context slot C+1 is instead (re)seeded to the
%   stationary distribution of its sampled dynamics, mean d/(1-a) and variance
%   sigma_process_noise^2/(1-a^2). This is the scalar counterpart of
%   predictStatesMD; keep it byte-for-byte equivalent to COIN.m's behaviour.
%
%   Writes obj.D.state_mean and obj.D.state_var
%   (each (max_contexts+1)-by-num_particles).

    qv = obj.sigma_process_noise^2;                 % process-noise variance Q
    % Vectorised Kalman predict across all contexts and particles at once.
    obj.D.state_mean = obj.D.retention .* obj.D.state_filtered_mean + obj.D.drift;
    obj.D.state_var = obj.D.retention.^2 .* obj.D.state_filtered_var + qv;

    for p = 1:obj.num_particles
        % Novel-context slot: one past the highest instantiated context, capped.
        novel = min(obj.D.C(p) + 1, obj.max_contexts + 1);
        if obj.D.C(p) < obj.max_contexts
            % Re-seed the novel context to its stationary (prior) distribution.
            obj.D.state_mean(novel,p) = obj.stationaryStateMean(obj.D.retention(novel,p), obj.D.drift(novel,p));
            obj.D.state_var(novel,p) = obj.stationaryStateVar(obj.D.retention(novel,p));
        end
    end
    obj.D.state_var = max(obj.D.state_var, 0);       % guard tiny negatives from round-off
end
