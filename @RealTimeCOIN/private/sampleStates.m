function sampleStates(obj, y)
%SAMPLESTATES Sample latent state trajectory (RTS smoother, scalar model).
%   sampleStates(obj, y) draws, for every context/particle, a joint sample of
%   the previous latent state s_{i-1} and the current latent state s_i given
%   the current feedback y. These sampled states feed the conjugate dynamics
%   and bias updates in updateSufficientStatistics. This is the scalar
%   (state_dim == 1) baseline; sampleStatesMD is the multivariate counterpart.
%
%   All quantities are (max_contexts+1)-by-num_particles arrays: one entry per
%   context (including the trailing "novel" slot) per particle. The routine has
%   two stages:
%     1. Backward step — a Rauch-Tung-Striebel (RTS) smoother draws s_{i-1}
%        using the Kalman-filtered and one-step-predicted moments held in obj.D.
%     2. Forward step — s_i is drawn from the one-step dynamics prior
%        N(a*s_{i-1}+d, qVar), which for the single active context is fused (in
%        precision form) with the observation likelihood N(y-bias, obsVar).
%
%   Inputs:
%     y   scalar feedback for the current trial, or [] for a missing
%         observation (in which case s_i is drawn from the prior only).

    qVar = obj.sigma_process_noise^2;      % process (state) noise variance Q
    obsVar = obj.observationVariance();    % observation noise variance R
    Cmax = obj.max_contexts + 1;           % context slots incl. the novel context
    P = obj.num_particles;

    % --- Stage 1: RTS smoother draw of the previous state s_{i-1} ------------
    % g is the scalar smoother gain J = a * P_{i-1|i-1} / P_{i|i-1}, i.e. the
    % retention coefficient scaled by the ratio of the previous filtered
    % variance to the current predicted variance (safeDivide guards P_pred==0).
    g = obj.D.retention .* obj.safeDivide(obj.D.previous_state_filtered_var, obj.D.state_var);
    % Smoothed mean/variance of s_{i-1}: correct the previous filtered estimate
    % by the smoother gain times the current filtered-minus-predicted residual.
    smoothMean = obj.D.previous_state_filtered_mean + g .* (obj.D.state_filtered_mean - obj.D.state_mean);
    smoothVar = obj.D.previous_state_filtered_var + g.^2 .* (obj.D.state_filtered_var - obj.D.state_var);
    smoothVar = max(smoothVar, 0);         % clamp tiny negative round-off to zero
    obj.D.previous_x_dynamics = obj.sampleScalarNormal(smoothMean, smoothVar, [Cmax, P], -Inf, Inf);

    % --- Stage 2: forward draw of the current state s_i ----------------------
    % One-step dynamics-prior mean: a*s_{i-1} + drift, per context/particle.
    dynMean = obj.D.retention .* obj.D.previous_x_dynamics + obj.D.drift;
    if isempty(y)
        % Missing observation: sample s_i straight from the dynamics prior.
        obj.D.x_dynamics = obj.sampleScalarNormal(dynMean, qVar, [Cmax, P], -Inf, Inf);
    else
        % Only the active context of each particle actually sees the
        % observation. Build the linear indices of those active entries and an
        % indicator mask "active" so the observation term is applied to exactly
        % one (context, particle) cell per column.
        active = zeros(Cmax, P);
        % sub2ind maps (row = active context label, col = particle 1..P) to a
        % single linear index into the Cmax-by-P array; obj.D.context is the
        % 1-by-P vector of active context labels.
        idx = sub2ind([Cmax, P], obj.D.context, 1:P);
        active(idx) = 1;
        if qVar == 0
            % Degenerate dynamics (deterministic state): prior is a point mass
            % at dynMean. If the observation is also noiseless, the active
            % context is pinned exactly to the residual y - bias.
            postMean = dynMean;
            postVar = zeros(Cmax, P);
            if obsVar == 0
                postMean(idx) = y - obj.D.bias(idx);
            end
        elseif obsVar == 0
            % Noiseless observation: the active context collapses onto the
            % residual y - bias (zero variance); all others keep the prior.
            postMean = dynMean;
            postVar = qVar * ones(Cmax, P);
            postMean(idx) = y - obj.D.bias(idx);
            postVar(idx) = 0;
        else
            % General Gaussian fusion in precision (information) form. Inactive
            % cells (active==0) keep the prior variance qVar; the active cell
            % adds the observation precision 1/obsVar and its residual.
            postVar = 1 ./ (1 ./ qVar + active ./ obsVar);
            postMean = postVar .* (dynMean ./ qVar + active .* (y - obj.D.bias) ./ obsVar);
        end
        obj.D.x_dynamics = obj.sampleScalarNormal(postMean, postVar, [Cmax, P], -Inf, Inf);
    end

    % Extract the sampled state of each particle's active context. This is the
    % value the bias update regresses the observation residual against, and
    % i_observed caches its linear index for updateSufficientStatistics.
    activeIdx = sub2ind([Cmax, P], obj.D.context, 1:P);
    obj.D.x_bias = obj.D.x_dynamics(activeIdx);
    obj.D.i_observed = activeIdx;
end
