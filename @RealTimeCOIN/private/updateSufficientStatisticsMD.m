function updateSufficientStatisticsMD(obj, y, q, obsMask)
%UPDATESUFFICIENTSTATISTICSMD Accumulate sufficient statistics (MD model).
%
%   Multi-dimensional counterpart of updateSufficientStatistics.m. The
%   context-transition and cue counts (n_context, n_cue) are accumulated
%   exactly as in the scalar model. The dynamics statistics become the matrix
%   accumulators of the matrix-normal regression s_i = Theta x_{i-1} + w,
%   with augmented regressor x_{i-1} = [s_{i-1}; 1]:
%
%       Lambda_xx <- Lambda_xx + x_{i-1} x_{i-1}'      ((N+1) x (N+1))
%       Lambda_yx <- Lambda_yx + s_i      x_{i-1}'      (N x (N+1))
%
%   computed from the sampled latent trajectory (previous_x_dynamics,
%   x_dynamics). As in the scalar code, only contexts that have been visited
%   (observedRows) accumulate, and accumulation starts after the first trial.
%   The optional bias statistics collect the observation residual y - s_i and
%   the per-context observation count.

    N = obj.state_dim;
    Cmax = obj.max_contexts + 1;
    P = obj.num_particles;
    if nargin < 4 || isempty(obsMask)
        obsMask = ~isnan(y(:));
    else
        obsMask = obsMask(:);
    end
    hasObservation = ~isempty(y) && any(obsMask);

    % --- Context-transition and cue counts (identical to scalar model) ---
    idx = sub2ind([Cmax, Cmax, P], obj.D.previous_context, obj.D.context, 1:P);
    obj.D.n_context(idx) = obj.D.n_context(idx) + 1;

    if ~isempty(q)
        obj.ensureCueColumn(q);
        idxCue = sub2ind(size(obj.D.n_cue), obj.D.context, q * ones(1, P), 1:P);
        obj.D.n_cue(idxCue) = obj.D.n_cue(idxCue) + 1;
    end

    % --- Matrix dynamics sufficient statistics ---
    if obj.trial > 0
        observedRows = squeeze(sum(obj.D.n_context, 2)) > 0;  % Cmax x P
        if P == 1
            observedRows = observedRows(:);
        end
        for p = 1:P
            for c = 1:Cmax
                if ~observedRows(c, p)
                    continue;
                end
                xAug = [obj.D.previous_x_dynamics(:, c, p); 1];   % (N+1) x 1
                s = obj.D.x_dynamics(:, c, p);                    % N x 1
                obj.D.Lambda_xx(:, :, c, p) = obj.D.Lambda_xx(:, :, c, p) + (xAug * xAug');
                obj.D.Lambda_yx(:, :, c, p) = obj.D.Lambda_yx(:, :, c, p) + (s * xAug');
            end
        end
    end

    % --- Bias sufficient statistics ---
    if obj.infer_bias && hasObservation
        yv = y(:);
        obsIdx = find(obsMask);
        R = obj.observationNoiseCov();
        Ri_obs = obj.safeInverse(R(obsIdx, obsIdx));
        precisionUpdate = zeros(N, N);
        precisionUpdate(obsIdx, obsIdx) = Ri_obs;
        for p = 1:P
            c = obj.D.context(p);
            infoUpdate = zeros(N, 1);
            infoUpdate(obsIdx) = Ri_obs * (yv(obsIdx) - obj.D.x_bias(obsIdx, p));
            obj.D.bias_info_ss(:, c, p) = obj.D.bias_info_ss(:, c, p) + infoUpdate;
            obj.D.bias_precision_ss(:, :, c, p) = ...
                obj.D.bias_precision_ss(:, :, c, p) + precisionUpdate;
        end
    end
end
