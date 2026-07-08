function updateSufficientStatistics(obj, y, q)
%UPDATESUFFICIENTSTATISTICS Accumulate conjugate sufficient statistics (scalar).
%   updateSufficientStatistics(obj, y, q) folds the current trial into the
%   running sufficient statistics that drive the conjugate/Gibbs parameter
%   updates in sampleParameters. Four groups are accumulated:
%     * n_context  — context-transition counts (previous -> current context).
%     * n_cue      — cue-emission counts per context (when a cue q is present).
%     * dynamics_ss_1 / dynamics_ss_2 — the cross- and Gram-statistics of the
%       scalar dynamics regression s_i = a*s_{i-1} + d + w, using the augmented
%       regressor x = [s_{i-1}; 1].
%     * bias_ss_1 / bias_ss_2 — residual sum and observation count for the bias.
%   This is the scalar (state_dim == 1) baseline; updateSufficientStatisticsMD
%   accumulates the matrix-valued generalisation.
%
%   Inputs:
%     y   scalar feedback ([] if the observation was missing).
%     q   integer cue index for this trial ([] if no cue channel is active).

    Cmax = obj.max_contexts + 1;
    P = obj.num_particles;

    % --- Context-transition counts ------------------------------------------
    % One count per particle for the observed (previous_context -> context)
    % pair; sub2ind selects that single cell of the Cmax x Cmax x P count tensor
    % for each particle column.
    idx = sub2ind([Cmax, Cmax, P], obj.D.previous_context, obj.D.context, 1:P);
    obj.D.n_context(idx) = obj.D.n_context(idx) + 1;

    % --- Cue-emission counts -------------------------------------------------
    if ~isempty(q)
        obj.ensureCueColumn(q);   % grow n_cue if this cue index is new
        % Select cell (current context, cue q, particle) in the n_cue tensor.
        idxCue = sub2ind(size(obj.D.n_cue), obj.D.context, q * ones(1, P), 1:P);
        obj.D.n_cue(idxCue) = obj.D.n_cue(idxCue) + 1;
    end

    % --- Dynamics regression sufficient statistics --------------------------
    % Skipped on trial 0 (no previous state yet). xAug stacks the augmented
    % regressor along the trailing dimension: slice 1 = s_{i-1}, slice 2 = 1.
    if obj.trial > 0
        xAug = ones(Cmax, P, 2);
        xAug(:,:,1) = obj.D.previous_x_dynamics;
        % Only contexts that have been visited (any incoming transition count)
        % contribute; observedRows is the Cmax x P mask of such contexts.
        observedRows = squeeze(sum(obj.D.n_context, 2)) > 0;
        % dynamics_ss_1 accumulates s_i * [s_{i-1}; 1] (the y*x cross term).
        ss1 = obj.D.x_dynamics .* xAug;
        obj.D.dynamics_ss_1 = obj.D.dynamics_ss_1 + ss1 .* observedRows;
        % dynamics_ss_2 accumulates the 2x2 Gram matrix [s_{i-1};1]*[s_{i-1};1]'
        % (x*x') for each context/particle; a,b index its four entries.
        for a = 1:2
            for b = 1:2
                obj.D.dynamics_ss_2(:,:,a,b) = obj.D.dynamics_ss_2(:,:,a,b) + ...
                    xAug(:,:,a) .* xAug(:,:,b) .* observedRows;
            end
        end
    end

    % --- Bias sufficient statistics -----------------------------------------
    % Accumulate the residual (y - active-context state) and a unit count at the
    % active-context cells (i_observed cached by sampleStates).
    if obj.infer_bias && ~isempty(y)
        obj.D.bias_ss_1(obj.D.i_observed) = obj.D.bias_ss_1(obj.D.i_observed) + (y - obj.D.x_bias);
        obj.D.bias_ss_2(obj.D.i_observed) = obj.D.bias_ss_2(obj.D.i_observed) + 1;
    end
end
