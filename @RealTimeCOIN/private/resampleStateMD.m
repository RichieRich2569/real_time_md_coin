function resampleStateMD(obj, idx)
%RESAMPLESTATEMD Resample all particle-indexed state for the MD model.
%
%   Multi-dimensional counterpart of resampleState.m. Because the MD tensors
%   carry the particle on different trailing dimensions (e.g. N x Cmax x P
%   for means, N x N x Cmax x P for covariances), fields are resampled
%   explicitly by shape rather than via the scalar size(X,2)==P heuristic,
%   which would misfire on tensors. As in the scalar routine, the previous
%   filtered estimate is refreshed at the end so the next-trial smoother in
%   sampleStatesMD has the correct lag-state.

    % Per-particle vectors and Cmax-by-P matrices.
    matFields = {'C', 'context', 'prior_probabilities', 'predicted_probabilities', ...
        'probability_cue', 'probability_state_feedback', ...
        'global_transition_probabilities', 'global_cue_probabilities', 'bias_ss_2'};
    for i = 1:numel(matFields)
        f = matFields{i};
        if ~isfield(obj.D, f)
            continue;
        end
        X = obj.D.(f);
        if isvector(X) && numel(X) == obj.num_particles
            obj.D.(f) = X(idx);
        else
            obj.D.(f) = X(:, idx);
        end
    end

    % N x Cmax x P fields (means, bias, bias residual stats).
    threeFields = {'state_mean', 'state_filtered_mean', 'state_feedback_mean', ...
        'bias', 'bias_ss_1', 'bias_info_ss'};
    for i = 1:numel(threeFields)
        f = threeFields{i};
        if isfield(obj.D, f)
            obj.D.(f) = obj.D.(f)(:, :, idx);
        end
    end

    % 4-D fields: covariances (N x N x Cmax x P), Theta, and matrix
    % sufficient statistics carry the particle on the 4th dimension.
    fourFields = {'state_cov', 'state_filtered_cov', 'state_feedback_cov', ...
        'Theta', 'Lambda_xx', 'Lambda_yx', 'bias_precision_ss'};
    for i = 1:numel(fourFields)
        f = fourFields{i};
        if isfield(obj.D, f)
            obj.D.(f) = obj.D.(f)(:, :, :, idx);
        end
    end

    % Context-inference tensors (shared shapes with the scalar model).
    obj.D.n_context = obj.D.n_context(:, :, idx);
    obj.D.n_cue = obj.D.n_cue(:, :, idx);
    obj.D.local_transition_matrix = obj.D.local_transition_matrix(:, :, idx);
    obj.D.local_cue_matrix = obj.D.local_cue_matrix(:, :, idx);

    % Refresh the lag (previous-trial) filtered estimate for the smoother.
    obj.D.previous_state_filtered_mean = obj.D.state_filtered_mean;
    obj.D.previous_state_filtered_cov = obj.D.state_filtered_cov;
end
