function resampleState(obj, idx)
%RESAMPLESTATE Resample all particle-indexed state for the scalar model.
%   resampleState(obj, idx) reorders every field of obj.D that carries a
%   per-particle axis so that particle k becomes the old particle idx(k),
%   applying the resampling index chosen in resampleParticles. Fields are
%   grouped by tensor rank into explicit name lists (mirroring
%   resampleStateMD): 1-by-P row vectors, Cmax-by-P matrices, and a small set
%   of higher-rank tensors that hold the particle on a specific trailing axis.
%
%   The explicit lists replace the earlier size-based heuristic (which guessed
%   the particle axis from numel(X)==P / size(X,2)==P). The result is identical
%   for the current schema but no longer silently mis-resamples a future field
%   whose second dimension happens to equal the particle count.
%
%   idx is the 1-by-P resampling index (values in 1..P, repeats allowed). After
%   reordering, the previous-trial filtered estimate is refreshed so the
%   next-trial smoother in sampleStates operates on the resampled ancestry.
%
%   See also RESAMPLESTATEMD, RESAMPLEPARTICLES.

    % 1-by-P per-particle row vectors: index along the single particle axis.
    vecFields = {'C', 'context'};
    for i = 1:numel(vecFields)
        f = vecFields{i};
        obj.D.(f) = obj.D.(f)(idx);
    end

    % Cmax-by-P fields: the particle is the column (2nd) axis.
    matFields = {'prior_probabilities', 'predicted_probabilities', ...
        'probability_cue', 'probability_state_feedback', ...
        'global_transition_probabilities', 'retention', 'drift', 'bias', ...
        'state_mean', 'state_var', 'state_feedback_mean', 'state_feedback_var', ...
        'state_filtered_mean', 'state_filtered_var', 'bias_ss_1', 'bias_ss_2'};
    for i = 1:numel(matFields)
        f = matFields{i};
        obj.D.(f) = obj.D.(f)(:, idx);
    end

    % Higher-rank fields carry the particle on an explicit trailing axis.
    obj.D.n_context = obj.D.n_context(:, :, idx);           % Cmax x Cmax x P
    obj.D.n_cue = obj.D.n_cue(:, :, idx);                   % Cmax x 1    x P
    obj.D.dynamics_ss_1 = obj.D.dynamics_ss_1(:, idx, :);   % Cmax x P    x 2
    obj.D.dynamics_ss_2 = obj.D.dynamics_ss_2(:, idx, :, :); % Cmax x P x 2 x 2
    obj.D.global_cue_probabilities = obj.D.global_cue_probabilities(:, idx); % 1 x P
    obj.D.local_transition_matrix = obj.D.local_transition_matrix(:, :, idx);
    obj.D.local_cue_matrix = obj.D.local_cue_matrix(:, :, idx);

    % Refresh the lag (previous-trial) filtered estimate for the smoother.
    obj.D.previous_state_filtered_mean = obj.D.state_filtered_mean;
    obj.D.previous_state_filtered_var = obj.D.state_filtered_var;
end
