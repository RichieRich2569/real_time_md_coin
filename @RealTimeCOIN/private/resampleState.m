function resampleState(obj, idx)
    fields2 = {'C','context','prior_probabilities','predicted_probabilities', ...
        'probability_cue','probability_state_feedback','global_transition_probabilities', ...
        'retention','drift','bias','state_mean','state_var','state_feedback_mean', ...
        'state_feedback_var','state_filtered_mean','state_filtered_var', ...
        'bias_ss_1','bias_ss_2'};
    for i = 1:numel(fields2)
        f = fields2{i};
        X = obj.D.(f);
        if isvector(X) && numel(X) == obj.num_particles
            obj.D.(f) = X(idx);
        elseif size(X,2) == obj.num_particles
            obj.D.(f) = X(:, idx);
        end
    end
    obj.D.n_context = obj.D.n_context(:,:,idx);
    obj.D.n_cue = obj.D.n_cue(:,:,idx);
    obj.D.dynamics_ss_1 = obj.D.dynamics_ss_1(:,idx,:);
    obj.D.dynamics_ss_2 = obj.D.dynamics_ss_2(:,idx,:,:);
    obj.D.global_cue_probabilities = obj.D.global_cue_probabilities(:,idx);
    obj.D.local_transition_matrix = obj.D.local_transition_matrix(:,:,idx);
    obj.D.local_cue_matrix = obj.D.local_cue_matrix(:,:,idx);
    obj.D.previous_state_filtered_mean = obj.D.state_filtered_mean;
    obj.D.previous_state_filtered_var = obj.D.state_filtered_var;
end
