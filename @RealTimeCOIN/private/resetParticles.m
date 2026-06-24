function resetParticles(obj)
    % Dispatch to the multi-dimensional initialiser when state_dim > 1; the
    % scalar branch below is left exactly as in the original implementation
    % so that the default (state_dim == 1) behaviour is unchanged.
    if obj.state_dim > 1
        resetParticlesMD(obj);
        return;
    end

    Cmax = obj.max_contexts + 1;
    P = obj.num_particles;
    D = struct();
    D.C = ones(1, P);
    D.context = ones(1, P);
    D.previous_context = ones(1, P);
    D.Q = 0;

    D.n_context = zeros(Cmax, Cmax, P);
    D.n_cue = zeros(Cmax, 1, P);
    D.dynamics_ss_1 = zeros(Cmax, P, 2);
    D.dynamics_ss_2 = zeros(Cmax, P, 2, 2);
    D.bias_ss_1 = zeros(Cmax, P);
    D.bias_ss_2 = zeros(Cmax, P);

    D.global_transition_probabilities = zeros(Cmax, P);
    D.global_transition_probabilities(1,:) = 1;
    D.global_cue_probabilities = ones(1, P);

    D.retention = obj.sampleScalarNormal(obj.prior_mean_retention, ...
        obj.precisionToVariance(obj.prior_precision_retention), [Cmax, P], 0, 1);
    D.drift = obj.sampleScalarNormal(obj.prior_mean_drift, ...
        obj.precisionToVariance(obj.prior_precision_drift), [Cmax, P], -Inf, Inf);
    if obj.infer_bias
        D.bias = obj.sampleScalarNormal(obj.prior_mean_bias, ...
            obj.precisionToVariance(obj.prior_precision_bias), [Cmax, P], -Inf, Inf);
    else
        D.bias = zeros(Cmax, P);
    end

    D.state_filtered_mean = obj.stationaryStateMean(D.retention, D.drift);
    D.state_filtered_var = obj.stationaryStateVar(D.retention);
    D.previous_state_filtered_mean = D.state_filtered_mean;
    D.previous_state_filtered_var = D.state_filtered_var;
    D.state_mean = D.state_filtered_mean;
    D.state_var = D.state_filtered_var;
    D.state_feedback_mean = D.state_mean + D.bias;
    D.state_feedback_var = D.state_var + obj.observationVariance();

    D.prior_probabilities = zeros(Cmax, P);
    D.prior_probabilities(1,:) = 1;
    D.predicted_probabilities = D.prior_probabilities;
    D.responsibilities = D.prior_probabilities;
    D.probability_cue = ones(Cmax, P);
    D.probability_state_feedback = ones(Cmax, P);
    D.i_resampled = 1:P;

    obj.D = D; %#ok<*PROP>
    obj.updateLocalTransitionMatrix();
    obj.updateLocalCueMatrix();
    obj.alignment_seed = [];
    obj.invalidateContextAlignment();
end
