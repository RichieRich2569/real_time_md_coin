function set_stationary(obj)
    obj.updateLocalTransitionMatrix();
    obj.updateLocalCueMatrix();

    Cmax = obj.max_contexts + 1;
    P = obj.num_particles;

    obj.D.prior_probabilities = zeros(Cmax, P);
    obj.D.predicted_probabilities = zeros(Cmax, P);
    obj.D.responsibilities = zeros(Cmax, P);
    obj.D.probability_cue = ones(Cmax, P);
    obj.D.probability_state_feedback = ones(Cmax, P);
    obj.D.i_resampled = 1:P;

    for p = 1:P
        valid = false(1, Cmax);
        valid(1:obj.D.C(p)) = true;
        if obj.D.C(p) < obj.max_contexts
            valid(obj.D.C(p) + 1) = true;
        end

        T = obj.D.local_transition_matrix(valid, valid, p);
        pi = RealTimeCOIN.stationary_distribution(T);

        obj.D.prior_probabilities(valid, p) = pi(:);
        obj.D.predicted_probabilities(valid, p) = pi(:);
        obj.D.responsibilities(valid, p) = pi(:);

        validIdx = find(valid);
        cumulative = cumsum(pi(:)');
        cumulative(end) = 1;
        obj.D.context(p) = validIdx(find(rand <= cumulative, 1, 'first'));
    end
    obj.D.previous_context = obj.D.context;

    if obj.state_dim > 1
        N = obj.state_dim;
        Q = obj.processNoiseCov();
        obj.D.state_filtered_mean = zeros(N, Cmax, P);
        obj.D.state_filtered_cov = zeros(N, N, Cmax, P);
        for p = 1:P
            for c = 1:Cmax
                A = obj.D.Theta(:, 1:N, c, p);
                d = obj.D.Theta(:, N+1, c, p);
                obj.D.state_filtered_mean(:, c, p) = obj.stationaryStateMeanMD(A, d);
                obj.D.state_filtered_cov(:, :, c, p) = obj.stationaryStateCovMD(A, Q);
            end
        end
        obj.D.previous_state_filtered_mean = obj.D.state_filtered_mean;
        obj.D.previous_state_filtered_cov = obj.D.state_filtered_cov;
        obj.D.state_mean = obj.D.state_filtered_mean;
        obj.D.state_cov = obj.D.state_filtered_cov;
        obj.predictStateFeedbackMD();
    else
        obj.D.state_filtered_mean = obj.stationaryStateMean(obj.D.retention, obj.D.drift);
        obj.D.state_filtered_var = obj.stationaryStateVar(obj.D.retention);
        obj.D.previous_state_filtered_mean = obj.D.state_filtered_mean;
        obj.D.previous_state_filtered_var = obj.D.state_filtered_var;
        obj.D.state_mean = obj.D.state_filtered_mean;
        obj.D.state_var = obj.D.state_filtered_var;
        obj.predictStateFeedback();
    end

    obj.pending_q = [];
    obj.trial = 0;
    obj.invalidateContextAlignment();
end
