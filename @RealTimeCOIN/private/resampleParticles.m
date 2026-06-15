function resampleParticles(obj, y, q)
    P = obj.num_particles;
    Cmax = obj.max_contexts + 1;
    if isempty(y)
        py = ones(Cmax, P);
    else
        py = obj.normal_pdf(y, obj.D.state_feedback_mean, obj.D.state_feedback_var);
    end
    obj.D.probability_state_feedback = py;

    log_pc = obj.safeLog(obj.D.prior_probabilities);
    if ~isempty(q)
        log_pc = log_pc + obj.safeLog(obj.D.probability_cue);
    end
    if ~isempty(y)
        log_pc = log_pc + obj.safeLog(py);
    end

    l_w = obj.log_sum_exp(log_pc, 1);
    log_resp = log_pc - l_w;
    resp = exp(log_resp);
    resp(~isfinite(resp)) = 0;

    if isempty(y) && isempty(q)
        idx = 1:P;
    else
        weights = exp(l_w - obj.log_sum_exp(l_w(:), 1));
        idx = obj.systematic_resampling(weights(:)');
    end

    obj.D.i_resampled = idx(:)';
    obj.resampleState(idx);
    obj.D.responsibilities = obj.normalizeColumns(resp(:, idx));
end
