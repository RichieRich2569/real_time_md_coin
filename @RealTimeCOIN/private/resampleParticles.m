function resampleParticles(obj, y, q)
%RESAMPLEPARTICLES Weight and resample particles (scalar likelihood).
%   resampleParticles(obj, y, q) forms the per-context log-joint over the
%   current particle set and resamples the particles in proportion to their
%   marginal weight. y is the (scalar) state-feedback observation and q the
%   cue; either may be [] to signal a missing component, in which case that
%   factor is dropped from the joint.
%
%   The per-context log-joint combines the three available factors:
%       log p(y, q, c) = log p(c | history)      (prior_probabilities)
%                      + log p(q | c)             (probability_cue,  if q given)
%                      + log p(y | c)             (Gaussian pdf,     if y given)
%   Marginalising over context (log-sum-exp down the context axis) gives each
%   particle's log weight l_w; the normalised per-context posterior is stored
%   as the responsibilities. When both y and q are missing the weights are
%   uniform, so resampling is skipped (identity index) to avoid needless
%   particle depletion.
%
%   Side effects on obj.D: probability_state_feedback (the likelihood py),
%   i_resampled (the chosen ancestry), responsibilities (post-resample), and
%   all particle-indexed fields via resampleState.
%
%   See also RESAMPLEPARTICLESMD, RESAMPLESTATE, SYSTEMATIC_RESAMPLING.

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
