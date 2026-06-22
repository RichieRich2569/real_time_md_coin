function resampleParticlesMD(obj, y, q)
%RESAMPLEPARTICLESMD Weight and resample particles (multivariate likelihood).
%
%   Multi-dimensional counterpart of resampleParticles.m. The only change
%   from the scalar version is that the per-context observation likelihood
%   p(y | c) is a multivariate Gaussian evaluated via a stable Cholesky
%   factorisation (gaussianLogLikChol) instead of a scalar normal pdf. The
%   log-joint combination, responsibilities, and systematic resampling are
%   identical in structure to the scalar routine.
%
%   log p(y, q, c) = log p(c | history) + log p(q | c) + log p(y | c)

    P = obj.num_particles;
    Cmax = obj.max_contexts + 1;

    if isempty(y)
        log_py = zeros(Cmax, P);
        py = ones(Cmax, P);
    else
        yv = y(:);
        log_py = zeros(Cmax, P);
        for p = 1:P
            for c = 1:Cmax
                innovation = yv - obj.D.state_feedback_mean(:, c, p);
                S = obj.D.state_feedback_cov(:, :, c, p);
                log_py(c, p) = obj.gaussianLogLikChol(innovation, S);
            end
        end
        py = exp(log_py);
    end
    obj.D.probability_state_feedback = py;

    log_pc = obj.safeLog(obj.D.prior_probabilities);
    if ~isempty(q)
        log_pc = log_pc + obj.safeLog(obj.D.probability_cue);
    end
    if ~isempty(y)
        % Add the log-likelihood directly (rather than log(exp(.))) for
        % numerical robustness when components underflow.
        log_pc = log_pc + log_py;
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
    obj.resampleStateMD(idx);
    obj.D.responsibilities = obj.normalizeColumns(resp(:, idx));
end
