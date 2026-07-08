function sampleBiasMD(obj)
%SAMPLEBIASMD Sample the observation bias vector b (multivariate conjugate).
%
%   Multi-dimensional counterpart of sampleBias.m. The observation model
%   y = s + b + v, v ~ N(0, R) with the isotropic Gaussian prior
%   b ~ N(prior_mean_bias * 1, prior_precision_bias^-1 * I) yields the
%   conjugate Gaussian posterior per context per particle
%       postPrec = prior_precision_bias * I + bias_precision_ss
%       postMean = postCov (prior_precision_bias * prior_mean_bias * 1 + bias_info_ss)
%   where the sufficient statistics are accumulated over whichever
%   coordinates were observed on each trial. Full observations reduce to the
%   previous n * R^-1 and R^-1 * sum(y - s) formula.

    N = obj.state_dim;
    Cmax = obj.max_contexts + 1;
    P = obj.num_particles;

    if ~obj.infer_bias
        obj.D.bias = zeros(N, Cmax, P);
        return;
    end

    priorPrec = obj.prior_precision_bias * eye(N);                       % isotropic prior precision
    priorTerm = obj.prior_precision_bias * obj.prior_mean_bias * ones(N, 1);  % prior contribution to info

    % One conjugate Gaussian draw per (context, particle). Each iteration
    % forms its own posterior covariance and Cholesky factor; see the deferred
    % note in sampleParametersMD callers about batching these factorizations.
    for p = 1:P
        for c = 1:Cmax
            postPrec = priorPrec + obj.D.bias_precision_ss(:, :, c, p);  % prior + observed precision
            postCov = obj.safeInverse(postPrec);
            postCov = (postCov + postCov') ./ 2;                         % re-symmetrize against round-off
            postMean = postCov * (priorTerm + obj.D.bias_info_ss(:, c, p));

            % Draw b = postMean + L*z, z ~ N(0, I), with L*L' = postCov.
            [L, ~] = obj.choljitter(postCov);
            obj.D.bias(:, c, p) = postMean + L * randn(N, 1);
        end
    end
end
