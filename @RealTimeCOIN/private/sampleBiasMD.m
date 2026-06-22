function sampleBiasMD(obj)
%SAMPLEBIASMD Sample the observation bias vector b (multivariate conjugate).
%
%   Multi-dimensional counterpart of sampleBias.m. The observation model
%   y = s + b + v, v ~ N(0, R) with the isotropic Gaussian prior
%   b ~ N(prior_mean_bias * 1, prior_precision_bias^-1 * I) yields the
%   conjugate Gaussian posterior per context per particle
%       postPrec = prior_precision_bias * I + n * R^-1
%       postMean = postCov (prior_precision_bias * prior_mean_bias * 1 + R^-1 * sum(y - s))
%   where n = bias_ss_2 (observation count) and sum(y - s) = bias_ss_1. At
%   N == 1 with R = obsVar this is exactly varB / muB in sampleBias.m. When
%   infer_bias is false the bias is held at zero.

    N = obj.state_dim;
    Cmax = obj.max_contexts + 1;
    P = obj.num_particles;

    if ~obj.infer_bias
        obj.D.bias = zeros(N, Cmax, P);
        return;
    end

    R = obj.observationNoiseCov();
    Ri = obj.safeInverse(R);
    priorPrec = obj.prior_precision_bias * eye(N);
    priorTerm = obj.prior_precision_bias * obj.prior_mean_bias * ones(N, 1);

    for p = 1:P
        for c = 1:Cmax
            n = obj.D.bias_ss_2(c, p);
            postPrec = priorPrec + n * Ri;
            postCov = obj.safeInverse(postPrec);
            postCov = (postCov + postCov') ./ 2;
            postMean = postCov * (priorTerm + Ri * obj.D.bias_ss_1(:, c, p));

            [L, ~] = obj.choljitter(postCov);
            obj.D.bias(:, c, p) = postMean + L * randn(N, 1);
        end
    end
end
