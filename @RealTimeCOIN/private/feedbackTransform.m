function [meanOut, covOut] = feedbackTransform(meanIn, covIn, bias, noiseCov)
%FEEDBACKTRANSFORM Map state moments to feedback moments (mean + bias, cov + R).
%
%   [meanOut, covOut] = feedbackTransform(meanIn, covIn, bias, noiseCov) applies
%   the observation model that turns a latent-state Gaussian into the predictive
%   feedback (observation-space) Gaussian: the mean is shifted by the sampled
%   bias and the covariance is inflated by the observation noise. It captures the
%   only difference between the state densities and their feedback counterparts,
%       feedback mean = state mean + bias,   feedback cov = state cov + R,
%   mirroring predictStateFeedback / predictStateFeedbackMD.
%
%   The transform is shape-agnostic: for the scalar model all inputs are scalars
%   (meanOut = M + B, covOut = V + R); for the multi-dimensional model meanIn and
%   bias are N-by-1 and covIn and noiseCov are N-by-N. Pure arithmetic on its
%   inputs; it does not read or mutate any model state.
    meanOut = meanIn + bias;
    covOut = covIn + noiseCov;
end
