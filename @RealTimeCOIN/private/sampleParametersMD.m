function sampleParametersMD(obj)
%SAMPLEPARAMETERSMD Resample all model parameters (MD model).
%
%   Multi-dimensional counterpart of sampleParameters.m. The global
%   transition/cue probability samplers and the local transition/cue matrix
%   updates are dimension-agnostic and reused unchanged; only the dynamics
%   (Theta) and bias samplers have dedicated MD implementations.

    obj.sampleGlobalTransitionProbabilities();  % sticky HDP-HMM context betas (shared)
    obj.sampleGlobalCueProbabilities();         % HDP cue-context betas (shared)
    obj.sampleDynamicsMD();                     % matrix-normal Theta = [A | d]
    obj.sampleBiasMD();                         % multivariate observation bias
    obj.updateLocalTransitionMatrix();          % rebuild local rows from betas (shared)
    obj.updateLocalCueMatrix();                 % rebuild local cue likelihoods (shared)
end
