function sampleParametersMD(obj)
%SAMPLEPARAMETERSMD Resample all model parameters (MD model).
%
%   Multi-dimensional counterpart of sampleParameters.m. The global
%   transition/cue probability samplers and the local transition/cue matrix
%   updates are dimension-agnostic and reused unchanged; only the dynamics
%   (Theta) and bias samplers have dedicated MD implementations.

    obj.sampleGlobalTransitionProbabilities();
    obj.sampleGlobalCueProbabilities();
    obj.sampleDynamicsMD();
    obj.sampleBiasMD();
    obj.updateLocalTransitionMatrix();
    obj.updateLocalCueMatrix();
end
