function sampleParameters(obj)
%SAMPLEPARAMETERS Resample all model parameters (scalar model).
%
%   sampleParameters(obj) runs the final stage of the per-trial particle
%   filter pipeline: given the freshly accumulated sufficient statistics it
%   draws new values for every conjugate/Gibbs parameter and then refreshes
%   the derived local transition and cue matrices used by predictContext.
%
%   Sampling order (each step conditions on the current sufficient stats):
%     1. sampleGlobalTransitionProbabilities - sticky HDP-HMM context betas
%     2. sampleGlobalCueProbabilities        - HDP cue-context betas
%     3. sampleDynamics                       - per-context [a; d] dynamics
%     4. sampleBias                           - per-context observation bias
%     5. updateLocalTransitionMatrix          - rebuild local rows from betas
%     6. updateLocalCueMatrix                 - rebuild local cue likelihoods
%
%   See sampleParametersMD.m for the multi-dimensional counterpart.

    obj.sampleGlobalTransitionProbabilities();
    obj.sampleGlobalCueProbabilities();
    obj.sampleDynamics();
    obj.sampleBias();
    obj.updateLocalTransitionMatrix();
    obj.updateLocalCueMatrix();
end
