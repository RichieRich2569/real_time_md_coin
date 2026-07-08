function alignment = computeContextAlignment(obj)
%COMPUTECONTEXTALIGNMENT Compute the global context alignment from scratch.
%   alignment = computeContextAlignment(obj) maps the arbitrary per-particle
%   context labels onto a single stable set of global labels used by the
%   context-facing query methods. This is a reporting-only computation: the
%   result never feeds back into the particle filter / inference pipeline.
%
%   Pipeline:
%     1. selectModalContexts      - pick the modal cardinality K and the subset
%                                   of particles carrying exactly K contexts,
%                                   with their (uniform) weights.
%     2. initializeContextAlignment - seed the assignment and prototypes, warm-
%                                   starting from the previous alignment when it
%                                   is still compatible.
%     3. optimizeContextAlignment - alternate Hungarian assignment and prototype
%                                   recomputation until the labels stabilise.
%
%   The result is cached back into obj.alignment_seed so the next call can warm-
%   start from it. ensureContextAlignment.m is the state-version caching wrapper.
%   Returned fields: K, assignment, modal_particle_*, global_contexts,
%   converged, iterations, used_seed, cache_state_version, computed_at_trial.

    % Modal particle subset and per-particle weights.
    [Km, modalMask, modalIdx, weights] = obj.selectModalContexts();
    % Seed assignment/prototypes (warm-started from alignment_seed when valid).
    [assignment, prototypes, usedSeed] = obj.initializeContextAlignment(Km, modalIdx);
    % Iterate assignment <-> prototype recomputation to a fixed point.
    [assignment, prototypes, converged, iter] = obj.optimizeContextAlignment( ...
        Km, modalIdx, weights, assignment, prototypes);

    alignment = struct();
    alignment.K = Km;
    alignment.assignment = assignment;
    alignment.modal_particle_mask = modalMask;
    alignment.modal_particle_indices = modalIdx;
    alignment.modal_particle_weights = weights;
    alignment.global_contexts = prototypes;
    alignment.converged = converged;
    alignment.iterations = iter;
    alignment.used_seed = usedSeed;
    alignment.cache_state_version = obj.state_version;
    alignment.computed_at_trial = obj.trial;

    % Retain this alignment as the warm-start seed for the next computation.
    obj.alignment_seed = alignment;
end
