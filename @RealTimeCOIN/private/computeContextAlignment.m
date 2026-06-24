function alignment = computeContextAlignment(obj)
    [Km, modalMask, modalIdx, weights] = obj.selectModalContexts();
    [assignment, prototypes, usedSeed] = obj.initializeContextAlignment(Km, modalIdx);
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

    obj.alignment_seed = alignment;
end
