function alignment = computeContextAlignment(obj)
    Cmax = obj.max_contexts + 1;
    P = obj.num_particles;
    cards = obj.D.C(:)';
    Km = obj.modalCardinality(cards);
    modalMask = cards == Km;
    modalIdx = find(modalMask);
    if isempty(modalIdx)
        modalIdx = 1:P;
        modalMask = true(1, P);
        Km = min(max(cards), obj.max_contexts);
    end

    nModal = numel(modalIdx);
    weights = ones(1, nModal) ./ nModal;
    assignment = zeros(Cmax, P);
    anchor = modalIdx(1);
    assignment(1:Km, anchor) = 1:Km;
    if Km < obj.max_contexts
        assignment(Km+1, anchor) = Km + 1;
    end

    prototypes = obj.updateGlobalContexts(Km, anchor, 1, assignment);
    oldAssignments = zeros(Cmax, P);
    converged = false;
    includeTransition = false;
    maxIterations = 20;

    for iter = 1:maxIterations
        for idx = 1:nModal
            p = modalIdx(idx);
            cost = obj.assignmentCostMatrix(p, Km, prototypes, assignment, includeTransition);
            perm = obj.minAssignment(cost);
            assignment(:,p) = 0;
            assignment(1:Km,p) = perm(:);
            if Km < obj.max_contexts
                assignment(Km+1,p) = Km + 1;
            end
        end

        prototypes = obj.updateGlobalContexts(Km, modalIdx, weights, assignment);
        if iter > 1 && isequal(assignment(1:Km,modalIdx), oldAssignments(1:Km,modalIdx))
            converged = true;
            break;
        end
        oldAssignments = assignment;
        includeTransition = true;
    end

    alignment = struct();
    alignment.K = Km;
    alignment.assignment = assignment;
    alignment.modal_particle_mask = modalMask;
    alignment.modal_particle_indices = modalIdx;
    alignment.modal_particle_weights = weights;
    alignment.global_contexts = prototypes;
    alignment.converged = converged;
    alignment.iterations = iter;
    alignment.cache_state_version = obj.state_version;
    alignment.computed_at_trial = obj.trial;
end
