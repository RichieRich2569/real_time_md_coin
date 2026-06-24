function [assignment, prototypes, converged, iter] = optimizeContextAlignment( ...
        obj, Km, modalIdx, weights, assignment, prototypes)
    Cmax = obj.max_contexts + 1;
    oldAssignments = zeros(Cmax, obj.num_particles);
    converged = false;
    includeTransition = false;
    maxIterations = 20;

    for iter = 1:maxIterations
        prepared = obj.prepareAssignmentPrototypes(Km, prototypes);
        for idx = 1:numel(modalIdx)
            p = modalIdx(idx);
            cost = obj.assignmentCostMatrix(p, Km, prototypes, assignment, ...
                includeTransition, prepared);
            perm = obj.minAssignment(cost);
            assignment(:, p) = 0;
            assignment(1:Km, p) = perm(:);
            if Km < obj.max_contexts
                assignment(Km+1, p) = Km + 1;
            end
        end

        prototypes = obj.updateGlobalContexts(Km, modalIdx, weights, assignment);
        if iter > 1 && isequal(assignment(1:Km, modalIdx), oldAssignments(1:Km, modalIdx))
            converged = true;
            break;
        end
        oldAssignments = assignment;
        includeTransition = true;
    end
end
