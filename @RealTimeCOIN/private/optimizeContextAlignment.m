function [assignment, prototypes, converged, iter] = optimizeContextAlignment( ...
        obj, Km, modalIdx, weights, assignment, prototypes)
%OPTIMIZECONTEXTALIGNMENT Refine the global alignment by block coordinate ascent.
%   [assignment, prototypes, converged, iter] = optimizeContextAlignment(obj, Km,
%   modalIdx, weights, assignment, prototypes) alternates two steps until the
%   per-particle labelling stops changing:
%     1. Assignment step - for each modal particle, build the local-to-global
%        cost matrix (assignmentCostMatrix) and solve the resulting square
%        min-cost matching (linearAssignment) to relabel that particle's contexts.
%     2. Prototype step - recompute the weighted global context prototypes
%        (updateGlobalContexts) from the freshly relabelled particles.
%   This is a hard-assignment EM-style fixed-point iteration; it is a reporting-
%   only relabelling and does not alter inference.
%
%   Loop controls (values preserved as-is; do not change without re-baselining
%   the alignment output against test_global_alignment.m):
%     maxIterations = 20  - cap on assignment/prototype sweeps; convergence
%                           normally occurs in a few iterations, so this is a
%                           safety bound, not the expected stopping point.
%     includeTransition   - toggle for whether the global transition-row term
%                           contributes to the assignment cost. It starts false
%                           on the first sweep (the transition prototypes are not
%                           yet meaningful before any relabelling) and is turned
%                           on from the second sweep onward, once prototypes have
%                           been recomputed at least once.
%
%   converged is true if the modal-particle assignment block was unchanged
%   between successive sweeps; iter reports the sweep at which the loop stopped.

    Cmax = obj.max_contexts + 1;
    oldAssignments = zeros(Cmax, obj.num_particles);
    converged = false;
    includeTransition = false;   % transition term disabled on the first sweep
    maxIterations = 20;          % safety cap on assignment/prototype sweeps

    for iter = 1:maxIterations
        % Precompute prototype quantities shared across all particles this sweep.
        prepared = obj.prepareAssignmentPrototypes(Km, prototypes);

        % --- Assignment step: relabel each modal particle by min-cost matching.
        for idx = 1:numel(modalIdx)
            p = modalIdx(idx);
            cost = obj.assignmentCostMatrix(p, Km, prototypes, assignment, ...
                includeTransition, prepared);
            perm = linearAssignment(cost);
            assignment(:, p) = 0;
            assignment(1:Km, p) = perm(:);
            if Km < obj.max_contexts
                assignment(Km+1, p) = Km + 1;   % label the novel-context slot
            end
        end

        % --- Prototype step: rebuild global prototypes from new labels.
        prototypes = obj.updateGlobalContexts(Km, modalIdx, weights, assignment);

        % Converged once the modal-particle assignment stops changing.
        if iter > 1 && isequal(assignment(1:Km, modalIdx), oldAssignments(1:Km, modalIdx))
            converged = true;
            break;
        end
        oldAssignments = assignment;
        includeTransition = true;   % enable transition term from the 2nd sweep
    end
end
