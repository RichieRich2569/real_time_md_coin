function cost = assignmentCostMatrixMD(obj, p, Km, prototypes, assignment, includeTransition, prepared)
%ASSIGNMENTCOSTMATRIXMD Local-to-global context assignment cost (MD model).
%
%   Multi-dimensional counterpart of assignmentCostMatrix.m. The cost of
%   matching a particle's local context to a global prototype combines:
%     - a symmetric (Jeffreys) divergence between the multivariate filtered
%       state distributions (mean vector + covariance matrix),
%     - a symmetric divergence between the augmented dynamics [A | d], treated
%       as a Gaussian with an isotropic reference covariance so contexts with
%       similar states but different dynamics are still separable, and
%     - categorical (cue and, optionally, transition) divergences.
%   These mirror the scalar cost terms, generalised to vectors/matrices.
    if nargin < 7 || isempty(prepared)
        prepared = obj.prepareAssignmentPrototypes(Km, prototypes);
    end

    N = obj.state_dim;
    cost = zeros(Km, Km);
    localStateCov = zeros(N, N, Km);
    localStateInv = zeros(N, N, Km);
    localTheta = zeros(N * (N + 1), Km);
    localCue = cell(1, Km);
    localTransition = cell(1, Km);

    for local = 1:Km
        covar = obj.regularizeCovariance(obj.D.state_filtered_cov(:, :, local, p));
        localStateCov(:, :, local) = covar;
        localStateInv(:, :, local) = obj.safeInverse(covar);
        localTheta(:, local) = reshape(obj.D.Theta(:, :, local, p), [], 1);
        localCue{local} = obj.localCueProbability(local, p);
        if includeTransition
            localTransition{local} = obj.globalTransitionRow(local, p, Km, assignment);
        end
    end

    for local = 1:Km
        sMean = obj.D.state_filtered_mean(:, local, p);
        sCov = localStateCov(:, :, local);
        sInv = localStateInv(:, :, local);
        thetaVec = localTheta(:, local);
        for globalIdx = 1:Km
            total = preparedGaussianJeffreys(sMean, sCov, sInv, ...
                prototypes.state_mean(:, globalIdx), ...
                prepared.state_cov(:, :, globalIdx), ...
                prepared.state_inv(:, :, globalIdx));

            thetaDelta = thetaVec - prepared.theta_vec(:, globalIdx);
            total = total + thetaDelta' * thetaDelta;

            total = total + obj.categoricalJeffreys(localCue{local}, prototypes.cue_prob(globalIdx, :));
            if includeTransition
                total = total + obj.categoricalJeffreys(localTransition{local}, prototypes.transition_prob(globalIdx, :));
            end
            cost(local, globalIdx) = total;
        end
    end
end

function d = preparedGaussianJeffreys(m1, s1, inv1, m2, s2, inv2)
    delta = m1(:) - m2(:);
    k = numel(delta);
    d = 0.5 .* (trace(inv2 * s1 + inv1 * s2) + delta' * (inv1 + inv2) * delta - 2 .* k);
    if ~isfinite(d)
        d = realmax;
    end
    d = max(d, 0);
end
