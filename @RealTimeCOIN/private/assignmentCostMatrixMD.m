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
%   These mirror the scalar cost terms, generalised to vectors/matrices. The
%   transition term is included only when includeTransition is true (see
%   optimizeContextAlignment). `prepared` holds the per-prototype precompute from
%   prepareAssignmentPrototypes; it is rebuilt here if not supplied.
    if nargin < 7 || isempty(prepared)
        prepared = obj.prepareAssignmentPrototypes(Km, prototypes);
    end

    N = obj.state_dim;
    cost = zeros(Km, Km);
    % Precompute local-context terms once per row (state cov + its inverse,
    % vectorised dynamics, cue/transition rows) before the pairwise cost loop.
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
            % State: symmetric Gaussian-Jeffreys divergence (mean + covariance).
            total = preparedGaussianJeffreys(sMean, sCov, sInv, ...
                prototypes.state_mean(:, globalIdx), ...
                prepared.state_cov(:, :, globalIdx), ...
                prepared.state_inv(:, :, globalIdx));

            % Dynamics: squared Euclidean distance on vectorised [A | d]. This is
            % a Gaussian-Jeffreys with an isotropic identity reference covariance
            % (see header) - it ignores dynamics scale/uncertainty by design.
            thetaDelta = thetaVec - prepared.theta_vec(:, globalIdx);
            total = total + thetaDelta' * thetaDelta;

            % Cue (always) and transition (only past the first sweep) terms.
            total = total + obj.categoricalJeffreys(localCue{local}, prototypes.cue_prob(globalIdx, :));
            if includeTransition
                total = total + obj.categoricalJeffreys(localTransition{local}, ...
                    prototypes.transition_prob(globalIdx, :));
            end
            cost(local, globalIdx) = total;
        end
    end
end

function d = preparedGaussianJeffreys(m1, s1, inv1, m2, s2, inv2)
%PREPAREDGAUSSIANJEFFREYS Jeffreys divergence between two Gaussians (cached inv).
%   d = preparedGaussianJeffreys(m1, s1, inv1, m2, s2, inv2) evaluates the
%   symmetric (Jeffreys) divergence between N(m1, s1) and N(m2, s2), given their
%   precomputed inverse covariances inv1 = s1^-1 and inv2 = s2^-1:
%       d = 0.5 * ( tr(inv2*s1 + inv1*s2) + (m1-m2)'(inv1+inv2)(m1-m2) - 2k )
%   with k the dimension. Non-finite results are clamped to realmax and the
%   value is floored at 0. Duplicates gaussianJeffreysMulti but takes cached
%   inverses to avoid recomputing them inside the pairwise cost loop.

    delta = m1(:) - m2(:);
    k = numel(delta);
    d = 0.5 .* (trace(inv2 * s1 + inv1 * s2) + delta' * (inv1 + inv2) * delta - 2 .* k);
    if ~isfinite(d)
        d = realmax;
    end
    d = max(d, 0);
end
