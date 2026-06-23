function cost = assignmentCostMatrixMD(obj, p, Km, prototypes, assignment, includeTransition)
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

    N = obj.state_dim;
    cost = zeros(Km, Km);
    thetaRefCov = eye(N * (N + 1));   % reference scale for the dynamics term

    for local = 1:Km
        sMean = obj.D.state_filtered_mean(:, local, p);
        sCov = obj.D.state_filtered_cov(:, :, local, p);
        thetaVec = reshape(obj.D.Theta(:, :, local, p), [], 1);
        cueProb = obj.localCueProbability(local, p);
        if includeTransition
            transitionProb = obj.globalTransitionRow(local, p, Km, assignment);
        end

        for globalIdx = 1:Km
            total = obj.gaussianJeffreysMulti(sMean, sCov, ...
                prototypes.state_mean(:, globalIdx), prototypes.state_cov(:, :, globalIdx));

            protoTheta = reshape(prototypes.theta_mean(:, :, globalIdx), [], 1);
            total = total + obj.gaussianJeffreysMulti(thetaVec, thetaRefCov, protoTheta, thetaRefCov);

            total = total + obj.categoricalJeffreys(cueProb, prototypes.cue_prob(globalIdx, :));
            if includeTransition
                total = total + obj.categoricalJeffreys(transitionProb, prototypes.transition_prob(globalIdx, :));
            end
            cost(local, globalIdx) = total;
        end
    end
end
