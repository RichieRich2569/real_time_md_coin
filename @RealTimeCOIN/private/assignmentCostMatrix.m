function cost = assignmentCostMatrix(obj, p, Km, prototypes, assignment, includeTransition, prepared)
%ASSIGNMENTCOSTMATRIX Local-to-global context assignment cost (scalar model).
%   cost = assignmentCostMatrix(obj, p, Km, prototypes, assignment,
%   includeTransition, prepared) returns the Km-by-Km cost of matching each local
%   context (row) of particle `p` to each global prototype (column). The
%   optimizer minimises the total matching cost to relabel the particle.
%
%   cost(local, globalIdx) sums symmetric (Jeffreys) divergences between the
%   local context and the global prototype over several parameter blocks:
%     - state:     Gaussian-Jeffreys on the filtered state (mean, variance).
%     - dynamics:  bivariate Gaussian-Jeffreys on [a; d] (mean + covariance).
%     - bias:      Gaussian-Jeffreys on the observation bias, only when
%                  infer_bias is set.
%     - cue:       categorical Jeffreys between cue probability rows.
%     - transition: categorical Jeffreys between transition rows, added only
%                  when includeTransition is true (see optimizeContextAlignment).
%   For state_dim > 1 the computation is delegated to assignmentCostMatrixMD.
%   `prepared` is the optional prototype precompute (unused on the scalar path).

    if nargin < 7
        prepared = [];
    end
    if obj.state_dim > 1
        cost = obj.assignmentCostMatrixMD(p, Km, prototypes, assignment, includeTransition, prepared);
        return;
    end
    cost = zeros(Km, Km);
    for local = 1:Km
        % Local-context distributions, computed once per row.
        [dynMean, dynCovar] = obj.localDynamicsDistribution(local, p);
        cueProb = obj.localCueProbability(local, p);
        if includeTransition
            transitionProb = obj.globalTransitionRow(local, p, Km, assignment);
        end
        for globalIdx = 1:Km
            % Accumulate the per-block symmetric divergences into `total`.
            total = obj.gaussianJeffreys(obj.D.state_filtered_mean(local,p), ...
                obj.D.state_filtered_var(local,p), ...
                prototypes.state_mean(globalIdx), prototypes.state_var(globalIdx));
            total = total + obj.gaussianJeffreysMulti(dynMean, dynCovar, ...
                prototypes.dynamics_mean(:,globalIdx), prototypes.dynamics_covar(:,:,globalIdx));
            if obj.infer_bias
                [biasMean, biasVar] = obj.localBiasDistribution(local, p);
                total = total + obj.gaussianJeffreys(biasMean, biasVar, ...
                    prototypes.bias_mean(globalIdx), prototypes.bias_var(globalIdx));
            end
            total = total + obj.categoricalJeffreys(cueProb, prototypes.cue_prob(globalIdx,:));
            if includeTransition
                total = total + obj.categoricalJeffreys(transitionProb, prototypes.transition_prob(globalIdx,:));
            end
            cost(local,globalIdx) = total;
        end
    end
end
