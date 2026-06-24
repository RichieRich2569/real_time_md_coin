function cost = assignmentCostMatrix(obj, p, Km, prototypes, assignment, includeTransition, prepared)
    if nargin < 7
        prepared = [];
    end
    if obj.state_dim > 1
        cost = obj.assignmentCostMatrixMD(p, Km, prototypes, assignment, includeTransition, prepared);
        return;
    end
    cost = zeros(Km, Km);
    for local = 1:Km
        [dynMean, dynCovar] = obj.localDynamicsDistribution(local, p);
        cueProb = obj.localCueProbability(local, p);
        if includeTransition
            transitionProb = obj.globalTransitionRow(local, p, Km, assignment);
        end
        for globalIdx = 1:Km
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
