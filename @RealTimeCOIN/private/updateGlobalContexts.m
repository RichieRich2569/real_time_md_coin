function prototypes = updateGlobalContexts(obj, Km, modalIdx, weights, assignment)
    modalIdx = modalIdx(:)';
    weights = weights(:)';
    weights = weights ./ sum(weights);
    Qn = max(1, size(obj.D.local_cue_matrix, 2));
    prototypes = struct();
    prototypes.state_mean = zeros(1, Km);
    prototypes.state_var = zeros(1, Km);
    prototypes.dynamics_mean = zeros(2, Km);
    prototypes.dynamics_covar = zeros(2, 2, Km);
    prototypes.bias_mean = zeros(1, Km);
    prototypes.bias_var = zeros(1, Km);
    prototypes.cue_prob = zeros(Km, Qn);
    prototypes.transition_prob = zeros(Km, Km + 1);

    for globalIdx = 1:Km
        totalWeight = 0;
        stateMean = 0;
        stateSecond = 0;
        dynMean = zeros(2,1);
        dynSecond = zeros(2,2);
        biasMean = 0;
        biasSecond = 0;
        cueAccum = zeros(1, Qn);
        transitionAccum = zeros(1, Km + 1);

        for idx = 1:numel(modalIdx)
            p = modalIdx(idx);
            local = find(assignment(1:Km,p) == globalIdx, 1);
            if isempty(local)
                continue;
            end
            w = weights(idx);
            totalWeight = totalWeight + w;

            m = obj.D.state_filtered_mean(local,p);
            v = max(obj.D.state_filtered_var(local,p), 0);
            stateMean = stateMean + w .* m;
            stateSecond = stateSecond + w .* (v + m.^2);

            [dm, dc] = obj.localDynamicsDistribution(local, p);
            dynMean = dynMean + w .* dm;
            dynSecond = dynSecond + w .* (dc + dm * dm');

            [bm, bv] = obj.localBiasDistribution(local, p);
            biasMean = biasMean + w .* bm;
            biasSecond = biasSecond + w .* (bv + bm.^2);

            cueAccum = cueAccum + w .* obj.localCueProbability(local, p);
            transitionAccum = transitionAccum + w .* obj.globalTransitionRow(local, p, Km, assignment);
        end

        if totalWeight > 0
            stateMean = stateMean ./ totalWeight;
            stateSecond = stateSecond ./ totalWeight;
            dynMean = dynMean ./ totalWeight;
            dynSecond = dynSecond ./ totalWeight;
            biasMean = biasMean ./ totalWeight;
            biasSecond = biasSecond ./ totalWeight;
            cueAccum = cueAccum ./ totalWeight;
            transitionAccum = transitionAccum ./ totalWeight;
        end

        prototypes.state_mean(globalIdx) = stateMean;
        prototypes.state_var(globalIdx) = max(stateSecond - stateMean.^2, 0);
        prototypes.dynamics_mean(:,globalIdx) = dynMean;
        dynCovar = dynSecond - dynMean * dynMean';
        prototypes.dynamics_covar(:,:,globalIdx) = obj.regularizeCovariance(dynCovar);
        prototypes.bias_mean(globalIdx) = biasMean;
        prototypes.bias_var(globalIdx) = max(biasSecond - biasMean.^2, 0);
        prototypes.cue_prob(globalIdx,:) = obj.normalizeProbability(cueAccum);
        prototypes.transition_prob(globalIdx,:) = obj.normalizeProbability(transitionAccum);
    end
end
