function prototypes = updateGlobalContextsMD(obj, Km, modalIdx, weights, assignment)
%UPDATEGLOBALCONTEXTSMD Build global context prototypes for the MD model.
%
%   Multi-dimensional counterpart of updateGlobalContexts.m. For each globally
%   aligned context, accumulate weighted moments across the modal particles
%   that have a local context assigned to it. Scalar state/dynamics/bias
%   summaries become vectors/matrices:
%       state_mean  : N x Km           (mixture mean of the filtered state)
%       state_cov   : N x N x Km        (mixture covariance)
%       theta_mean  : N x (N+1) x Km    (mean augmented dynamics [A | d])
%       bias_mean   : N x Km
%   Cue and transition prototypes are identical in form to the scalar model.

    N = obj.state_dim;
    modalIdx = modalIdx(:)';
    weights = weights(:)';
    weights = weights ./ sum(weights);
    Qn = max(1, size(obj.D.local_cue_matrix, 2));

    prototypes = struct();
    prototypes.state_mean = zeros(N, Km);
    prototypes.state_cov = zeros(N, N, Km);
    prototypes.theta_mean = zeros(N, N+1, Km);
    prototypes.bias_mean = zeros(N, Km);
    prototypes.cue_prob = zeros(Km, Qn);
    prototypes.transition_prob = zeros(Km, Km + 1);

    for globalIdx = 1:Km
        totalWeight = 0;
        stateMean = zeros(N, 1);
        stateSecond = zeros(N, N);
        thetaMean = zeros(N, N+1);
        biasMean = zeros(N, 1);
        cueAccum = zeros(1, Qn);
        transitionAccum = zeros(1, Km + 1);

        for idx = 1:numel(modalIdx)
            p = modalIdx(idx);
            local = find(assignment(1:Km, p) == globalIdx, 1);
            if isempty(local)
                continue;
            end
            w = weights(idx);
            totalWeight = totalWeight + w;

            m = obj.D.state_filtered_mean(:, local, p);
            V = obj.D.state_filtered_cov(:, :, local, p);
            stateMean = stateMean + w .* m;
            stateSecond = stateSecond + w .* (V + (m * m'));

            thetaMean = thetaMean + w .* obj.D.Theta(:, :, local, p);
            biasMean = biasMean + w .* obj.D.bias(:, local, p);

            cueAccum = cueAccum + w .* obj.localCueProbability(local, p);
            transitionAccum = transitionAccum + w .* obj.globalTransitionRow(local, p, Km, assignment);
        end

        if totalWeight > 0
            stateMean = stateMean ./ totalWeight;
            stateSecond = stateSecond ./ totalWeight;
            thetaMean = thetaMean ./ totalWeight;
            biasMean = biasMean ./ totalWeight;
            cueAccum = cueAccum ./ totalWeight;
            transitionAccum = transitionAccum ./ totalWeight;
        end

        prototypes.state_mean(:, globalIdx) = stateMean;
        stateCov = stateSecond - (stateMean * stateMean');
        prototypes.state_cov(:, :, globalIdx) = obj.regularizeCovariance(stateCov);
        prototypes.theta_mean(:, :, globalIdx) = thetaMean;
        prototypes.bias_mean(:, globalIdx) = biasMean;
        prototypes.cue_prob(globalIdx, :) = obj.normalizeProbability(cueAccum);
        prototypes.transition_prob(globalIdx, :) = obj.normalizeProbability(transitionAccum);
    end
end
