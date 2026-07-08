function updateBeliefAboutStates(obj, y)
%UPDATEBELIEFABOUTSTATES Scalar Kalman measurement update for the active context.
%   updateBeliefAboutStates(obj, y) applies the scalar Kalman update to the
%   sampled (active) context of each particle given feedback y. With
%   observation model y = s + b + v, v ~ N(0, obsVar):
%
%       K       = predVar / (predVar + obsVar)      (Kalman gain)
%       s_post  = s_pred + K * (y - (s_pred + b))
%       P_post  = (1 - K) * predVar
%
%   Only the active context is updated; every inactive context keeps its
%   prediction (state_filtered_* is seeded from state_* up front). This is the
%   scalar counterpart of updateBeliefAboutStatesMD, to which it reduces exactly
%   at N == 1.
%
%   Missing-observation asymmetry vs. the MD path: here y is a scalar, so the
%   only "missing" case is an empty y, which returns with the posterior equal to
%   the prediction. The MD path additionally supports partial observation via an
%   obsMask (some feedback dimensions present, others NaN); the scalar path has
%   no per-dimension mask because there is only one dimension.
%
%   Writes obj.D.state_filtered_mean and obj.D.state_filtered_var.

    % Inactive contexts (and, on a missing trial, all contexts): posterior ==
    % prediction. Seed the filtered fields from the predictions, then overwrite
    % the active context below.
    obj.D.state_filtered_mean = obj.D.state_mean;
    obj.D.state_filtered_var = obj.D.state_var;
    if isempty(y)
        return;                                     % missing observation: no update
    end
    obsVar = obj.observationVariance();
    for p = 1:obj.num_particles
        c = obj.D.context(p);                        % active context for this particle
        predVar = obj.D.state_var(c,p);
        totalVar = predVar + obsVar;
        if totalVar <= 0
            K = 0;                                   % degenerate variance: trust the prior
        else
            K = predVar ./ totalVar;
        end
        innovation = y - obj.D.state_feedback_mean(c,p);
        obj.D.state_filtered_mean(c,p) = obj.D.state_mean(c,p) + K .* innovation;
        obj.D.state_filtered_var(c,p) = max((1 - K) .* predVar, 0);   % guard tiny negatives
    end
end
