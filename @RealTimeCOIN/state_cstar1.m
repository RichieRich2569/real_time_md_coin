function s = state_cstar1(obj)
%STATE_CSTAR1 Expected latent state of the highest-responsibility context.
%
%   s = state_cstar1(obj) returns the predictive latent-state mean of the
%   context with the highest responsibility (c*1) in each particle, averaged
%   over particles. This mirrors COIN's plot_state_given_cstar1: for each
%   particle the context maximising the current responsibilities is selected
%   and its state mean read off, then the selection is averaged across
%   particles. Scalar for state_dim == 1; an N-by-1 vector for the
%   multi-dimensional model. Reflects the model state as of the most recently
%   processed trial (obj.trial).
    arguments
        obj (1, 1) RealTimeCOIN
    end
    [~, idx] = max(obj.D.responsibilities, [], 1);
    s = selectContextStateMean(obj, idx);
end
