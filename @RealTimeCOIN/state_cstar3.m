function s = state_cstar3(obj)
%STATE_CSTAR3 Expected latent state of the highest predicted-probability context.
%
%   s = state_cstar3(obj) returns the predictive latent-state mean of the
%   context with the highest predicted probability (c*3) in each particle,
%   averaged over particles. Mirrors COIN's plot_state_given_cstar3, which
%   selects the argmax of the current trial's predicted probabilities. Scalar
%   for state_dim == 1; an N-by-1 vector for the multi-dimensional model.
%
%   c*3 differs from c*2 (see state_cstar2) only in which trial the predicted
%   probabilities are attributed to: c*3 uses the predicted probabilities of
%   the current model state (as of obj.trial), whereas c*2 corresponds to the
%   prediction carried forward to the next trial.
    arguments
        obj (1, 1) RealTimeCOIN
    end
    [~, idx] = max(obj.D.predicted_probabilities, [], 1);
    s = selectContextStateMean(obj, idx);
end
