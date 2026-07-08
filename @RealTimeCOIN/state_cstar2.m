function s = state_cstar2(obj, q)
%STATE_CSTAR2 Expected state of the highest next-trial predicted-prob context.
%
%   s = state_cstar2(obj, q) returns the current predictive latent-state mean of
%   the context that has the highest predicted probability on the *next* trial
%   (c*2), averaged over particles. The upcoming cue q (a raw cue value, default
%   the pending cue) conditions the one-step-ahead context prediction, exactly
%   as COIN's plot_state_given_cstar2, which selects the argmax of the next
%   trial's predicted probabilities but reads the state mean of the current
%   trial. Scalar for state_dim == 1; an N-by-1 vector otherwise.
%
%   Unlike state_cstar3 (which uses the current trial's predicted
%   probabilities), c*2 requires a one-step look-ahead; it is computed here from
%   the sampled transition (and cue) matrices without mutating the model state.
    arguments
        obj (1, 1) RealTimeCOIN
        q double {mustBeScalarOrEmpty} = [];
    end
    if isempty(q)
        q = obj.pending_q;
    end
    qLabel = peekCueLabel(obj, q);
    W = nextTrialContextWeights(obj, qLabel);
    [~, idx] = max(W, [], 1);
    s = selectContextStateMean(obj, idx);
end
