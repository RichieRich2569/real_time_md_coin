function k = kalman_gain_cstar2(obj, q)
%KALMAN_GAIN_CSTAR2 Kalman gain of the highest next-trial predicted-prob context.
%
%   k = kalman_gain_cstar2(obj, q) selects, in each particle, the context with
%   the highest predicted probability on the *next* trial (c*2, given the
%   optional upcoming cue q, default the pending cue) and reads off its current
%   scalar Kalman gain (state_var ./ state_feedback_var), then averages over
%   particles. Mirrors COIN's plot_Kalman_gain_given_cstar2: the selector uses
%   the one-step-ahead prediction while the gain is that of the current trial.
%
%   Defined for the scalar model only (state_dim == 1); see kalman_gain_cstar1.
    arguments
        obj (1, 1) RealTimeCOIN
        q double {mustBeScalarOrEmpty} = [];
    end
    mustBeScalarModel(obj, 'kalman_gain_cstar2');
    if isempty(q)
        q = obj.pending_q;
    end
    qLabel = peekCueLabel(obj, q);
    W = nextTrialContextWeights(obj, qLabel);
    P = obj.num_particles;
    gains = obj.D.state_var ./ obj.D.state_feedback_var;
    [~, idx] = max(W, [], 1);
    lin = sub2ind(size(gains), idx, 1:P);
    k = mean(gains(lin));
end

function mustBeScalarOrEmpty(x)
    if ~(isempty(x) || isscalar(x))
        error('RealTimeCOIN:InvalidCue', 'q must be empty or a scalar cue label.');
    end
end
