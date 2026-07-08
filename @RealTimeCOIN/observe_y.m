function observe_y(obj, y)
%OBSERVE_Y Process one trial's feedback and advance the inference pipeline.
%   observe_y(obj, y) consumes the cue staged by observe_q, runs the full COIN
%   particle-filter update for the trial using the state feedback y, advances
%   the trial counter and invalidates the cached context alignment. Call it
%   exactly once per trial (after any observe_q).
%
%   y is the observed state feedback: a column vector with obj.state_dim
%   elements (a scalar when state_dim == 1). Pass [] or NaN for a missing
%   observation - the trial still runs (states are predicted, not corrected).
%   In the multi-dimensional case individual NaN entries of y mark
%   partially-observed dimensions via an observation mask.
%
%   Dispatch: state_dim == 1 runs the original scalar pipeline verbatim, which
%   is the byte-for-byte regression baseline against COIN.m; state_dim > 1 runs
%   the *MD variants. The per-step order is identical in both branches.
%
%   See also OBSERVE_Q, SET_STATIONARY.
    arguments
        obj (1, 1) RealTimeCOIN
        y (:, 1) double {mustBeNumeric} = []
    end
    % Normalise y into y_val plus, for the MD path, a per-dimension mask of the
    % entries that are actually observed (non-NaN).
    obs_mask = [];
    if obj.state_dim > 1
        if isempty(y)
            y_val = [];
            obs_mask = false(obj.state_dim, 1);
        else
            mustBeStateDim(obj.state_dim, y);
            y_val = y(:);
            obs_mask = ~isnan(y_val);
        end
    elseif isempty(y) || (isnumeric(y) && isscalar(y) && anynan(y))
        y_val = [];
    else
        mustBeStateDim(obj.state_dim, y);
        y_val = y;
    end

    % Resolve the cue staged by observe_q into a context-cue index for this
    % trial (also clears pending_q).
    q_val = consumePendingCue(obj);

    if obj.state_dim > 1
        % Multi-dimensional pipeline. The context-inference step
        % (predictContext) is dimension-agnostic and reused; the state and
        % parameter steps have dedicated *MD implementations.
        predictContext(obj, q_val);
        predictStatesMD(obj);
        predictStateFeedbackMD(obj);
        resampleParticlesMD(obj, y_val, q_val, obs_mask);
        sampleContextMD(obj, q_val);
        updateBeliefAboutStatesMD(obj, y_val, obs_mask);
        sampleStatesMD(obj, y_val, obs_mask);
        updateSufficientStatisticsMD(obj, y_val, q_val, obs_mask);
        sampleParametersMD(obj);
    else
        % Original scalar pipeline, unchanged.
        predictContext(obj, q_val);
        predictStates(obj);
        predictStateFeedback(obj);
        resampleParticles(obj, y_val, q_val);
        sampleContext(obj, q_val);
        updateBeliefAboutStates(obj, y_val);
        sampleStates(obj, y_val);
        updateSufficientStatistics(obj, y_val, q_val);
        sampleParameters(obj);
    end

    % Advance the trial and drop the cached context alignment, which is now
    % stale relative to the updated particle set.
    obj.trial = obj.trial + 1;
    invalidateContextAlignment(obj);
end

function mustBeStateDim(dim, y)
%MUSTBESTATEDIM Validate that the feedback y has exactly dim elements.
    if ~isequal(dim,numel(y))
        eid = 'Size:notStateDim';
        msg = 'Size of observed y must match the state dimension.';
        error(eid,msg)
    end
end
