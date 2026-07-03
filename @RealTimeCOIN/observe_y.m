function observe_y(obj, y)
    % y must be empty or a numeric value with dim = obj.state_dim.
    arguments
        obj (1, 1) RealTimeCOIN
        y (:, 1) double {mustBeNumeric} = []
    end
    if isempty(y) || (isnumeric(y) && isscalar(y) && anynan(y))
        % TODO: Having one dimension being NaN means inference has to be done with other non-nan values.
        y_val = [];
    else
        mustBeStateDim(obj.state_dim, y);
        y_val = y;
    end

    q_val = consumePendingCue(obj);

    if obj.state_dim > 1
        % Multi-dimensional pipeline. The context-inference step
        % (predictContext) is dimension-agnostic and reused; the state and
        % parameter steps have dedicated *MD implementations.
        predictContext(obj, q_val);
        predictStatesMD(obj);
        predictStateFeedbackMD(obj);
        resampleParticlesMD(obj, y_val, q_val);
        sampleContextMD(obj, q_val);
        updateBeliefAboutStatesMD(obj, y_val);
        sampleStatesMD(obj, y_val);
        updateSufficientStatisticsMD(obj, y_val, q_val);
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

    obj.trial = obj.trial + 1;
    invalidateContextAlignment(obj);
end

function mustBeStateDim(dim, y)
    % Test for equal size
    if ~isequal(dim,numel(y))
        eid = 'Size:notStateDim';
        msg = 'Size of observed y must match the state dimension.';
        error(eid,msg)
    end
end