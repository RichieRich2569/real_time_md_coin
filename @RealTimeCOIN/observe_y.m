function observe_y(obj, y)
    if isempty(y) || (isnumeric(y) && isscalar(y) && isnan(y))
        y_val = [];
    else
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
