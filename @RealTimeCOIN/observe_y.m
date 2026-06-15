function observe_y(obj, y)
    if isempty(y) || (isnumeric(y) && isscalar(y) && isnan(y))
        y_val = [];
    else
        y_val = y;
    end

    q_val = consumePendingCue(obj);
    predictContext(obj, q_val);
    predictStates(obj);
    predictStateFeedback(obj);
    resampleParticles(obj, y_val, q_val);
    sampleContext(obj, q_val);
    updateBeliefAboutStates(obj, y_val);
    sampleStates(obj, y_val);
    updateSufficientStatistics(obj, y_val, q_val);
    sampleParameters(obj);

    obj.trial = obj.trial + 1;
    invalidateContextAlignment(obj);
end
