function updateBeliefAboutStates(obj, y)
    obj.D.state_filtered_mean = obj.D.state_mean;
    obj.D.state_filtered_var = obj.D.state_var;
    if isempty(y)
        return;
    end
    obsVar = obj.observationVariance();
    for p = 1:obj.num_particles
        c = obj.D.context(p);
        predVar = obj.D.state_var(c,p);
        totalVar = predVar + obsVar;
        if totalVar <= 0
            K = 0;
        else
            K = predVar ./ totalVar;
        end
        innovation = y - obj.D.state_feedback_mean(c,p);
        obj.D.state_filtered_mean(c,p) = obj.D.state_mean(c,p) + K .* innovation;
        obj.D.state_filtered_var(c,p) = max((1 - K) .* predVar, 0);
    end
end
