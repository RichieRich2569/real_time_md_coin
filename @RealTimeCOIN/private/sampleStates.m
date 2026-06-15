function sampleStates(obj, y)
    qVar = obj.sigma_process_noise^2;
    obsVar = obj.observationVariance();
    Cmax = obj.max_contexts + 1;
    P = obj.num_particles;

    g = obj.D.retention .* obj.safeDivide(obj.D.previous_state_filtered_var, obj.D.state_var);
    smoothMean = obj.D.previous_state_filtered_mean + g .* (obj.D.state_filtered_mean - obj.D.state_mean);
    smoothVar = obj.D.previous_state_filtered_var + g.^2 .* (obj.D.state_filtered_var - obj.D.state_var);
    smoothVar = max(smoothVar, 0);
    obj.D.previous_x_dynamics = obj.sampleScalarNormal(smoothMean, smoothVar, [Cmax, P], -Inf, Inf);

    dynMean = obj.D.retention .* obj.D.previous_x_dynamics + obj.D.drift;
    if isempty(y)
        obj.D.x_dynamics = obj.sampleScalarNormal(dynMean, qVar, [Cmax, P], -Inf, Inf);
    else
        active = zeros(Cmax, P);
        idx = sub2ind([Cmax, P], obj.D.context, 1:P);
        active(idx) = 1;
        if qVar == 0
            postMean = dynMean;
            postVar = zeros(Cmax, P);
            if obsVar == 0
                postMean(idx) = y - obj.D.bias(idx);
            end
        elseif obsVar == 0
            postMean = dynMean;
            postVar = qVar * ones(Cmax, P);
            postMean(idx) = y - obj.D.bias(idx);
            postVar(idx) = 0;
        else
            postVar = 1 ./ (1 ./ qVar + active ./ obsVar);
            postMean = postVar .* (dynMean ./ qVar + active .* (y - obj.D.bias) ./ obsVar);
        end
        obj.D.x_dynamics = obj.sampleScalarNormal(postMean, postVar, [Cmax, P], -Inf, Inf);
    end

    activeIdx = sub2ind([Cmax, P], obj.D.context, 1:P);
    obj.D.x_bias = obj.D.x_dynamics(activeIdx);
    obj.D.i_observed = activeIdx;
end
