function sampleContext(obj, q)
    P = obj.num_particles;
    oldC = obj.D.C;
    obj.D.previous_context = obj.D.context;
    cumResp = cumsum(obj.D.responsibilities, 1);
    r = rand(1, P);
    newContext = sum(r > cumResp, 1) + 1;
    for p = 1:P
        if newContext(p) > obj.D.C(p)
            if obj.D.C(p) < obj.max_contexts
                obj.D.C(p) = obj.D.C(p) + 1;
                newContext(p) = obj.D.C(p);
            else
                newContext(p) = obj.D.C(p);
            end
        end
    end
    obj.D.context = newContext;

    pNew = find(obj.D.C > oldC & obj.D.C < obj.max_contexts);
    if ~isempty(pNew)
        b = obj.betaSample(ones(1, numel(pNew)), obj.gamma_context * ones(1, numel(pNew)));
        for k = 1:numel(pNew)
            p = pNew(k);
            c = obj.D.C(p);
            mass = obj.D.global_transition_probabilities(c, p);
            obj.D.global_transition_probabilities(c+1, p) = mass .* (1 - b(k));
            obj.D.global_transition_probabilities(c, p) = mass .* b(k);
            obj.D.state_filtered_mean(c,p) = obj.stationaryStateMean(obj.D.retention(c,p), obj.D.drift(c,p));
            obj.D.state_filtered_var(c,p) = obj.stationaryStateVar(obj.D.retention(c,p));
        end
    end
    obj.instantiateCueIfNeeded(q);
end
