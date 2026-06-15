function predictStates(obj)
    qv = obj.sigma_process_noise^2;
    obj.D.state_mean = obj.D.retention .* obj.D.state_filtered_mean + obj.D.drift;
    obj.D.state_var = obj.D.retention.^2 .* obj.D.state_filtered_var + qv;

    for p = 1:obj.num_particles
        novel = min(obj.D.C(p) + 1, obj.max_contexts + 1);
        if obj.D.C(p) < obj.max_contexts
            obj.D.state_mean(novel,p) = obj.stationaryStateMean(obj.D.retention(novel,p), obj.D.drift(novel,p));
            obj.D.state_var(novel,p) = obj.stationaryStateVar(obj.D.retention(novel,p));
        end
    end
    obj.D.state_var = max(obj.D.state_var, 0);
end
