function sampleContextMD(obj, q)
%SAMPLECONTEXTMD Sample the active context and instantiate novel contexts.
%
%   Multi-dimensional counterpart of sampleContext.m. The context-sampling
%   and stick-breaking logic are identical to the scalar model (and operate
%   on the same shaped fields); the only difference is that a newly
%   instantiated context's latent state is seeded to the MD stationary
%   distribution (vector mean and N-by-N covariance) rather than the scalar
%   stationary mean/variance.

    N = obj.state_dim;
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
        Q = obj.processNoiseCov();
        b = obj.betaSample(ones(1, numel(pNew)), obj.gamma_context * ones(1, numel(pNew)));
        for k = 1:numel(pNew)
            p = pNew(k);
            c = obj.D.C(p);
            mass = obj.D.global_transition_probabilities(c, p);
            obj.D.global_transition_probabilities(c+1, p) = mass .* (1 - b(k));
            obj.D.global_transition_probabilities(c, p) = mass .* b(k);

            A = obj.D.Theta(:, 1:N, c, p);
            d = obj.D.Theta(:, N+1, c, p);
            obj.D.state_filtered_mean(:, c, p) = obj.stationaryStateMeanMD(A, d);
            obj.D.state_filtered_cov(:, :, c, p) = obj.stationaryStateCovMD(A, Q);
        end
    end
    obj.instantiateCueIfNeeded(q);
end
