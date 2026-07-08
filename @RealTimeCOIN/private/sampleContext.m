function sampleContext(obj, q)
%SAMPLECONTEXT Sample the active context and instantiate novel contexts.
%
%   sampleContext(obj, q) draws, for each particle, the context that
%   generated the current trial from its categorical responsibilities, then
%   grows the context set when the "novel context" slot is selected. q is
%   the current cue label, forwarded to instantiateCueIfNeeded so the cue
%   likelihood is extended in step with a new context.
%
%   The novel context is the last responsibility entry. Drawing it (subject
%   to the max_contexts cap) increments the particle's context count C and
%   splits the novel context's stick-breaking mass via a Beta(1, gamma) draw
%   b: the retained proportion b stays with the new context and (1 - b)
%   passes to the next (still novel) slot, i.e. the DP stick-breaking growth
%   of the global transition distribution. A freshly instantiated context's
%   latent state is seeded from its stationary (long-run) mean/variance.
%
%   See sampleContextMD.m for the multi-dimensional counterpart.

    P = obj.num_particles;
    oldC = obj.D.C;                              % context counts before this draw
    obj.D.previous_context = obj.D.context;      % remember source context for transition stats

    % Inverse-CDF categorical draw of the context from the responsibilities.
    cumResp = cumsum(obj.D.responsibilities, 1);
    r = rand(1, P);
    newContext = sum(r > cumResp, 1) + 1;
    for p = 1:P
        % A draw beyond the current count means the novel context was chosen.
        if newContext(p) > obj.D.C(p)
            if obj.D.C(p) < obj.max_contexts
                obj.D.C(p) = obj.D.C(p) + 1;     % instantiate the new context
                newContext(p) = obj.D.C(p);
            else
                newContext(p) = obj.D.C(p);      % at cap: fold back onto the last context
            end
        end
    end
    obj.D.context = newContext;

    % Particles that gained a context (and are below the cap) split the novel
    % context's stick-breaking mass.
    pNew = find(obj.D.C > oldC & obj.D.C < obj.max_contexts);
    if ~isempty(pNew)
        b = obj.betaSample(ones(1, numel(pNew)), obj.gamma_context * ones(1, numel(pNew)));
        for k = 1:numel(pNew)
            p = pNew(k);
            c = obj.D.C(p);
            mass = obj.D.global_transition_probabilities(c, p);
            % Stick-breaking: keep proportion b for the new context, pass the
            % remainder (1 - b) to the next (novel) slot.
            obj.D.global_transition_probabilities(c+1, p) = mass .* (1 - b(k));
            obj.D.global_transition_probabilities(c, p) = mass .* b(k);
            % Seed the new context's state at its stationary distribution.
            obj.D.state_filtered_mean(c,p) = obj.stationaryStateMean(obj.D.retention(c,p), obj.D.drift(c,p));
            obj.D.state_filtered_var(c,p) = obj.stationaryStateVar(obj.D.retention(c,p));
        end
    end
    obj.instantiateCueIfNeeded(q);
end
