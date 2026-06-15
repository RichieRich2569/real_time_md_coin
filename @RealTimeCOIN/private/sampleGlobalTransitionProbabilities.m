function sampleGlobalTransitionProbabilities(obj)
    Cmax = obj.max_contexts + 1;
    P = obj.num_particles;
    kappa = obj.kappa();
    m = zeros(Cmax, Cmax, P);
    for p = 1:P
        base = obj.alpha_context .* obj.D.global_transition_probabilities(:,p)' + kappa .* eye(Cmax);
        m(:,:,p) = obj.sample_num_tables(base, obj.D.n_context(:,:,p));
        if obj.rho_context > 0
            for j = 1:Cmax
                if m(j,j,p) > 0
                    betaJ = obj.D.global_transition_probabilities(j,p);
                    prob = obj.rho_context ./ max(obj.rho_context + betaJ .* (1 - obj.rho_context), realmin);
                    m(j,j,p) = m(j,j,p) - obj.binomialSample(m(j,j,p), prob);
                end
            end
        end
        if m(1,1,p) == 0
            m(1,1,p) = 1;
        end
        alpha = squeeze(sum(m(:,:,p), 1))';
        if obj.D.C(p) < obj.max_contexts
            alpha(obj.D.C(p)+1) = obj.gamma_context;
            alpha(obj.D.C(p)+2:end) = 0;
        else
            alpha(obj.D.C(p)+1:end) = 0;
        end
        obj.D.global_transition_probabilities(:,p) = obj.dirichletSample(alpha(:));
    end
end
