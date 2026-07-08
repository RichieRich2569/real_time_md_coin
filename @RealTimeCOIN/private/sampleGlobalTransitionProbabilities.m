function sampleGlobalTransitionProbabilities(obj)
%SAMPLEGLOBALTRANSITIONPROBABILITIES Resample the global transition distribution (sticky HDP-HMM).
%
%   sampleGlobalTransitionProbabilities(obj) updates, per particle, the
%   global ("beta") distribution over contexts that anchors the hierarchical
%   Dirichlet process prior on each context's transition row. It implements
%   the sticky HDP-HMM posterior of Fox et al. (2011), "A Sticky HDP-HMM with
%   Application to Speaker Diarization", Annals of Applied Statistics 5(2):
%     1. Draw Antoniak/CRP table counts m for the observed transition counts
%        n_context, with base measure alpha_context * beta plus the sticky
%        self-transition mass stickyKappa on the diagonal.
%     2. Remove the "override" self-transition tables attributed to the sticky
%        term via a Binomial(m_jj, rho / (rho + beta_j (1 - rho))) draw, so the
%        top-level counts reflect only the shared (non-sticky) mass.
%     3. Sum the tables across source contexts, add novelty mass gamma_context
%        at the first unused slot, and draw the refreshed beta from a
%        Dirichlet.
%
%   See sampleGlobalCueProbabilities.m for the non-sticky cue counterpart.

    Cmax = obj.max_contexts + 1;                % context slots incl. novel context
    P = obj.num_particles;
    % Sticky self-transition mass kappa = alpha_context * rho / (1 - rho)
    % (Fox et al.). Named stickyKappa to avoid shadowing the kappa() method.
    stickyKappa = obj.kappa();
    m = zeros(Cmax, Cmax, P);
    for p = 1:P
        % Antoniak base measure: shared alpha_context * beta plus the sticky
        % self-transition boost on the diagonal.
        base = obj.alpha_context .* obj.D.global_transition_probabilities(:,p)' + stickyKappa .* eye(Cmax);
        m(:,:,p) = obj.sample_num_tables(base, obj.D.n_context(:,:,p));
        if obj.rho_context > 0
            % Strip the sticky "override" tables from each diagonal so only the
            % shared mass propagates to the top-level Dirichlet (Fox et al.).
            for j = 1:Cmax
                if m(j,j,p) > 0
                    betaJ = obj.D.global_transition_probabilities(j,p);
                    prob = obj.rho_context ./ max(obj.rho_context + betaJ .* (1 - obj.rho_context), realmin);
                    m(j,j,p) = m(j,j,p) - obj.binomialSample(m(j,j,p), prob);
                end
            end
        end
        % Numerical floor: context 1 is always present, so guarantee it
        % contributes at least one table. Without this a particle whose
        % self-transition tables were all stripped could yield an all-zero
        % first Dirichlet parameter and a degenerate global distribution.
        if m(1,1,p) == 0
            m(1,1,p) = 1;
        end
        alpha = squeeze(sum(m(:,:,p), 1))';     % Dirichlet params from summed tables
        if obj.D.C(p) < obj.max_contexts
            alpha(obj.D.C(p)+1) = obj.gamma_context;  % novelty mass at first unused slot
            alpha(obj.D.C(p)+2:end) = 0;              % zero slots beyond the novel context
        else
            alpha(obj.D.C(p)+1:end) = 0;              % at cap: no novel slot
        end
        obj.D.global_transition_probabilities(:,p) = obj.dirichletSample(alpha(:));
    end
end
