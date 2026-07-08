function updateLocalTransitionMatrix(obj)
%UPDATELOCALTRANSITIONMATRIX Rebuild per-particle context transition matrices.
%   updateLocalTransitionMatrix(obj) computes obj.D.local_transition_matrix, a
%   Cmax-by-Cmax-by-P tensor whose (r, :, p) row is the posterior-mean
%   distribution over the next context given current context r, for particle p.
%   Each row is the expectation of a Dirichlet whose parameters are the
%   observed transition counts plus the sticky HDP-HMM prior:
%       raw(r, c) = n_context(r, c)                      (observed counts)
%                 + alpha_context * beta(c)              (global HDP base measure)
%                 + kappaSelf * (r == c)                 (self-transition stickiness)
%   normalised over c. Rows/columns for contexts that do not yet exist (beyond
%   the instantiated count plus one novel slot) are zeroed out before
%   normalisation. Scalar and MD models share this dimension-agnostic routine.

    Cmax = obj.max_contexts + 1;
    P = obj.num_particles;
    % Self-transition concentration kappa = alpha_context*rho/(1-rho) (sticky
    % HDP-HMM; Fox et al. 2011 / COIN paper). Local name avoids shadowing the
    % kappa() method that computes it.
    kappaSelf = obj.kappa();
    T = zeros(Cmax, Cmax, P);
    for p = 1:P
        % Unnormalised Dirichlet parameters: counts + HDP base measure + sticky
        % diagonal (kappaSelf added to self-transitions via the identity term).
        raw = obj.nContextSlice(p) + obj.alpha_context .* obj.D.global_transition_probabilities(:,p)' ...
            + kappaSelf .* eye(Cmax);
        % A context is "valid" if it is already instantiated (1..C) or is the
        % single next novel slot (C+1, unless the cap max_contexts is reached).
        valid = false(1, Cmax);
        valid(1:obj.D.C(p)) = true;
        if obj.D.C(p) < obj.max_contexts
            valid(obj.D.C(p)+1) = true;
        end
        % Zero out rows and columns of contexts that cannot occur yet.
        raw(:, ~valid) = 0;
        raw(~valid, :) = 0;
        % Row-normalise each valid row to a proper transition distribution;
        % all-zero rows (invalid source contexts) are left as zeros.
        rowSums = sum(raw, 2);
        for r = 1:Cmax
            if rowSums(r) > 0
                T(r,:,p) = raw(r,:) ./ rowSums(r);
            end
        end
    end
    obj.D.local_transition_matrix = T;
end
