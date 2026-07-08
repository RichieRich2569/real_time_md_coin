function predictContext(obj, q)
%PREDICTCONTEXT Predict the per-particle context distribution for this trial.
%   predictContext(obj, q) forms the prior over context assignments for the
%   upcoming observation and, when a cue q is present, tilts it by the cue
%   likelihood p(q|c). This is the first step of the per-trial pipeline and is
%   shared by both the scalar and multi-dimensional paths (it operates only on
%   discrete context labels, so it is dimension-agnostic).
%
%   The prior for particle p is the transition row out of that particle's
%   current context, taken from the local (per-particle) transition matrix:
%
%       prior(:,p) = T_p(context_p, :)'          (then column-normalised)
%
%   When a cue is observed the predicted distribution multiplies in the cue
%   emission probabilities and re-normalises:
%
%       predicted(:,p) = normalise( prior(:,p) .* p(q | c, p) )
%
%   Inputs:
%     q   cue label for this trial ([] when the trial is un-cued). A non-empty
%         q indexes the second dimension of D.local_cue_matrix.
%
%   Writes obj.D.prior_probabilities, obj.D.probability_cue and
%   obj.D.predicted_probabilities (each (max_contexts+1)-by-num_particles). The
%   trailing (+1) slot is the novel-context slot.

    % This step owns its dependencies: refresh the local transition matrix (and,
    % below, the local cue matrix) here so predictContext has no hidden ordering
    % dependency on an external caller having updated them first.
    obj.updateLocalTransitionMatrix();
    P = obj.num_particles;
    Cmax = obj.max_contexts + 1;
    % D is a local read-only alias of the particle struct obj.D, used only to
    % keep the transition-row gather below terse. Reads go through D; every write
    % goes back through obj.D so the handle object is the single source of truth.
    % %#ok<*PROPLC> suppresses the "local variable D shadows property D" lint for
    % the whole file - the aliasing is intentional, not an accidental shadow.
    D = obj.D; %#ok<*PROPLC>
    prior = zeros(Cmax, P);
    for p = 1:P
        % Prior = transition row out of the particle's current context.
        prior(:,p) = D.local_transition_matrix(D.context(p), :, p)';
    end
    obj.D.prior_probabilities = obj.normalizeColumns(prior);

    if isempty(q)
        % Un-cued trial: cue term is a no-op, so predicted == prior.
        obj.D.probability_cue = ones(Cmax, P);
        obj.D.predicted_probabilities = obj.D.prior_probabilities;
    else
        obj.updateLocalCueMatrix();
        % Gather p(q|c,p) for the observed cue across all contexts/particles.
        pcue = squeeze(obj.D.local_cue_matrix(:, q, :));
        if P == 1
            % squeeze collapses the singleton particle axis; restore a column.
            pcue = pcue(:);
        end
        obj.D.probability_cue = pcue;
        obj.D.predicted_probabilities = obj.normalizeColumns(obj.D.prior_probabilities .* pcue);
    end
end
