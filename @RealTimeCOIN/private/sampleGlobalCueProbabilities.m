function sampleGlobalCueProbabilities(obj)
%SAMPLEGLOBALCUEPROBABILITIES Resample the global cue distribution (HDP top level).
%
%   sampleGlobalCueProbabilities(obj) updates, per particle, the global
%   ("beta") distribution over cues that anchors the hierarchical Dirichlet
%   process (HDP) prior on each context's cue emission distribution. It
%   follows the standard Teh et al. HDP posterior for the top-level measure:
%     1. Draw Antoniak/CRP table counts m for the observed cue counts n_cue,
%        with base measure alpha_cue * current global cue probabilities.
%     2. Sum tables across contexts to form the Dirichlet parameters, add the
%        novelty mass gamma_cue in the first unused ("novel cue") slot, and
%        zero any slots beyond it.
%     3. Draw the refreshed global cue probabilities from that Dirichlet.
%   Unlike the transition sampler there is no self-transition (sticky) term.

    % Nothing to do until at least one cue has been registered.
    if size(obj.D.global_cue_probabilities, 1) == 0
        return;
    end
    P = obj.num_particles;
    % Column budget: at least one slot past the highest seen cue label Q.
    Qn = max(obj.D.Q + 1, size(obj.D.global_cue_probabilities, 1));
    obj.ensureCueColumn(Qn);
    for p = 1:P
        counts = obj.D.n_cue(:,1:Qn,p);         % per-context cue counts
        % Antoniak base measure alpha_cue * beta, broadcast over contexts (rows).
        base = repmat(obj.alpha_cue .* obj.D.global_cue_probabilities(1:Qn,p)', size(counts,1), 1);
        m = obj.sample_num_tables(base, counts);
        alpha = sum(m, 1);                      % Dirichlet params from summed tables
        alpha(obj.D.Q + 1) = obj.gamma_cue;     % novelty mass at the first unused cue slot
        if obj.D.Q + 2 <= Qn
            alpha(obj.D.Q + 2:end) = 0;         % zero slots beyond the novel cue
        end
        obj.D.global_cue_probabilities(1:Qn,p) = obj.dirichletSample(alpha(:));
    end
end
