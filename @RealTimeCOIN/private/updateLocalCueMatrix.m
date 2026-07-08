function updateLocalCueMatrix(obj)
%UPDATELOCALCUEMATRIX Rebuild per-particle context-to-cue emission matrices.
%   updateLocalCueMatrix(obj) computes obj.D.local_cue_matrix, a
%   Cmax-by-Qn-by-P tensor whose (c, :, p) row is the posterior-mean
%   distribution over cue emissions for context c in particle p. Each row is the
%   mean of a Dirichlet whose parameters combine the observed cue counts with
%   the global HDP cue base measure:
%       raw(c, k) = n_cue(c, k) + alpha_cue * beta_cue(k)
%   normalised over cue k. Rows for contexts that do not yet exist are zeroed
%   before normalisation. Structurally mirrors updateLocalTransitionMatrix (see
%   the noted control-flow duplication) but has no sticky self-transition term.

    Cmax = obj.max_contexts + 1;
    P = obj.num_particles;
    Qn = max(1, size(obj.D.global_cue_probabilities, 1));   % number of cue values
    L = zeros(Cmax, Qn, P);
    for p = 1:P
        % Take this particle's cue counts, trimming to the known cue values and
        % right-padding with zeros if fewer columns have been instantiated.
        counts = obj.D.n_cue(:,1:min(Qn,size(obj.D.n_cue,2)),p);
        if size(counts,2) < Qn
            counts(:,end+1:Qn) = 0;
        end
        % Unnormalised Dirichlet parameters: counts + HDP cue base measure.
        raw = counts + obj.alpha_cue .* obj.D.global_cue_probabilities(1:Qn,p)';
        % A context is "valid" if instantiated (1..C) or the single novel slot.
        valid = false(Cmax,1);
        valid(1:obj.D.C(p)) = true;
        if obj.D.C(p) < obj.max_contexts
            valid(obj.D.C(p)+1) = true;
        end
        raw(~valid,:) = 0;   % zero out non-existent context rows
        % Row-normalise each valid context row to a proper emission distribution.
        rowSums = sum(raw, 2);
        for c = 1:Cmax
            if rowSums(c) > 0
                L(c,:,p) = raw(c,:) ./ rowSums(c);
            end
        end
    end
    obj.D.local_cue_matrix = L;
end
