function ensureCueColumn(obj, q)
%ENSURECUECOLUMN Grow the cue-indexed particle arrays to hold at least q cues.
%   ensureCueColumn(obj, q) zero-pads the cue dimension of the sufficient
%   statistics and cue-probability arrays in obj.D so that cue label q is
%   addressable:
%     - D.n_cue                  (Cmax-by-Q-by-P) cue-count sufficient stats,
%     - D.global_cue_probabilities (Q-by-P) global cue distribution,
%     - D.local_cue_matrix       (Cmax-by-Q-by-P) per-context cue likelihoods,
%   where Cmax = max_contexts + 1 and P = num_particles. Existing entries are
%   preserved; new columns/rows are initialised to 0. No-op for empty q.
    if isempty(q)
        return;
    end
    Cmax = obj.max_contexts + 1;
    P = obj.num_particles;
    if size(obj.D.n_cue, 2) < q
        extra = q - size(obj.D.n_cue, 2);
        obj.D.n_cue(:, end+1:end+extra, :) = zeros(Cmax, extra, P);
    end
    if size(obj.D.global_cue_probabilities, 1) < q
        extra = q - size(obj.D.global_cue_probabilities, 1);
        obj.D.global_cue_probabilities(end+1:end+extra, :) = 0;
    end
    if size(obj.D.local_cue_matrix, 2) < q
        extra = q - size(obj.D.local_cue_matrix, 2);
        obj.D.local_cue_matrix(:, end+1:end+extra, :) = zeros(Cmax, extra, P);
    end
end
