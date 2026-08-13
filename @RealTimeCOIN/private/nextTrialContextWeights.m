function W = nextTrialContextWeights(obj, qLabel)
%NEXTTRIALCONTEXTWEIGHTS One-step-ahead predicted context weights (read-only).
%
%   W = nextTrialContextWeights(obj, qLabel) returns the (max_contexts+1)-by-P
%   normalised context weights the model would predict on the next trial, given
%   the optional upcoming cue label qLabel ([] marginalises the cue out). It
%   propagates each particle's currently sampled context through its expected
%   local transition matrix (and, if a cue is supplied, multiplies by the local
%   cue likelihood), exactly as predictContext / previewPredictiveFeedback do
%   for their weight term, without mutating the model state. Dimension-agnostic:
%   the transition and cue matrices have the same layout for the scalar and
%   multi-dimensional pipelines. Used by the c*2 query methods.
    P = obj.num_particles;
    prior = obj.currentTransitionPrior();

    if isempty(qLabel)
        W = prior;
    else
        qCol = min(qLabel, size(obj.D.local_cue_matrix, 2));
        pcue = squeeze(obj.D.local_cue_matrix(:, qCol, :));
        if P == 1
            pcue = pcue(:);
        end
        W = obj.normalizeColumns(prior .* pcue);
    end
end
