function [pmf, labels] = previewCuePmf(obj)
%PREVIEWCUEPMF Predicted distribution over cue labels for the next trial.
%   [pmf, labels] = previewCuePmf(obj) returns a read-only one-step-ahead
%   probability mass function over the currently instantiated cue labels. For
%   each particle it propagates the sampled context through its local
%   transition matrix, weights the per-context cue likelihoods by the resulting
%   context distribution, then averages across particles and renormalises.
%     pmf    : 1-by-Q predicted cue probabilities (sums to 1 when non-degenerate),
%     labels : 1-by-Q integer cue labels (1:Q) indexing obj.cue_values.
%   Q is the current number of cue columns. Does not mutate model state.
    Cmax = obj.max_contexts + 1;
    P = obj.num_particles;
    Qn = size(obj.D.local_cue_matrix, 2);
    labels = 1:Qn;
    pmf = zeros(1, Qn);
    prior = obj.currentTransitionPrior();
    for p = 1:P
        cueGivenContext = obj.D.local_cue_matrix(:,:,p);
        if size(cueGivenContext,1) < Cmax
            cueGivenContext(Cmax,Qn) = 0;
        end
        pmf = pmf + (prior(:,p)' * cueGivenContext);
    end
    pmf = pmf ./ P;
    if sum(pmf) > 0
        pmf = pmf ./ sum(pmf);
    end
end
