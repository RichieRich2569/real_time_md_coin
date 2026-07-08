function prior = currentTransitionPrior(obj)
%CURRENTTRANSITIONPRIOR Column-normalised one-step context transition prior.
%   prior = currentTransitionPrior(obj) returns the (max_contexts+1)-by-P
%   matrix whose p-th column is particle p's currently sampled context row of
%   its local transition matrix, transposed to a column and normalised to sum
%   to 1 (via normalizeColumns). This is the read-only context-weight term
%   shared by the preview helpers (previewPredictiveFeedback[MD],
%   nextTrialContextWeights) and previewCuePmf. Does not mutate model state.
    arguments
        obj (1, 1) RealTimeCOIN
    end
    Cmax = obj.max_contexts + 1;
    P = obj.num_particles;
    prior = zeros(Cmax, P);
    for p = 1:P
        prior(:, p) = obj.D.local_transition_matrix(obj.D.context(p), :, p)';
    end
    prior = obj.normalizeColumns(prior);
end
