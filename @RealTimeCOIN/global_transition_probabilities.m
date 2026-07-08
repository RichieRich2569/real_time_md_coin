function w = global_transition_probabilities(obj)
%GLOBAL_TRANSITION_PROBABILITIES Expected global (franchise) transition distribution.
%
%   w = global_transition_probabilities(obj) returns the 1-by-(max_contexts+1)
%   expected global context distribution of the hierarchical Dirichlet process,
%   averaged over the modal particles and mapped into the aligned global-context
%   frame (trailing entries beyond the number of active contexts are zero; the
%   last active entry is the novel-context stick). Mirrors COIN's
%   plot_global_transition_probabilities. Dimension-independent.
    arguments
        obj (1, 1) RealTimeCOIN
    end
    alignment = ensureContextAlignment(obj);
    Wg = obj.globalContextWeights(obj.D.global_transition_probabilities, alignment);
    w = mean(Wg, 2)';
    w = renormalizeGlobalWeights(w);
end
