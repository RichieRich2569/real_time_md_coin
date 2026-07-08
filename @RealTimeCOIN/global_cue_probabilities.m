function w = global_cue_probabilities(obj)
%GLOBAL_CUE_PROBABILITIES Expected global (franchise) cue distribution.
%
%   w = global_cue_probabilities(obj) returns the 1-by-Q expected global cue
%   distribution of the hierarchical Dirichlet process, averaged over particles.
%   Cue labels are numbered by order of presentation (so they need no context
%   alignment); the trailing entry is the novel-cue stick. Mirrors COIN's
%   plot_global_cue_probabilities.
%
%   Requires at least one observed sensory cue (as in COIN).
    arguments
        obj (1, 1) RealTimeCOIN
    end
    if isempty(obj.cue_values)
        error("RealTimeCOIN:NoCues", ...
            "global_cue_probabilities requires the model to have observed sensory cues.");
    end
    w = mean(obj.D.global_cue_probabilities, 2)';
    w = renormalizeGlobalWeights(w);
end
