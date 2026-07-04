function L = local_cue_probabilities(obj)
%LOCAL_CUE_PROBABILITIES Expected local cue-emission matrix.
%
%   L = local_cue_probabilities(obj) returns the K-by-Q expected local cue
%   probability matrix in the aligned global-context frame, where K =
%   context_alignment(obj).K and Q is the number of observed cue labels (plus a
%   trailing novel-cue column). Row i is the cue-emission distribution of global
%   context i. Mirrors COIN's plot_local_cue_probabilities (a single trial).
%
%   Requires at least one observed sensory cue (as in COIN).
    arguments
        obj (1, 1) RealTimeCOIN
    end
    if isempty(obj.cue_values)
        error('RealTimeCOIN:NoCues', ...
            'local_cue_probabilities requires the model to have observed sensory cues.');
    end
    alignment = ensureContextAlignment(obj);
    K = alignment.K;
    L = alignment.global_contexts.cue_prob(1:K, :);
end
