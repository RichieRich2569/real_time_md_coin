function S = diagnostics(obj)
%DIAGNOSTICS Full globally-aligned snapshot of the current particle state.
%
%   S = diagnostics(obj) returns a struct summarising every per-context quantity
%   of the current trial, relabelled from each particle's arbitrary local
%   contexts into a single aligned global-context frame. For the scalar model the
%   fields are populated below; for the multi-dimensional model (state_dim > 1)
%   the call is delegated to diagnosticsMD (see its help for the MD field list).
%   This triggers (and caches) the lazy context alignment.
%
%   Scalar fields (K = number of aligned contexts):
%       trial, C                          trial index and context count K
%       context                           sampled global context per modal particle
%       predicted_probabilities           prior per-context weights
%       responsibilities                  posterior per-context weights
%       state_mean, state_var             filtered state moments per context
%       state_feedback_mean/_var          predicted feedback moments per context
%       retention, drift, bias            per-context dynamics parameters
%       global_transition_probabilities   franchise transition weights
%       local_transition_matrix           per-context transition tensor
%       global_cue_probabilities          franchise cue weights (modal particles)
%       local_cue_matrix                  per-context cue tensor
%       alignment                         the context-alignment struct
%       raw                               handle to the raw particle struct D
%
%   See also DIAGNOSTICSMD, CONTEXT_ALIGNMENT, RESPONSIBILITIES,
%   PREDICTED_CONTEXT_PROBABILITIES.
    arguments
        obj (1, 1) RealTimeCOIN
    end
    if obj.state_dim > 1
        S = diagnosticsMD(obj);
        return;
    end

    alignment = ensureContextAlignment(obj);

    S = struct();
    S.trial = obj.trial;
    S.C = alignment.K;
    S.context = globalSampledContexts(obj, alignment);
    S.predicted_probabilities = globalContextWeights(obj, obj.D.predicted_probabilities, alignment);
    S.responsibilities = globalContextWeights(obj, obj.D.responsibilities, alignment);

    % Per-context parameter matrices that share the same global-frame relabelling
    % via globalContextMatrix. Each field name matches its raw D field, so a
    % single table drives all of them (assignment order preserves field order).
    matrixFields = ["state_mean", "state_var", ...
                    "state_feedback_mean", "state_feedback_var", ...
                    "retention", "drift", "bias", ...
                    "global_transition_probabilities"];
    for f = matrixFields
        S.(f) = globalContextMatrix(obj, obj.D.(f), alignment);
    end

    S.local_transition_matrix = globalTransitionTensor(obj, obj.D.local_transition_matrix, alignment);
    S.global_cue_probabilities = obj.D.global_cue_probabilities(:, alignment.modal_particle_indices);
    S.local_cue_matrix = globalCueTensor(obj, obj.D.local_cue_matrix, alignment);
    S.alignment = alignment;
    S.raw = obj.D;
end
