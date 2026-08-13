function G = scatterToGlobal(obj, X, alignment, mode)
%SCATTERTOGLOBAL Scatter per-particle local context slots into the global frame.
%   G = scatterToGlobal(obj, X, alignment, mode) is the shared skeleton behind the
%   global-aggregation query helpers. It walks the modal particles (those at the
%   modal context cardinality), maps each local context slot to its global slot via
%   alignment.assignment (0 = unmatched), guards the target to (0, Cmax], and
%   accumulates X into the global-frame output. One output column/page is produced
%   per modal particle.
%
%   The mode selects the accumulation rule, tensor rank, and post-processing so the
%   single skeleton reproduces each caller byte-for-byte:
%     "overwrite"  - rank-2 matrix; last local slot mapped to a global slot wins.
%                    Backs globalContextMatrix.
%     "add"        - rank-2 matrix; local slots summed into shared global slots.
%                    When the modal cardinality Km has reached max_contexts the
%                    novel slot carries no mass, so slots beyond Km are zeroed.
%                    Backs globalContextWeights.
%     "cue"        - rank-3 context-by-cue tensor; the cue axis is carried through
%                    unchanged while context slots are summed. Backs globalCueTensor.
%     "transition" - rank-3 from-by-to tensor; BOTH context axes are realigned and
%                    matching (from,to) pairs summed. Backs globalTransitionTensor.
%     "labels"     - 1-by-nModal row of the global label of each particle's sampled
%                    context (obj.D.context); X is ignored. Backs
%                    globalSampledContexts.
%
%   Inputs:
%     X          per-particle array indexed by local context slot; shape depends on
%                mode (Cmax-by-P, Cmax-by-Q-by-P, or Cmax-by-Cmax-by-P). Ignored
%                for the "labels" mode.
%     alignment  alignment struct (see ensureContextAlignment).
%     mode       accumulation mode string (see above).
%
%   Output:
%     G   global-frame aggregate with one column/page per modal particle. Unmatched
%         local slots are dropped. Shape and fill value depend on mode.
    Cmax = obj.max_contexts + 1;                 % context slots incl. novel (+1)
    Km = alignment.K;                            % modal context cardinality
    modalIdx = alignment.modal_particle_indices; % particles at the modal cardinality
    nModal = numel(modalIdx);
    switch mode
        case "cue"
            G = zeros(Cmax, size(X, 2), nModal);
        case "transition"
            G = zeros(Cmax, Cmax, nModal);
        case "labels"
            G = NaN(1, nModal);
        otherwise
            G = zeros(Cmax, nModal);
    end
    for idx = 1:nModal
        p = modalIdx(idx);
        switch mode
            case "overwrite"
                for local = 1:Cmax
                    target = alignment.assignment(local, p); % global slot for this local slot
                    if target > 0 && target <= Cmax
                        G(target, idx) = X(local, p);
                    end
                end
            case "add"
                for local = 1:Cmax
                    target = alignment.assignment(local, p); % global slot for this local slot
                    if target > 0 && target <= Cmax
                        G(target, idx) = G(target, idx) + X(local, p);
                    end
                end
                if Km >= obj.max_contexts
                    % All contexts instantiated: no novel slot, drop trailing slots.
                    G(Km+1:end, idx) = 0;
                end
            case "cue"
                for local = 1:Cmax
                    target = alignment.assignment(local, p); % global slot for this local slot
                    if target > 0 && target <= Cmax
                        G(target, :, idx) = G(target, :, idx) + X(local, :, p);
                    end
                end
            case "transition"
                for localFrom = 1:Cmax
                    globalFrom = alignment.assignment(localFrom, p); % global "from" slot
                    if globalFrom <= 0 || globalFrom > Cmax
                        continue;
                    end
                    for localTo = 1:Cmax
                        globalTo = alignment.assignment(localTo, p); % global "to" slot
                        if globalTo > 0 && globalTo <= Cmax
                            G(globalFrom, globalTo, idx) = ...
                                G(globalFrom, globalTo, idx) + X(localFrom, localTo, p);
                        end
                    end
                end
            case "labels"
                local = obj.D.context(p);           % this particle's sampled local slot
                if local <= size(alignment.assignment, 1)
                    target = alignment.assignment(local, p); % corresponding global slot
                    if target > 0
                        G(idx) = target;
                    end
                end
        end
    end
end
