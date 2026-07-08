function alignment = context_alignment(obj)
%CONTEXT_ALIGNMENT Global context alignment across particles (cached).
%
%   alignment = context_alignment(obj) returns the struct that maps each
%   particle's arbitrary local context labels onto a single, globally consistent
%   labelling for the current trial. It is computed lazily and cached on first
%   call after a state change (the cache is invalidated by each observe_y), so
%   repeated context-facing queries within a trial share one alignment.
%
%   Key fields include:
%       K                       number of instantiated (aligned) global contexts
%       assignment              per-particle local-to-global label mapping
%       modal_particle_mask     particles whose context count equals the mode
%       modal_particle_indices  indices of those modal particles
%       global_contexts         per-global-context prototype parameters
%       converged, iterations   alignment-solver diagnostics
%
%   See also DIAGNOSTICS, PREDICTED_CONTEXT_PROBABILITIES,
%   CONTEXT_RESPONSIBILITIES, LOCAL_TRANSITION_PROBABILITIES.
    arguments
        obj (1, 1) RealTimeCOIN
    end
    alignment = ensureContextAlignment(obj);
end
