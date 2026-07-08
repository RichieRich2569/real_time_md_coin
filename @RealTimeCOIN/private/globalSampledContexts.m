function c = globalSampledContexts(obj, alignment)
%GLOBALSAMPLEDCONTEXTS Sampled context of each modal particle in global labels.
%   c = globalSampledContexts(obj, alignment) returns the currently sampled
%   context (obj.D.context) of every modal particle, translated from its local
%   per-particle label into the shared global-context label via
%   alignment.assignment.
%
%   Inputs:
%     alignment  alignment struct (see ensureContextAlignment). Optional;
%                recomputed lazily when omitted.
%
%   Output:
%     c   1-by-nModal row of global context labels. Entries are NaN where the
%         sampled local context has no global assignment (unmatched slot).
    if nargin < 2
        alignment = obj.ensureContextAlignment();
    end
    c = obj.scatterToGlobal([], alignment, "labels");
end
