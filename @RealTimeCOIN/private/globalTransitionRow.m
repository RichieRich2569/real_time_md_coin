function row = globalTransitionRow(obj, local, p, Km, assignment)
%GLOBALTRANSITIONROW Transition row of one local context mapped to global labels.
%   row = globalTransitionRow(obj, local, p, Km, assignment) takes the outgoing
%   transition probabilities of local context `local` in particle `p` and
%   scatters them into the global label frame defined by `assignment(:,p)`. The
%   returned 1-by-(Km+1) row (Km aligned contexts plus the novel-context slot) is
%   renormalised to sum to 1. Used both to build transition prototypes
%   (updateGlobalContexts) and as a cost term during assignment.

    row = zeros(1, Km + 1);
    % Consider the Km aligned destinations, plus the novel slot when it exists.
    maxLocal = Km;
    if Km < obj.max_contexts
        maxLocal = Km + 1;
    end
    for dest = 1:maxLocal
        target = assignment(dest, p);   % global label of this local destination
        if target > 0 && target <= Km + 1
            % Accumulate mass onto the destination's global label.
            row(target) = row(target) + obj.D.local_transition_matrix(local, dest, p);
        end
    end
    row = obj.normalizeProbability(row);
end
