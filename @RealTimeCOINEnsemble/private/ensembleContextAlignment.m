function ali = ensembleContextAlignment(obj)
%ENSEMBLECONTEXTALIGNMENT Match every member's contexts onto one reference frame.
%   ali = ensembleContextAlignment(obj) computes the cross-run context alignment
%   used by the ensemble's context-indexed queries (docs/SPEC_ensemble.md Part
%   10). Each member already aligns its particles to its own arbitrary global
%   frame (RealTimeCOIN/context_alignment); this routine picks a common reference
%   frame -- the member with the most contexts (Kref = max_r K_r, ties broken by
%   lowest index) -- and, for every member r, finds the minimum-prototype-distance
%   matching of its K_r contexts onto distinct reference labels 1..Kref.
%
%   Distance is the Euclidean distance between context prototype state means
%   (global_contexts.state_mean); the matching is a linear assignment solved with
%   matchpairs. Since K_r <= Kref every member context is matched. The novel
%   context always maps to the reference novel slot (Kref+1); it is not part of
%   this matching. The alignment is deterministic (no randomness).
%
%   Returned struct fields:
%     Kref   number of reference contexts (0 when no member holds any context).
%     refIdx index of the reference member.
%     K      1-by-R vector of per-member context counts.
%     perm   1-by-R cell; perm{r}(i) is the reference label for member r's
%            context i (empty when K(r) == 0).
    R = obj.runs;
    K = zeros(1, R);
    proto = cell(1, R);
    for r = 1:R
        a = obj.members{r}.context_alignment();
        K(r) = a.K;
        proto{r} = a.global_contexts;
    end

    [Kref, refIdx] = max(K);
    perm = cell(1, R);

    if Kref == 0
        for r = 1:R
            perm{r} = zeros(1, 0);
        end
        ali = struct('Kref', 0, 'refIdx', refIdx, 'K', K, 'perm', {perm});
        return;
    end

    refMean = proto{refIdx}.state_mean;   % scalar: 1-by-Kref ; MD: N-by-Kref
    for r = 1:R
        Kr = K(r);
        if Kr == 0
            perm{r} = zeros(1, 0);
            continue;
        end
        C = memberRefCost(proto{r}.state_mean, refMean, Kr, Kref);
        perm{r} = assignMemberContexts(C);
    end

    ali = struct('Kref', Kref, 'refIdx', refIdx, 'K', K, 'perm', {perm});
end

function C = memberRefCost(memberMean, refMean, Kr, Kref)
%MEMBERREFCOST Kr-by-Kref prototype-distance cost between member and reference.
    if size(memberMean, 1) == 1
        % Scalar model: |mean_i - mean_j|.
        C = abs(memberMean(1:Kr).' - refMean(1:Kref));
    else
        % MD model: Euclidean distance between state-mean vectors.
        C = zeros(Kr, Kref);
        for i = 1:Kr
            d = refMean(:, 1:Kref) - memberMean(:, i);
            C(i, :) = sqrt(sum(d.^2, 1));
        end
    end
end

function perm = assignMemberContexts(C)
%ASSIGNMEMBERCONTEXTS Match each member context (row) to a distinct ref label.
%   Solves the rectangular linear assignment (Kr-by-Kref, Kr <= Kref) with
%   matchpairs, forcing every row to match by making "leave unmatched" costlier
%   than any real pairing.
    Kr = size(C, 1);
    costUnmatched = max(C(:)) * 10 + 1e6;   % force all rows to be matched
    M = matchpairs(C, costUnmatched);       % rows = member contexts, cols = ref labels
    perm = zeros(1, Kr);
    for t = 1:size(M, 1)
        perm(M(t, 1)) = M(t, 2);
    end
    % Safety net: assign any row left unmatched to a remaining free label.
    if any(perm == 0)
        used = perm(perm > 0);
        free = setdiff(1:size(C, 2), used, 'stable');
        missing = find(perm == 0);
        for i = 1:numel(missing)
            perm(missing(i)) = free(i);
        end
    end
end
