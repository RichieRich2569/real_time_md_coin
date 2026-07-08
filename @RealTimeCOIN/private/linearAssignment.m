function assignment = linearAssignment(cost)
%LINEARASSIGNMENT Minimum-cost one-to-one assignment for a square matrix.
%   assignment = linearAssignment(cost) solves the linear assignment problem for
%   the n-by-n cost matrix `cost`, returning a 1-by-n row vector that maps each
%   source row to a distinct target column so the total cost is minimised.
%
%   Pure-MATLAB implementation of the Jonker-Volgenant / Hungarian shortest-
%   augmenting-path method (o(n^3)); no toolbox dependency. Used to relabel a
%   particle's local contexts onto the global frame during context alignment.
%
%   Non-finite / oversized costs are replaced by a large sentinel so infeasible
%   pairings are effectively forbidden while the solver still runs. An empty
%   matrix yields an empty assignment; a non-square matrix raises an error.

    C = double(cost);
    n = size(C, 1);
    if n == 0
        assignment = zeros(1, 0);
        return;
    end
    if size(C, 2) ~= n
        error('RealTimeCOIN:AssignmentMatrixNotSquare', ...
            'Assignment cost matrix must be square.');
    end

    % large: finite sentinel standing in for +Inf / forbidden pairings, chosen
    % well above any real cost so such pairs are avoided yet keep the arithmetic
    % finite (Inf would poison the dual-variable updates below).
    large = 1e100;
    C(~isfinite(C)) = large;
    C(C > large) = large;
    % Shift so the minimum entry is 0 (assignment is invariant to a constant
    % offset); improves numerical conditioning of the potentials.
    offset = min(C(:));
    if isfinite(offset)
        C = C - offset;
    end

    % Dual potentials (u, v), column->row matching (p) and augmenting-path
    % back-pointers (way). Column index 1 is a virtual "unmatched" sentinel, so
    % real columns occupy indices 2..n+1.
    u = zeros(n, 1);
    v = zeros(n + 1, 1);
    p = zeros(n + 1, 1);
    way = zeros(n + 1, 1);

    % Add rows one at a time, each time growing the matching along a shortest
    % augmenting path in the reduced-cost graph.
    for i = 1:n
        p(1) = i;
        j0 = 1;
        minv = inf(n + 1, 1);
        used = false(n + 1, 1);
        while true
            used(j0) = true;
            i0 = p(j0);
            delta = inf;
            j1 = 1;
            for j = 2:(n + 1)
                if used(j)
                    continue;
                end
                cur = C(i0, j - 1) - u(i0) - v(j);
                if cur < minv(j)
                    minv(j) = cur;
                    way(j) = j0;
                end
                if minv(j) < delta
                    delta = minv(j);
                    j1 = j;
                end
            end
            if ~isfinite(delta)
                delta = large;
            end
            for j = 1:(n + 1)
                if used(j)
                    u(p(j)) = u(p(j)) + delta;
                    v(j) = v(j) - delta;
                else
                    minv(j) = minv(j) - delta;
                end
            end
            j0 = j1;
            if p(j0) == 0
                break;
            end
        end

        while true
            j1 = way(j0);
            p(j0) = p(j1);
            j0 = j1;
            if j0 == 1
                break;
            end
        end
    end

    % Read the row->column matching out of the column->row map p.
    assignment = zeros(1, n);
    for j = 2:(n + 1)
        if p(j) > 0
            assignment(p(j)) = j - 1;
        end
    end

    % Safety net: assign any row left unmatched (possible only under degenerate
    % all-sentinel costs) to the remaining free columns, preserving order.
    missing = assignment == 0;
    if any(missing)
        unused = setdiff(1:n, assignment(~missing), 'stable');
        assignment(missing) = unused(1:sum(missing));
    end
end
