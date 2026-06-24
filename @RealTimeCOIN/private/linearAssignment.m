function assignment = linearAssignment(cost)
%LINEARASSIGNMENT Minimum-cost one-to-one assignment for a square matrix.
%
%   Pure-MATLAB Hungarian/shortest augmenting path implementation. The
%   returned row vector maps each source row to a target column.

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

    large = 1e100;
    C(~isfinite(C)) = large;
    C(C > large) = large;
    offset = min(C(:));
    if isfinite(offset)
        C = C - offset;
    end

    u = zeros(n, 1);
    v = zeros(n + 1, 1);
    p = zeros(n + 1, 1);
    way = zeros(n + 1, 1);

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

    assignment = zeros(1, n);
    for j = 2:(n + 1)
        if p(j) > 0
            assignment(p(j)) = j - 1;
        end
    end

    missing = assignment == 0;
    if any(missing)
        unused = setdiff(1:n, assignment(~missing), 'stable');
        assignment(missing) = unused(1:sum(missing));
    end
end
