function row = globalTransitionRow(obj, local, p, Km, assignment)
    row = zeros(1, Km + 1);
    maxLocal = Km;
    if Km < obj.max_contexts
        maxLocal = Km + 1;
    end
    for dest = 1:maxLocal
        target = assignment(dest,p);
        if target > 0 && target <= Km + 1
            row(target) = row(target) + obj.D.local_transition_matrix(local,dest,p);
        end
    end
    row = obj.normalizeProbability(row);
end
