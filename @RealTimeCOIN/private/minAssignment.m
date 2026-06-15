function assignment = minAssignment(~, cost)
    n = size(cost, 1);
    nMasks = 2^n;
    dp = Inf(n+1, nMasks);
    parent = zeros(n+1, nMasks);
    dp(1,1) = 0;
    for source = 1:n
        for mask = 0:(nMasks-1)
            current = dp(source, mask+1);
            if ~isfinite(current)
                continue;
            end
            for target = 1:n
                bit = bitshift(1, target-1);
                if bitand(mask, bit) == 0
                    nextMask = bitor(mask, bit);
                    candidate = current + cost(source,target);
                    if candidate < dp(source+1, nextMask+1)
                        dp(source+1, nextMask+1) = candidate;
                        parent(source+1, nextMask+1) = target;
                    end
                end
            end
        end
    end
    assignment = zeros(1,n);
    mask = nMasks - 1;
    for source = n:-1:1
        target = parent(source+1, mask+1);
        if target == 0
            target = source;
        end
        assignment(source) = target;
        mask = mask - bitshift(1, target-1);
    end
end
