function V = precisionToVariance(~, precision)
    if precision == 0
        V = Inf;
    else
        V = 1 ./ precision;
    end
end
