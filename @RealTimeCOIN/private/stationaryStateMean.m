function m = stationaryStateMean(~, a, d)
    denom = 1 - a;
    m = zeros(size(a));
    good = abs(denom) > eps;
    m(good) = d(good) ./ denom(good);
end
