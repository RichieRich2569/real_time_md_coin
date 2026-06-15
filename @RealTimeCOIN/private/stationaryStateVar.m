function v = stationaryStateVar(obj, a)
    denom = 1 - a.^2;
    v = zeros(size(a));
    good = denom > eps;
    v(good) = obj.sigma_process_noise^2 ./ denom(good);
end
