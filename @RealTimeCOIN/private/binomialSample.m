function n = binomialSample(~, trials, prob)
    if trials <= 0 || prob <= 0
        n = 0;
    elseif prob >= 1
        n = trials;
    else
        n = sum(rand(1, trials) < prob);
    end
end
