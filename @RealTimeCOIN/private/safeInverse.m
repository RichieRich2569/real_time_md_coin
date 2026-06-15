function Ainv = safeInverse(~, A)
    if rcond(A) < 1e-12
        Ainv = pinv(A);
    else
        Ainv = inv(A);
    end
end
