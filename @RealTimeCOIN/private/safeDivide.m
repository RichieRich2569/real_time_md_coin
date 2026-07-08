function z = safeDivide(~, a, b)
%SAFEDIVIDE Element-wise division that returns zero where the divisor is ~0.
%   z = safeDivide(obj, a, b) computes a ./ b element-wise, but sets the result
%   to 0 wherever abs(b) <= eps instead of producing Inf/NaN. a and b must be
%   the same size (z is preallocated from size(a)). The leading obj argument is
%   ignored (private RealTimeCOIN method invoked as obj.safeDivide).
    z = zeros(size(a));
    good = abs(b) > eps;
    z(good) = a(good) ./ b(good);
end
