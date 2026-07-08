function V = precisionToVariance(~, precision)
%PRECISIONTOVARIANCE Convert a scalar precision to a variance.
%   V = precisionToVariance(obj, precision) returns 1 ./ precision, mapping a
%   precision of exactly 0 (a completely uninformative belief) to V = Inf
%   rather than dividing by zero. The leading obj argument is ignored (private
%   RealTimeCOIN method invoked as obj.precisionToVariance).
    if precision == 0
        V = Inf;
    else
        V = 1 ./ precision;
    end
end
