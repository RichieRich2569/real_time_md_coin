function y = safeLog(~, x)
%SAFELOG Natural logarithm floored at realmin to avoid log(0) = -Inf.
%   y = safeLog(obj, x) returns log(max(x, realmin)) element-wise, so that
%   exact-zero probabilities map to log(realmin) (a large finite negative
%   number) instead of -Inf. The leading obj argument is ignored (private
%   RealTimeCOIN method invoked as obj.safeLog).
    y = log(max(x, realmin));
end
