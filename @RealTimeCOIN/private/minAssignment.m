function assignment = minAssignment(~, cost)
%MINASSIGNMENT Method-dispatched wrapper for the min-cost assignment solver.
%   assignment = minAssignment(obj, cost) returns the minimum-cost one-to-one
%   assignment of the square matrix `cost`. It is a thin pass-through to the
%   package-private linearAssignment function, exposed as a RealTimeCOIN method
%   (first argument is the object, ignored) so callers can invoke it as
%   obj.minAssignment(cost) - the indirection is a seam for swapping in an
%   alternative solver without touching optimizeContextAlignment.

    assignment = linearAssignment(cost);
end
