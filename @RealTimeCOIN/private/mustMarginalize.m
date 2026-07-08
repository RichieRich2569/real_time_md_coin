function mustMarginalize = mustMarginalize(obj, prob)
%MUSTMARGINALIZE Argument validator: mixing weights marginalise over context.
%
%   mustMarginalize(obj, prob) is an arguments-block validator shared by the
%   state_probability / state_feedback_probability density mixtures. prob is a
%   (max_contexts+1)-by-num_particles array of per-context mixing weights. When
%   prob has one row per context slot the columns are expected to sum to one
%   (each particle's context distribution is a proper pmf); the 1e-6 tolerance
%   absorbs floating-point round-off in that sum.
%
%   The function also returns the check result as a logical for callers that
%   want it, but its primary role is as a validator: the arguments block below
%   enforces that prob is finite and nonnegative.
%
%   See also state_probability, state_feedback_probability.
    arguments
        obj (1, 1) RealTimeCOIN
        prob (:, :) double {mustBeFinite, mustBeNonnegative}
    end
    mustMarginalize = true;
    if size(prob, 1) == obj.max_contexts + 1
        if any(abs(sum(prob, 1) - 1) > 1e-6)
            mustMarginalize = false;
        end
    end
end
