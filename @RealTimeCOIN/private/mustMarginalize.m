function mustMarginalize(obj, prob)
%MUSTMARGINALIZE Argument validator: mixing weights marginalise over context.
%
%   mustMarginalize(obj, prob) is an arguments-block validator shared by the
%   state_probability / state_feedback_probability density mixtures. prob is a
%   (max_contexts+1)-by-num_particles array of per-context mixing weights. When
%   prob has one row per context slot the columns are expected to sum to one
%   (each particle's context distribution is a proper pmf); the 1e-6 tolerance
%   absorbs floating-point round-off in that sum.
%
%   As an arguments-block validator the function returns no value: it validates
%   and throws on failure. The arguments block enforces that prob is finite and
%   nonnegative; when prob has one row per context slot and any column-sum
%   deviates from one by more than 1e-6, a "RealTimeCOIN:MustMarginalize" error
%   is raised. Column layouts with a different row count are not context weight
%   matrices and are left unchecked.
%
%   See also state_probability, state_feedback_probability.
    arguments
        obj (1, 1) RealTimeCOIN
        prob (:, :) double {mustBeFinite, mustBeNonnegative}
    end
    if size(prob, 1) == obj.max_contexts + 1
        if any(abs(sum(prob, 1) - 1) > 1e-6)
            error("RealTimeCOIN:MustMarginalize", ...
                "Per-context mixing weights must marginalise to one over " + ...
                "contexts; a column-sum deviates from one by more than the " + ...
                "1e-6 tolerance.");
        end
    end
end
