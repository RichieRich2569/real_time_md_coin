function densities = state_given_context_probability(obj, values)
%STATE_GIVEN_CONTEXT_PROBABILITY Per-context posterior state density on a grid.
%
%   Returns a containers.Map keyed by global context label, each value the
%   posterior state density of that context evaluated at the query points. For
%   the scalar model (state_dim == 1) values is a vector and each density is a
%   row vector of equal length; for the multi-dimensional model values is an
%   N-by-K matrix of column query points and each density is a 1-by-K row.
%   Densities use the aligned global-context prototype moments.

    densities = containers.Map('KeyType', 'double', 'ValueType', 'any');
    alignment = ensureContextAlignment(obj);
    active = activeSummaryContexts(obj);
    active = active(active <= alignment.K);
    if isempty(active) && alignment.K > 0
        active = 1;
    end

    if obj.state_dim == 1
        values = values(:)';
        M = alignment.global_contexts.state_mean;
        V = alignment.global_contexts.state_var;
        for c = active
            densities(c) = obj.normal_pdf(values, M(c), V(c));
        end
        return;
    end

    N = obj.state_dim;
    if size(values, 1) ~= N
        error('RealTimeCOIN:GridDimensionMismatch', ...
            ['state_given_context_probability expects an %d-by-K grid (each ', ...
             'column a query point) for state_dim == %d; received a %d-by-%d array.'], ...
            N, N, size(values, 1), size(values, 2));
    end
    M = alignment.global_contexts.state_mean;     % N-by-Km
    V = alignment.global_contexts.state_cov;      % N-by-N-by-Km
    for c = active
        densities(c) = obj.gaussianPdfColumnsMD(values, M(:,c), V(:,:,c));
    end
end
