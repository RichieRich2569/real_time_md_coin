function m = randnumtable(base, counts)
%RANDNUMTABLE MATLAB fallback for Chinese-restaurant table counts.
%
%   The original COIN code expects the npbayes MEX function randnumtable.
%   This fallback keeps validation runnable when the MEX file has not been
%   compiled. Inputs are elementwise concentration values and customer
%   counts of the same size.

if ~isequal(size(base), size(counts))
    error('randnumtable:SizeMismatch', 'base and counts must have the same size.');
end

m = zeros(size(counts));
for i = 1:numel(counts)
    n = counts(i);
    b = base(i);
    if n <= 0 || b <= 0
        continue;
    end
    tables = 0;
    for customer = 1:n
        tables = tables + (rand < b ./ (b + customer - 1));
    end
    m(i) = tables;
end
end
