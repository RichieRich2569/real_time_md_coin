function m = randnumtable(base, counts)
%RANDNUMTABLE MATLAB fallback for Chinese-restaurant table counts.
%
%   M = RANDNUMTABLE(BASE, COUNTS) returns, elementwise, the number of
%   occupied tables in a Chinese Restaurant Process (CRP) given a
%   concentration value BASE and a customer count COUNTS.
%
%   The original COIN code expects the npbayes MEX function randnumtable.
%   This fallback keeps validation runnable when the MEX file has not been
%   compiled. BASE and COUNTS must be the same size; each output element
%   M(i) is drawn independently from the CRP defined by BASE(i), COUNTS(i).
%
%   IMPORTANT: the customer-by-customer loop below reproduces the reference
%   CRP sampling algorithm exactly and MUST match the npbayes MEX RNG
%   behaviour. Do not vectorise or otherwise alter the sampling logic: it
%   consumes one RAND draw per customer, in order, so any change would alter
%   the random-number stream and diverge from the MEX implementation.

if ~isequal(size(base), size(counts))
    error('randnumtable:SizeMismatch', 'base and counts must have the same size.');
end

m = zeros(size(counts));
for i = 1:numel(counts)
    n = counts(i);
    b = base(i);
    % A non-positive count or concentration seats no customers -> zero tables.
    if n <= 0 || b <= 0
        continue;
    end
    tables = 0;
    for customer = 1:n
        % Customer 'customer' starts a new table (increments the count) with
        % probability b / (b + customer - 1); otherwise it joins an existing
        % table. One rand() draw per customer, matching the reference stream.
        tables = tables + (rand < b ./ (b + customer - 1));
    end
    m(i) = tables;
end
end
