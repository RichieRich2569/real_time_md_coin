function [passed, checks] = validation_pass_summary(checks)
%VALIDATION_PASS_SUMMARY Aggregate named validation checks.
%
%   CHECKS is a struct whose fields are scalar logical or numeric values.
%   NaN counts as a failure because it indicates the validation did not
%   produce an interpretable metric.

names = fieldnames(checks);
passed = true;
for i = 1:numel(names)
    value = checks.(names{i});
    ok = islogical(value) && isscalar(value) && value;
    ok = ok || (isnumeric(value) && isscalar(value) && isfinite(value) && value ~= 0);
    checks.(names{i}) = ok;
    passed = passed && ok;
end
end
