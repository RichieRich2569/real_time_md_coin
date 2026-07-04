function C = palette(n)
%COINVIZ.PALETTE Consistent, colourblind-friendly line/patch colours.
%   C = coinviz.palette(n) returns an n-by-3 matrix of RGB colours drawn from
%   the Okabe-Ito colourblind-safe qualitative palette, cycling if n exceeds
%   the eight base colours. Calling with no argument returns all eight.
%
%   Every plot in the examples routes its colours through this one function so
%   that context k has the same colour across every figure and both notebooks.
    base = [ ...
        0.00 0.45 0.70;   % blue
        0.90 0.60 0.00;   % orange
        0.00 0.60 0.50;   % bluish green
        0.80 0.40 0.00;   % vermillion
        0.35 0.70 0.90;   % sky blue
        0.80 0.60 0.70;   % reddish purple
        0.95 0.90 0.25;   % yellow
        0.00 0.00 0.00];  % black
    if nargin < 1 || isempty(n)
        C = base;
        return;
    end
    idx = mod((0:n-1), size(base, 1)) + 1;
    C = base(idx, :);
end
