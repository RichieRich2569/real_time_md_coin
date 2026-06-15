function y = safeLog(~, x)
    y = log(max(x, realmin));
end
