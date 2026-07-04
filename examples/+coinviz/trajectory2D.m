function fig = trajectory2D(pathXY, targets, opts)
%COINVIZ.TRAJECTORY2D 2-D state / motor-output trajectory with target markers.
%   fig = coinviz.trajectory2D(pathXY, targets, opts) plots a 2-D path (the
%   posterior state mean or the motor-output path) together with the block
%   target locations. Both scalar-lifted and genuinely 2-D examples use this.
%
%   Inputs:
%     pathXY   2-by-T path (row 1 = dim 1, row 2 = dim 2).
%     targets  2-by-M marker locations (block targets); [] to omit.
%
%   Name-value options:
%     FigName     figure title bar text.
%     Title       axes title.
%     PathName    legend label for the path.
%     TargetName  legend label for the target markers.
%     ObservedXY  2-by-T observed points to overlay as faint markers; [] to omit.
    arguments
        pathXY (2, :) double
        targets double = []
        opts.FigName (1, :) char = '2-D trajectory'
        opts.Title (1, :) char = '2-D trajectory'
        opts.PathName (1, :) char = 'Path'
        opts.TargetName (1, :) char = 'Targets'
        opts.ObservedXY double = []
    end

    cols = coinviz.palette(2);
    fig = figure('Name', opts.FigName);
    ax = axes(fig); hold(ax, 'on');
    handles = gobjects(0); labels = {};

    if ~isempty(opts.ObservedXY)
        plot(ax, opts.ObservedXY(1, :), opts.ObservedXY(2, :), 'x', ...
            'Color', [0.75 0.75 0.75], 'MarkerSize', 4, 'HandleVisibility', 'off');
    end
    h = plot(ax, pathXY(1, :), pathXY(2, :), '-', 'Color', cols(1, :), 'LineWidth', 1.3);
    handles(end+1) = h; labels{end+1} = opts.PathName;
    if ~isempty(targets)
        h = plot(ax, targets(1, :), targets(2, :), 'p', 'MarkerSize', 14, ...
            'MarkerFaceColor', cols(2, :), 'MarkerEdgeColor', 'k');
        handles(end+1) = h; labels{end+1} = opts.TargetName;
    end

    xlabel(ax, 'Dim 1'); ylabel(ax, 'Dim 2');
    title(ax, opts.Title);
    legend(ax, handles, labels, 'Location', 'best');
    axis(ax, 'equal');
end
