function fig = densityEvolution(trials, grid, dens, opts)
%COINVIZ.DENSITYEVOLUTION Trial-by-trial marginal density heat-map.
%   fig = coinviz.densityEvolution(trials, grid, dens, opts) draws a single
%   imagesc panel with trial on the x-axis and the query grid on the y-axis,
%   the colour encoding a marginal density (e.g. state_probability or
%   state_feedback_probability) recorded once per trial. It is the natural way
%   to watch a scalar (or one dimension of a multi-dimensional) posterior evolve
%   - for instance through blocks of missing (NaN) observations, where the
%   distribution widens while no feedback arrives.
%
%   Inputs:
%     trials  1-by-T vector of trial indices (x-axis).
%     grid    1-by-G query grid the densities were evaluated on (y-axis).
%     dens    G-by-T matrix; column t is the density over 'grid' at trial t.
%
%   Name-value options:
%     FigName / Title / YLabel   figure and axes text.
%     TrueLine   1-by-T true latent target overlaid as a black dashed line; [].
%     PredLine   1-by-T model prediction overlaid as a white line; [].
%     MissingTrials  1-by-M trial indices to mark (e.g. NaN feedback) with faint
%                    vertical guides; [] for none.
%     BlockEdges     1-by-K contingency-switch trials drawn as vertical guides.
    arguments
        trials (1, :) double
        grid (1, :) double
        dens double
        opts.FigName (1, :) char = 'Density evolution'
        opts.Title (1, :) char = 'Marginal density over trials'
        opts.YLabel (1, :) char = 'State value'
        opts.TrueLine double = []
        opts.PredLine double = []
        opts.MissingTrials (1, :) double = []
        opts.BlockEdges (1, :) double = []
    end

    fig = figure('Name', opts.FigName);
    ax = axes(fig); hold(ax, 'on');
    colormap(ax, parula);

    imagesc(ax, trials, grid, dens);
    set(ax, 'YDir', 'normal');
    axis(ax, 'tight');
    colorbar(ax);

    yl = [grid(1), grid(end)];
    for e = opts.MissingTrials
        plot(ax, [e e], yl, ':', 'Color', [0.85 0.85 0.85 0.6], 'HandleVisibility', 'off');
    end
    for e = opts.BlockEdges
        plot(ax, [e e], yl, 'Color', [0.5 0.5 0.5 0.5], 'HandleVisibility', 'off');
    end

    handles = gobjects(0); labels = {};
    if ~isempty(opts.TrueLine)
        h = plot(ax, trials, opts.TrueLine, 'k--', 'LineWidth', 1.3);
        handles(end+1) = h; labels{end+1} = 'True'; %#ok<AGROW>
    end
    if ~isempty(opts.PredLine)
        h = plot(ax, trials, opts.PredLine, '-', 'Color', [1 1 1], 'LineWidth', 1.3);
        handles(end+1) = h; labels{end+1} = 'Predicted'; %#ok<AGROW>
    end

    xlabel(ax, 'Trial'); ylabel(ax, opts.YLabel);
    title(ax, opts.Title);
    ylim(ax, yl);
    if ~isempty(handles)
        legend(ax, handles, labels, 'Location', 'best', 'TextColor', [0.1 0.1 0.1]);
    end
end
