function fig = densityLines(grid, mapContainer, opts)
%COINVIZ.DENSITYLINES 1-D per-context density line plot.
%   fig = coinviz.densityLines(grid, mapContainer, opts) plots one density
%   curve per context for the scalar model. mapContainer is the containers.Map
%   returned by state_given_context_probability / state_feedback_given_context_
%   probability (keyed by global context label). The novel (not-yet-seen)
%   context density can be overlaid via the NovelDensity option.
%
%   Inputs:
%     grid          1-by-G query grid the densities were evaluated on.
%     mapContainer  containers.Map (double key -> 1-by-G density row).
%
%   Name-value options:
%     FigName / Title / XLabel   figure and axes text.
%     NovelDensity  1-by-G density of the novel context (drawn dashed); [] omits.
    arguments
        grid (1, :) double
        mapContainer
        opts.FigName (1, :) char = 'Per-context densities'
        opts.Title (1, :) char = 'Per-context densities'
        opts.XLabel (1, :) char = 'Value'
        opts.NovelDensity double = []
    end

    keys = cell2mat(mapContainer.keys);
    cols = coinviz.palette(max(numel(keys), 1) + 1);

    fig = figure('Name', opts.FigName);
    ax = axes(fig); hold(ax, 'on');
    labels = {};
    for k = 1:numel(keys)
        plot(ax, grid, mapContainer(keys(k)), 'Color', cols(k, :), 'LineWidth', 1.4);
        labels{end+1} = sprintf('Context %d', keys(k)); %#ok<AGROW>
    end
    if ~isempty(opts.NovelDensity)
        plot(ax, grid, opts.NovelDensity, '--', 'Color', [0.4 0.4 0.4], 'LineWidth', 1.4);
        labels{end+1} = 'Novel context';
    end
    xlabel(ax, opts.XLabel); ylabel(ax, 'Density');
    title(ax, opts.Title);
    if ~isempty(labels)
        legend(ax, labels, 'Location', 'best');
    end
end
