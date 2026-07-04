function fig = contextDensityEvolution(trials, grid, ctxDens, opts)
%COINVIZ.CONTEXTDENSITYEVOLUTION Composite mixed-colour state|context evolution.
%   fig = coinviz.contextDensityEvolution(trials, grid, ctxDens, opts) renders a
%   single heat-map (trial on x, state grid on y) of the per-context posterior
%   state density state_given_context_probability recorded once per trial. Each
%   context is tinted its own palette colour and the densities are additively
%   blended into one truecolor image, so where contexts overlap their colours
%   mix - a compact way to watch which context "owns" which part of state space
%   as learning proceeds. Scalar-model only: a multi-dimensional per-trial
%   density is 2-D and cannot be shown as a single trial x state image.
%
%   Inputs:
%     trials   1-by-T vector of trial indices (x-axis).
%     grid     1-by-G query grid the densities were evaluated on (y-axis).
%     ctxDens  containers.Map keyed by global context label (double) -> G-by-T
%              density matrix (column t = that context's density at trial t).
%
%   Name-value options:
%     FigName / Title / YLabel   figure and axes text.
%     NovelDens   G-by-T novel-context density, drawn as a grey tint and
%                 labelled "Novel context"; [] to omit.
%     TrueLine    1-by-T true latent target overlaid as a white dashed line; [].
%     BlockEdges  1-by-K contingency-switch trials drawn as vertical guides.
    arguments
        trials (1, :) double
        grid (1, :) double
        ctxDens
        opts.FigName (1, :) char = 'State|context evolution'
        opts.Title (1, :) char = 'Per-context posterior state density over trials'
        opts.YLabel (1, :) char = 'State value'
        opts.NovelDens double = []
        opts.TrueLine double = []
        opts.BlockEdges (1, :) double = []
    end

    keys = sort(cell2mat(ctxDens.keys));
    G = numel(grid); T = numel(trials);

    % Assemble the tint list: one row per context (coloured by its global label
    % so the colour matches contextBars/densityLines), plus the novel context.
    tintColors = coinviz.palette(max([keys, 1]));
    layers = {};      % each: struct(D = G-by-T, color = 1x3, label)
    for k = 1:numel(keys)
        L = keys(k);
        layers{end+1} = struct('D', ctxDens(L), ...
            'color', tintColors(L, :), 'label', sprintf('Context %d', L)); %#ok<AGROW>
    end
    if ~isempty(opts.NovelDens)
        layers{end+1} = struct('D', opts.NovelDens, ...
            'color', [0.6 0.6 0.6], 'label', 'Novel context');
    end
    assert(~isempty(layers), 'coinviz:contextDensityEvolution:noData', ...
        'Provide at least one context density.');

    % Joint normalisation so brightness is comparable across contexts, then
    % additive colour mixing (overlaps brighten toward a blended hue).
    M = 0;
    for i = 1:numel(layers)
        M = max(M, max(layers{i}.D(:)));
    end
    if M <= 0, M = 1; end

    rgb = zeros(G, T, 3);
    for i = 1:numel(layers)
        w = min(max(layers{i}.D, 0) ./ M, 1);          % G-by-T in [0,1]
        for ch = 1:3
            rgb(:, :, ch) = rgb(:, :, ch) + w .* layers{i}.color(ch);
        end
    end
    rgb = min(rgb, 1);

    fig = figure('Name', opts.FigName);
    ax = axes(fig); hold(ax, 'on');
    image(ax, trials, grid, rgb);
    set(ax, 'YDir', 'normal');
    axis(ax, 'tight');

    yl = [grid(1), grid(end)];
    for e = opts.BlockEdges
        plot(ax, [e e], yl, 'Color', [1 1 1 0.4], 'HandleVisibility', 'off');
    end
    if ~isempty(opts.TrueLine)
        plot(ax, trials, opts.TrueLine, 'w--', 'LineWidth', 1.3, 'HandleVisibility', 'off');
    end

    % Colour legend via off-screen marker proxies (image() carries no legend).
    handles = gobjects(1, numel(layers));
    labels = cell(1, numel(layers));
    for i = 1:numel(layers)
        handles(i) = plot(ax, NaN, NaN, 's', 'MarkerSize', 10, ...
            'MarkerFaceColor', layers{i}.color, 'MarkerEdgeColor', 'none');
        labels{i} = layers{i}.label;
    end
    legend(ax, handles, labels, 'Location', 'eastoutside');

    xlabel(ax, 'Trial'); ylabel(ax, opts.YLabel);
    title(ax, opts.Title);
    ylim(ax, yl);
end
