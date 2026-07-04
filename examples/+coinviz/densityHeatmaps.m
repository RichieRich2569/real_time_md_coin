function fig = densityHeatmaps(coin, D, ctxKeys, ctxTargets, opts)
%COINVIZ.DENSITYHEATMAPS 2-D per-context density heat-map grid (incl. novel).
%   fig = coinviz.densityHeatmaps(coin, D, ctxKeys, ctxTargets, opts) draws one
%   imagesc panel per instantiated context plus a final panel for the novel
%   (not-yet-instantiated) context, for a 2-D (state_dim == 2) model. One call
%   serves both the feedback-space and state-space figures via opts.Space.
%
%   Inputs:
%     coin        the RealTimeCOIN object (queried on a fresh grid per panel).
%     D           its diagnostics() struct (for per-context means and bias).
%     ctxKeys     global context labels to draw (e.g. from the density Map keys).
%     ctxTargets  2-by-numel(ctxKeys) true target of each context's cue.
%
%   Name-value options:
%     Space      'feedback' -> p(y | context) = state + bias + noise, or
%                'state'    -> p(state | context) (latent, no bias). Feedback is
%                the default; the true target marks the peak in feedback space,
%                and target-minus-bias marks it in state space.
%     CueLabels  optional vector (indexed by global context) of the dominant cue
%                per context, used only for panel titles.
%     Grid       density grid resolution per axis (default 120).
    arguments
        coin (1, 1) RealTimeCOIN
        D struct
        ctxKeys (1, :) double
        ctxTargets (2, :) double
        opts.Space (1, :) char {mustBeMember(opts.Space, {'feedback', 'state'})} = 'feedback'
        opts.CueLabels double = []
        opts.Grid (1, 1) double = 120
    end
    assert(coin.state_dim == 2, 'coinviz:densityHeatmaps:not2D', ...
        'densityHeatmaps requires a 2-D (state_dim == 2) model.');

    isFb = strcmp(opts.Space, 'feedback');
    if isFb
        figName = 'Per-context feedback densities';
        figTitle = 'Per-context predictive feedback densities  p(y \mid context)';
        axlabel = 'Feedback';
    else
        figName = 'Per-context state densities';
        figTitle = 'Per-context posterior state densities  p(state \mid context)';
        axlabel = 'State';
    end

    ng = opts.Grid;
    nInst = numel(ctxKeys);
    nPanels = nInst + 1;                    % +1 for the novel context
    nCols = min(3, nPanels);
    nRows = ceil(nPanels / nCols);

    fig = figure('Name', figName);
    colormap(fig, parula);

    for i = 1:nInst
        key = ctxKeys(i);
        bias = D.bias(:, key);
        if isFb
            mu = D.state_mean(:, key) + bias;      % feedback (observation) mean
            mk = ctxTargets(:, i);                 % target lands on the peak
            mkName = 'true target';
        else
            mu = D.state_mean(:, key);             % latent state mean
            mk = ctxTargets(:, i) - bias;          % target - bias lands on the peak
            mkName = 'true target - bias';
        end

        m = 0.10;                                  % margin so blob + marker fit
        axX = linspace(min(mu(1), mk(1)) - m, max(mu(1), mk(1)) + m, ng);
        axY = linspace(min(mu(2), mk(2)) - m, max(mu(2), mk(2)) + m, ng);
        [GX, GY] = meshgrid(axX, axY);
        pts = [GX(:)'; GY(:)'];
        if isFb
            dmap = coin.state_feedback_given_context_probability(pts);
        else
            dmap = coin.state_given_context_probability(pts);
        end
        Z = reshape(dmap(key), size(GX));

        ax = subplot(nRows, nCols, i);
        imagesc(ax, axX, axY, Z); set(ax, 'YDir', 'normal'); hold(ax, 'on');
        plot(ax, mk(1), mk(2), 'rp', 'MarkerSize', 14, 'MarkerFaceColor', 'r');
        plot(ax, mu(1), mu(2), 'w+', 'MarkerSize', 10, 'LineWidth', 1.5);
        xlabel(ax, sprintf('%s dim 1', axlabel));
        ylabel(ax, sprintf('%s dim 2', axlabel));
        colorbar(ax);
        if isempty(opts.CueLabels)
            title(ax, sprintf('Context %d', key));
        else
            title(ax, sprintf('Context %d (cue %d)', key, opts.CueLabels(key)));
        end
        legend(ax, mkName, 'density mean', 'Location', 'best');
        axis(ax, 'tight');
    end

    % Novel context: autoscale a grid to where its density is appreciable.
    broad = linspace(-1.5, 1.5, ng);
    [BX, BY] = meshgrid(broad, broad);
    if isFb
        novelFcn = @(P) coin.novel_state_feedback_probability(P);
    else
        novelFcn = @(P) coin.novel_state_probability(P);
    end
    nd = reshape(novelFcn([BX(:)'; BY(:)']), size(BX));
    if max(nd(:)) > 0
        mask = nd > 0.02 * max(nd(:));
        xr = broad(any(mask, 1)); yr = broad(any(mask, 2));
        pad = 0.1;
        axX = linspace(min(xr) - pad, max(xr) + pad, ng);
        axY = linspace(min(yr) - pad, max(yr) + pad, ng);
        [GX, GY] = meshgrid(axX, axY);
        Zn = reshape(novelFcn([GX(:)'; GY(:)']), size(GX));
    else
        axX = broad; axY = broad; Zn = nd;
    end
    ax = subplot(nRows, nCols, nPanels);
    imagesc(ax, axX, axY, Zn); set(ax, 'YDir', 'normal');
    xlabel(ax, sprintf('%s dim 1', axlabel));
    ylabel(ax, sprintf('%s dim 2', axlabel));
    colorbar(ax);
    title(ax, 'Novel context');
    axis(ax, 'tight');

    sgtitle(fig, figTitle);
end
