function fig = stateTrace(t, observed, trueVal, predicted, blockEdges, opts)
%COINVIZ.STATETRACE Per-dimension observed / true / predicted trace over trials.
%   fig = coinviz.stateTrace(t, observed, trueVal, predicted, blockEdges, opts)
%   draws one stacked subplot per state dimension comparing the noisy
%   observations, the true (latent) target and the model's prediction as a
%   function of trial. It handles the scalar case (1-by-T rows) and the
%   multi-dimensional case (N-by-T rows) uniformly, so the same helper serves
%   both notebooks.
%
%   Positional inputs (any of the data matrices may be empty to omit that
%   series):
%     t          1-by-T vector of trial indices.
%     observed   N-by-T noisy observations   (plotted as grey x markers).
%     trueVal    N-by-T true latent targets  (plotted as a black dashed line).
%     predicted  N-by-T model prediction     (plotted as a coloured line).
%     blockEdges 1-by-K trial indices at which the contingency switches
%                (drawn as faint vertical lines); [] for none.
%
%   Name-value options:
%     FigName        figure title bar text.
%     DimLabel       y-axis label stem (default 'State dim').
%     Band           N-by-T predictive standard deviation; when supplied a
%                    +/-1 sigma shaded band is drawn around 'predicted'.
%     PredictedName  legend label for the prediction series.
%     ObservedName / TrueName   legend labels for the other two series.
    arguments
        t (1, :) double
        observed double = []
        trueVal double = []
        predicted double = []
        blockEdges (1, :) double = []
        opts.FigName (1, :) char = 'State trace'
        opts.DimLabel (1, :) char = 'State dim'
        opts.Band double = []
        opts.PredictedName (1, :) char = 'Predicted'
        opts.ObservedName (1, :) char = 'Observed'
        opts.TrueName (1, :) char = 'True'
    end

    N = max([size(observed, 1), size(trueVal, 1), size(predicted, 1), 1]);
    cols = coinviz.palette(3);
    predColor = cols(1, :);
    obsColor  = [0.7 0.7 0.7];

    fig = figure('Name', opts.FigName);
    for dim = 1:N
        ax = subplot(N, 1, dim);
        hold(ax, 'on');
        handles = gobjects(0);
        labels = {};

        % Predictive +/-1 sigma band (drawn first so it sits behind the lines).
        if ~isempty(opts.Band) && ~isempty(predicted)
            hi = predicted(dim, :) + opts.Band(dim, :);
            lo = predicted(dim, :) - opts.Band(dim, :);
            fill([t, fliplr(t)], [hi, fliplr(lo)], predColor, ...
                'FaceAlpha', 0.15, 'EdgeColor', 'none', 'Parent', ax);
        end
        if ~isempty(observed)
            h = plot(ax, t, observed(dim, :), 'x', 'Color', obsColor, 'MarkerSize', 4);
            handles(end+1) = h; labels{end+1} = opts.ObservedName; %#ok<AGROW>
        end
        if ~isempty(trueVal)
            h = plot(ax, t, trueVal(dim, :), 'k--', 'LineWidth', 1.2);
            handles(end+1) = h; labels{end+1} = opts.TrueName; %#ok<AGROW>
        end
        if ~isempty(predicted)
            h = plot(ax, t, predicted(dim, :), '-', 'Color', predColor, 'LineWidth', 1.4);
            handles(end+1) = h; labels{end+1} = opts.PredictedName; %#ok<AGROW>
        end

        % Contingency-switch guides.
        yl = ylim(ax);
        for e = blockEdges
            plot(ax, [e e], yl, 'Color', [0.5 0.5 0.5 0.4], 'HandleVisibility', 'off');
        end
        ylim(ax, yl);

        xlabel(ax, 'Trial');
        if N == 1
            ylabel(ax, 'State value');
        else
            ylabel(ax, sprintf('%s %d', opts.DimLabel, dim));
        end
        if ~isempty(handles)
            legend(ax, handles, labels, 'Location', 'best');
        end
    end
end
