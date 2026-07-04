function fig = contextBars(prevProbs, postProbs, opts)
%COINVIZ.CONTEXTBARS Context probabilities over trials.
%   fig = coinviz.contextBars(prevProbs, postProbs, opts) plots the evolution
%   of the context probabilities across trials. Pass the predicted (before the
%   observation, "prev") probabilities and/or the responsibilities (after the
%   observation, "post"); each argument is a T-by-C matrix (trials-by-context
%   slots). When both are supplied they are drawn in a stacked pair of
%   subplots; supplying [] for one draws only the other.
%
%   Name-value options:
%     FigName    figure title bar text.
%     Mode       'area' (default, stacked) or 'line'.
%     PrevTitle / PostTitle   subplot titles.
%     NovelContext  logical (default false). When true the final retained
%                   context column is the novel (not-yet-instantiated) slot; it
%                   is relabelled "Novel context" and drawn in grey, matching
%                   densityLines. Only pass true when it is proper - i.e. the
%                   trailing used slot really is the K+1 novel context (compare
%                   the number of retained columns with diagnostics().K).
    arguments
        prevProbs double = []
        postProbs double = []
        opts.FigName (1, :) char = 'Context probabilities'
        opts.Mode (1, :) char {mustBeMember(opts.Mode, {'area', 'line'})} = 'area'
        opts.PrevTitle (1, :) char = 'Predicted context probabilities (before observation)'
        opts.PostTitle (1, :) char = 'Context responsibilities (after observation)'
        opts.NovelContext (1, 1) logical = false
    end

    panels = {};
    titles = {};
    ylabels = {};
    if ~isempty(prevProbs)
        panels{end+1} = prevProbs; titles{end+1} = opts.PrevTitle; ylabels{end+1} = 'Predicted prob';
    end
    if ~isempty(postProbs)
        panels{end+1} = postProbs; titles{end+1} = opts.PostTitle; ylabels{end+1} = 'Responsibility';
    end
    assert(~isempty(panels), 'coinviz:contextBars:noData', 'Provide prevProbs and/or postProbs.');

    % Trim trailing all-zero context slots so the legend is not cluttered by
    % never-instantiated contexts, but keep at least one column.
    used = false(1, max(cellfun(@(P) size(P, 2), panels)));
    for k = 1:numel(panels)
        used(1:size(panels{k}, 2)) = used(1:size(panels{k}, 2)) | any(panels{k} > 0, 1);
    end
    C = max(find(used, 1, 'last'), 1); %#ok<MXFND>

    fig = figure('Name', opts.FigName);
    nP = numel(panels);
    for k = 1:nP
        ax = subplot(nP, 1, k);
        P = panels{k};
        C_k = min(C, size(P, 2));
        T = size(P, 1);
        cols = coinviz.palette(C_k);
        legendLabels = arrayfun(@(c) sprintf('Context %d', c), 1:C_k, ...
            'UniformOutput', false);
        % The trailing slot is the novel (not-yet-instantiated) context: recolour
        % it grey and relabel it so the legend does not read it as a real context.
        markNovel = opts.NovelContext && C_k >= 1;
        if markNovel
            cols(C_k, :) = [0.6 0.6 0.6];
            legendLabels{C_k} = 'Novel context';
        end
        if strcmp(opts.Mode, 'area')
            h = area(ax, 1:T, P(:, 1:C_k));
            for c = 1:numel(h)
                h(c).FaceColor = cols(c, :);
                h(c).EdgeColor = 'none';
            end
        else
            hold(ax, 'on');
            for c = 1:C_k
                plot(ax, 1:T, P(:, c), 'Color', cols(c, :), 'LineWidth', 1.2);
            end
        end
        ylim(ax, [0 1]);
        xlabel(ax, 'Trial'); ylabel(ax, ylabels{k});
        title(ax, titles{k});
        legend(ax, legendLabels, 'Location', 'southoutside');
    end
end
