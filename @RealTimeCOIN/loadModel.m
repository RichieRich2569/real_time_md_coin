function loadModel(obj, filename)
    S = load(filename);
    if isfield(S, 'model')
        restoreSerializableState(obj, S.model);
    else
        % Backward compatibility with the first implementation.
        if isfield(S, 'particles')
            error('RealTimeCOIN:LegacySave', ...
                ['This save file contains legacy cell particles. ', ...
                 'Create a fresh model or resave with the vectorized implementation.']);
        end
    end
end
