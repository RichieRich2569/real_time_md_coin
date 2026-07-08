function loadModel(obj, filename)
%LOADMODEL Restore model state from a .mat file written by saveModel.
%   loadModel(obj, filename) reads the "model" snapshot stored in filename and
%   overwrites the state of obj with it, so obj continues from the saved
%   trial/posterior. The file must have been produced by saveModel.
%
%   Errors:
%     RealTimeCOIN:LegacySave    - the file is from the original cell-particle
%                                  implementation and cannot be restored.
%     RealTimeCOIN:InvalidSave   - the file has neither a "model" nor a
%                                  "particles" field (empty/malformed save).
%
%   See also SAVEMODEL.
    arguments
        obj (1, 1) RealTimeCOIN
        filename (1, :) char {mustBeNonempty}
    end
    S = load(filename);
    if isfield(S, 'model')
        restoreSerializableState(obj, S.model);
    elseif isfield(S, 'particles')
        % Backward compatibility with the first implementation.
        error('RealTimeCOIN:LegacySave', ...
            ['This save file contains legacy cell particles. ', ...
             'Create a fresh model or resave with the vectorized implementation.']);
    else
        error('RealTimeCOIN:InvalidSave', ...
            ['File "%s" is not a valid RealTimeCOIN save: it contains neither ', ...
             'a "model" nor a "particles" field.'], filename);
    end
end
