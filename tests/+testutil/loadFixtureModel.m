function coin = loadFixtureModel(template, D, alignmentSeed)
%TESTUTIL.LOADFIXTUREMODEL Build a RealTimeCOIN from a hand-crafted particle state.
%   coin = testutil.loadFixtureModel(template, D) serialises template's public
%   properties (except Trial) together with the particle-state struct D through a
%   temporary .mat file and loads them into a fresh RealTimeCOIN, yielding a model
%   whose internal state is fully controlled by the caller.
%
%   coin = testutil.loadFixtureModel(template, D, alignmentSeed) additionally
%   seeds the persisted alignment cache. The temporary file is removed via
%   onCleanup even if loadModel errors.
if nargin < 3
    alignmentSeed = [];
end
model = struct();
model.properties = struct();
props = properties(template);
for i = 1:numel(props)
    if ~strcmp(props{i}, 'Trial')
        model.properties.(props{i}) = template.(props{i});
    end
end
model.D = D;
model.pending_q = [];
model.trial = 1;
model.cue_values = 1;
model.state_version = 1;
model.alignment_seed = alignmentSeed;

tmpfile = [tempname, '.mat'];
save(tmpfile, 'model');
cleanup = onCleanup(@() testutil.deleteTempFile(tmpfile)); %#ok<NASGU>
coin = RealTimeCOIN('num_particles', 1);
coin.loadModel(tmpfile);
end
