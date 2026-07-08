%STARTUP Compile and add the RealTimeCOIN third-party dependencies to the path.
%   Running STARTUP prepares a fresh clone for use with the offline COIN
%   reference and the validation scripts. It:
%
%     1. Adds the bundled 'lightspeed' and 'npbayes-r21' libraries (and their
%        subfolders) to the MATLAB path.
%     2. Compiles their MEX components in place by invoking each library's own
%        build routine ('install_lightspeed' and 'make').
%
%   The two libraries are only required by the offline reference (COIN.m) and
%   some validation scripts; core RealTimeCOIN use only needs the repo root on
%   the path. Run STARTUP once per fresh clone or new machine.
%
%   Paths are resolved relative to this file (not the current working folder),
%   so STARTUP may be run from anywhere. The current folder is restored on exit,
%   including if a build errors, via an onCleanup guard.
%
%   See also ADDPATH, MEX.

% Resolve dependency locations relative to this script so STARTUP is
% independent of the caller's current working folder.
rootDir = fileparts(mfilename('fullpath'));
lightspeedDir = fullfile(rootDir, 'lightspeed');
npbayesDir = fullfile(rootDir, 'npbayes-r21');

% Each library builds "in place": its build routine must run with the library
% folder as the current folder. Capture the caller's folder once and restore it
% on any exit path (normal or error) with a single cleanup guard.
originalDir = pwd;
cleanupObj = onCleanup(@() cd(originalDir)); % restores folder on exit

if exist(lightspeedDir, 'dir')
    addpath(genpath(lightspeedDir));

    cd(lightspeedDir);
    try
        install_lightspeed;
    catch buildErr
        error('startup:LightspeedBuildFailed', ...
            "Failed to build the 'lightspeed' MEX components in '%s': %s", ...
            lightspeedDir, buildErr.message);
    end
    cd(originalDir);
end

if exist(npbayesDir, 'dir')
    addpath(genpath(npbayesDir));

    cd(npbayesDir);
    try
        make;
    catch buildErr
        error('startup:NpbayesBuildFailed', ...
            "Failed to build the 'npbayes-r21' MEX components in '%s': %s", ...
            npbayesDir, buildErr.message);
    end
    cd(originalDir);
end
