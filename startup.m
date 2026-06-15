rootDir = fileparts(mfilename('fullpath'));

lightspeedDir = fullfile(rootDir, 'lightspeed');
npbayesDir = fullfile(rootDir, 'npbayes-r21');

if exist(lightspeedDir, 'dir')
    addpath(genpath(lightspeedDir));
end

if exist(npbayesDir, 'dir')
    addpath(genpath(npbayesDir));
end
