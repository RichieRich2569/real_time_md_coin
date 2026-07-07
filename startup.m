rootDir = fileparts(mfilename('fullpath'));

lightspeedDir = fullfile(rootDir, 'lightspeed');
npbayesDir = fullfile(rootDir, 'npbayes-r21');

oldDir = pwd;
cleanupObj = onCleanup(@() cd(oldDir));

if exist(lightspeedDir, 'dir')
    addpath(genpath(lightspeedDir));
    
    cd('lightspeed')
    install_lightspeed
    cd(oldDir);
end

if exist(npbayesDir, 'dir')
    addpath(genpath(npbayesDir));
    oldDir = pwd;
    cleanupObj = onCleanup(@() cd(oldDir));

    cd('npbayes-r21')
    make
    cd(oldDir)
end
