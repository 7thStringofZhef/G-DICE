close all
clear all
clc

colors = {'b' 'r' 'g' 'c' 'm' 'k' 'y'};

directory = 'results\';

files = {'crossEntropySearch_numNodes=2_Nk=300_Ns=30_alpha=0p2_bestValue=0',...
    'crossEntropySearch_numNodes=3_Nk=300_Ns=30_alpha=0p2_bestValue=1p548',...
    'crossEntropySearch_numNodes=4_Nk=300_Ns=30_alpha=0p2_bestValue=5p246',...
    'crossEntropySearch_numNodes=5_Nk=300_Ns=30_alpha=0p2_bestValue=10p3091',...
    'crossEntropySearch_numNodes=7_Nk=300_Ns=30_alpha=0p2_bestValue=11p8535',...
    'crossEntropySearch_numNodes=10_Nk=300_Ns=30_alpha=0p2_bestValue=14p2232',...
    'crossEntropySearch_numNodes=13_Nk=300_Ns=30_alpha=0p2_bestValue=14p4418'};

fHandle = figure('position',[200 200 800 400]);

alphas = zeros(length(files),1);
numNodesList = zeros(length(files),1);
plotHandlesFiles = zeros(length(files),1);
idxFile = 1;
for file = files
    load(cell2mat([directory file]));
    plotHandlesFiles(idxFile) = plotValueWithErrBars(fHandle, allValues, N_s, colors(idxFile));
    numNodesList(idxFile) = numNodes;
    alphas(idxFile) = alpha;
    idxFile = idxFile+1
end

legend(plotHandlesFiles, cellstr(num2str(numNodesList, 'N_n = %d')),'location','eastoutside')
xlabel('Iteration')
ylabel('Value')
% title(['Alpha = ' num2str(alpha)])
grid on

set(gcf,'PaperPositionMode','auto')
print('-depsc2','plotValueComparisons_varyNumNodes.eps')