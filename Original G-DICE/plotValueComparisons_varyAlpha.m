close all
clear all
clc

colors = {'b' 'r' 'g' 'c' 'm' 'k' 'y'};

directory = 'results\';

files = {'crossEntropySearch_numNodes=13_Nk=300_Ns=30_alpha=0p1_bestValue=14p4427',...
    'crossEntropySearch_numNodes=13_Nk=300_Ns=30_alpha=0p2_bestValue=14p4418',...
    'crossEntropySearch_numNodes=13_Nk=300_Ns=30_alpha=0p3_bestValue=11p3835',...
    'crossEntropySearch_numNodes=13_Nk=300_Ns=30_alpha=0p5_bestValue=10p0664',...
    'crossEntropySearch_numNodes=13_Nk=300_Ns=30_alpha=0p8_bestValue=5p0056'};

fHandle = figure('position',[200 200 800 400]);

numSamples = 50;
values=zeros(numSamples,1);

alphas = zeros(length(files),1);
plotHandlesFile = zeros(length(files),1);
idxFile = 1;
for file = files
    load(cell2mat([directory file]));
    plotHandlesFile(idxFile) = plotValueWithErrBars(fHandle, allValues, N_s, colors(idxFile));
    alphas(idxFile) = alpha
    idxFile = idxFile+1;
end


for idx = 1:length(alphas)
   legendEntries{idx} = ['\alpha = ' num2str(alphas(idx))] 
end

legend(plotHandlesFile, legendEntries,'location','eastoutside')
xlabel('Iteration')
ylabel('Value')
grid on

set(gcf,'PaperPositionMode','auto')
print('-depsc2','plotValueComparisons_varyAlpha.eps')