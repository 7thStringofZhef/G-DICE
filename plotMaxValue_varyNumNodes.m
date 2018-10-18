close all
clear all
clc

colors = {'b' 'r' 'g' 'c' 'm' 'k' 'y'};

directory = 'results\';

files = {'crossEntropySearch_numNodes=3_Nk=300_Ns=30_alpha=0p2_bestValue=1p548',...
    'crossEntropySearch_numNodes=4_Nk=300_Ns=30_alpha=0p2_bestValue=5p246',...
    'crossEntropySearch_numNodes=5_Nk=300_Ns=30_alpha=0p2_bestValue=10p3091',...
    'crossEntropySearch_numNodes=7_Nk=300_Ns=30_alpha=0p2_bestValue=11p8535',...
    'crossEntropySearch_numNodes=8_Nk=300_Ns=30_alpha=0p2_bestValue=13p3594',...
    'crossEntropySearch_numNodes=10_Nk=300_Ns=30_alpha=0p2_bestValue=14p2232',...
    'crossEntropySearch_numNodes=13_Nk=300_Ns=30_alpha=0p2_bestValue=14p4418'};

fHandle = figure;

alphas = zeros(length(files),1);
plotHandles = zeros(length(files),1);

nIterations = [25 50 100 200 300];
maxValuesVsIteration = zeros(length(files),length(nIterations));

idx = 1;
for file = files
    load(cell2mat([directory file]));
    parfor idxSamples = 1:20
        newValues(idxSamples) = evalPolicy(mGraphPolicyController);
    end
    maxVal_mean(idx) = mean(newValues);
    maxVal_std(idx) = std(newValues);
    numNodesList(idx) = numNodes;
    idx = idx+1
end

%%
%plot of full dataset
close all
fHandle = figure('position',[200 200 800 400]);
errorbar(numNodesList, maxVal_mean,maxVal_std,'k')
xlabel('N_n')
ylabel('Max value')
set(gcf,'PaperPositionMode','auto')
grid on
print('-depsc2','plotMaxValue_varyNumNodes.eps')

%plot if cut-off occurs
% figure
% for idxIteration = 1:length(nIterations)
%    plot(alphas, maxValuesVsIteration(:,idxIteration))
%    hold on
% end