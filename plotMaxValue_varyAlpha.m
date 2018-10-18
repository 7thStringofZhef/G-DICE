close all
clear all
clc

colors = {'b' 'r' 'g' 'c' 'm' 'k' 'y'};

directory = 'results\';

files = {'crossEntropySearch_numNodes=13_Nk=300_Ns=30_alpha=0p1_bestValue=14p4427',...
    'crossEntropySearch_numNodes=13_Nk=300_Ns=30_alpha=0p2_bestValue=14p4418',...
    'crossEntropySearch_numNodes=13_Nk=300_Ns=30_alpha=0p3_bestValue=11p3835',...
    'crossEntropySearch_numNodes=13_Nk=300_Ns=30_alpha=0p4_bestValue=9p0765',...
    'crossEntropySearch_numNodes=13_Nk=300_Ns=30_alpha=0p5_bestValue=10p0664',...
    'crossEntropySearch_numNodes=13_Nk=300_Ns=30_alpha=0p8_bestValue=5p0056'};

fHandle = figure;

alphas = zeros(length(files),1);
plotHandles = zeros(length(files),1);

nIterations = [25 50 100 200 300];
maxValuesVsIteration = zeros(length(files),length(nIterations));

idx = 1;
for file = files
    load(cell2mat([directory file]));
    parfor idxSamples = 1:20
        values(idxSamples) = evalPolicy(mGraphPolicyController);
    end
    maxVal_mean(idx) = mean(values);
    maxVal_std(idx) = std(values);
    maxValuesVsIteration(idx,:) = allValues(nIterations.*N_s)
    alphas(idx) = alpha;
    idx = idx+1;
end

%%
%plot of full dataset
close all
fHandle = figure('position',[200 200 800 400]);
errorbar(alphas, maxVal_mean,maxVal_std,'k')
xlabel('\alpha')
ylabel('Max value')
set(gcf,'PaperPositionMode','auto')
grid on
print('-depsc2','plotMaxValue_varyAlpha.eps')

%plot if cut-off occurs
% figure
% for idxIteration = 1:length(nIterations)
%    plot(alphas, maxValuesVsIteration(:,idxIteration))
%    hold on
% end