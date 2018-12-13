clc
clear all
close all

dom_static = Domain();
% dom.constructTree();

%some observations are impossible, for those we put -1 as next TMA idx
%for observations that are possible, we put 0
                  %  1   2   3   4   5   6   7   8   9  10  11  12  13
                  %000 110 120 130 210 220 230 111 121 131 211 221 231
refPolicyTable = [  -1   0   0   0   0   0   0   0   0   0   0   0   0; %1 -goto b1 (at b1)
                    -1   0   0   0   0   0   0   0   0   0   0   0   0; %2 -goto b2 (at b2)
                     0  -1  -1  -1  -1  -1  -1  -1  -1  -1  -1  -1  -1;%3 -goto r (at r)
                     0  -1  -1  -1  -1  -1  -1  -1  -1  -1  -1  -1  -1;%4 -goto d1 (at d1)
                     0  -1  -1  -1  -1  -1  -1  -1  -1  -1  -1  -1  -1;%5 -goto d2 (at d2)
                     0  -1  -1  -1  -1  -1  -1  -1  -1  -1  -1  -1  -1;%6 -joint goto d1 (at d1)
                     0  -1  -1  -1  -1  -1  -1  -1  -1  -1  -1  -1  -1;%7 -joint goto d2 (at d2)
                    -1   0   0   0  -1  -1  -1   0   0   0  -1  -1  -1;%8 -pickup pkg (at b1 or b2) %this is a special case where the observations MUST COME BEFOREA NEW PKG IS GENERATED!!
                    -1  -1  -1  -1  -1  -1  -1  -1  -1  -1   0   0  -1;%9 -joint pickup pkg (at b1 or b2)
                     0  -1  -1  -1  -1  -1  -1  -1  -1  -1  -1  -1  -1;%10 -put down pkg (at d1 or d2)
                     0  -1  -1  -1  -1  -1  -1  -1  -1  -1  -1  -1  -1;%11 -joint put down pkg (at d1 or d2) 
                     0  -1  -1  -1  -1  -1  -1  -1  -1  -1  -1  -1  -1;%12 -place on truck (at r)
                     0   0   0   0   0   0   0   0   0   0   0   0   0;%13 -wait (anywhere)
                 ];

allValues = [0];
curBestValue = [0];             
curBestPolicy = refPolicyTable;

clc
curBestValue(end)
exhaustivePolicyTable = refPolicyTable;

numPolicies = 1;

for idxRow = 1:size(exhaustivePolicyTable,1)
    for idxCol = 1:size(exhaustivePolicyTable,2)
        if (exhaustivePolicyTable(idxRow,idxCol)~=-1)
            allowableChildTMAIdxs = dom_static.TMAs(idxRow).allowableChildTMAIdxs;
            allowableChildTMAIdxsSingleAgent = allowableChildTMAIdxs(allowableChildTMAIdxs ~= 6 & allowableChildTMAIdxs ~= 7 & allowableChildTMAIdxs ~= 9 & allowableChildTMAIdxs ~= 11 );
            
            %cannot jointly deliver to rendezvous, or when alone
            if (idxCol >=2 && idxCol <=10 || idxCol == 13)
                numPolicies = numPolicies*length(allowableChildTMAIdxsSingleAgent);
                for idxAllowableTMAs = 1:length(allowableChildTMAIdxsSingleAgent)
                    exhaustivePolicyTable(idxRow,idxCol) = allowableChildTMAIdxsSingleAgent(idxAllowableTMAs);
                end
            else
                numPolicies = numPolicies*length(allowableChildTMAIdxs);
                for idxAllowableTMAs = 1:length(allowableChildTMAIdxs)
                    exhaustivePolicyTable(idxRow,idxCol) = allowableChildTMAIdxs(idxAllowableTMAs);
                end
            end
        end
        
%         newValue = evalPolicy(exhaustivePolicyTable);
% 
%         allValues(end+1) = newValue;
%         
%         if (newValue > curBestValue(end))
%             curBestValue(end+1) = newValue;
%             curBestPolicy = exhaustivePolicyTable;
%         else
%             curBestValue(end+1) = curBestValue(end);
%         end
    end
end

numPolicies
numPolicies*1/60/60/24/365


    

%%
plot(allValues,'rx')
hold on
plot(curBestValue)
grid on
axis([0 1000 0 4])
legend('Current Value','Best Value')
xlabel('Iteration')
ylabel('Policy Value')
set(gcf,'PaperPositionMode','auto')
print('-depsc','-zbuffer','-r200','policyIterationExhaustive.eps')
