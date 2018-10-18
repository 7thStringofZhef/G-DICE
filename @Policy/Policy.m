classdef Policy < handle
   properties
       table; 
   end
   
   methods
       function obj = Policy(table)
           if nargin > 0
               obj.table = table;
           end
       end
       
       function nextTMAIdx = getNextTMAIdx(obj, curTMAIdx, curXeIdx)
%           nextTMAIdx = obj.table(curTMAIdx, curXeIdx); %for tree-based
%             nextTMAIdx = obj.
       end
   end
end