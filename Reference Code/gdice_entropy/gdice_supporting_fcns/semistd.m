function [LowerSTD UpperSTD]=semistd(x)
% Syntax
%
% LowerSTD = semistd(x)
% [LowerSTD UpperSTD]= semistd(x)
% 
% Description
%
% LowerSTD = semistd(x) computes the sample lower semideviation 
% of x. if x is an n-by-1 vector with n observations and 1 variable, the
% function returns the lower semideviation for the variable. 
% If x is an n-by-m matrix with n observations and m variables, the
% function returns the lower semideviation for the m variables.
%
% [LowerSTD UpperSTD] = semistd(x) computes the sample lower and upper
% semideviation of x. if x is an n-by-1 vector with n
% observations and 1 variable, the function returns the lower semideviation
% for the variable. If x is an n-by-m matrix with n
% observations and m variables, the function returns the lower
% semideviation for the m variables.
%--------------------------------------------------------------------------
% Author:         Abdulaziz Alhouti
% Affiliation:    The George Washington University
% email:          abdulaziz.alhouti@gmail.com
% Last revision:  19-January-2014
%--------------------------------------------------------------------------


% determine number of input arguments
if nargin <1
    error('Matlab: semistd function: too few input arguments');
end
% determine the number of observations and varibles in x
[n m]=size(x);
% transpose x if user inputs 1-by-n instead of n-by-1
if n==1 && m>1
   x=x';
end
% determine the number of observations and varibles in x after
% transposition
[n m]=size(x);
% lower semi standard deviation
ldev=detrend(x,'constant');
ldev(ldev>0)=0;
ldevsq=ldev.^2;
lvar=sum(ldevsq)./(n-1);
LowerSTD=sqrt(lvar);
% uppur semi standard deviation
udev=detrend(x,'constant');
udev(udev<0)=0;
udevsq=udev.^2;
uvar=sum(udevsq)./(n-1);
UpperSTD=sqrt(uvar);
end