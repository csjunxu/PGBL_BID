function [Snew,Znew] = SZUpdate(Xnew,Sold,Zold,K)
% function [Dnew,Snew,Znew] = SZUpdate(Xnew,Dold,Sold,Zold,K)
%Initializing new S and Z with neighboring pacthes which have already been
%used for training in previous training rounds
%Version 1: 09/12/2009
%Version 2: 11/02/2009
%Written by Mingyuan Zhou, Duke ECE, mz1@ee.duke.edu
[P,N]=size(Xnew);
[U_1,S_1,V_1] = svd(full(Xnew),'econ');
S = zeros(N,K);
S(:,1:size(V_1,2)) = V_1;
Z = true(N,K);
S = S.*Z;
Snew=[Sold;S];
Znew=[Zold;Z];
end
%  %% FASTER Version 2: 03/11/2015
% [P,N]=size(Xnew);
% S = Sold(1:N,1:K);
% Z = true(N,K);
% S = S.*Z;
% Snew=[Sold;S];
% Znew=[Zold;Z];
% end