     %% load data and create vector matrix
     clc;close all;clear all;
     filename ='demoFile.mat';
     load(filename);
     R=3;
     %% create the tensor and matraix
     K = size(Net,2);
     [I, J] = size(Net{1});
     X = zeros(I,J,K);
      for i = 1:K
        X(:,:,i) = Net{i};
      end
     X = sptensor(X);
   
    %% Process 
     disp('Process started...');
     [labels_i, A]=SHOCDALL.SHOCD(X,L,R);
     disp('Process ended...');
     %% for non-overlapping communities
     labels=SHOCDALL.permuteLabels(labels_i,true_idx);
     n=SHOCDALL.nmi(labels,true_idx);
     p=SHOCDALL.purity(labels,true_idx);
     fprintf('NMI[0-1]: %1.3f and Accuracy[0-1]: %1.3f \n',n,p);

     %% for finding overlapping communities.
%       t=0.1; % threshold where A_i,j > t
%      [USER , PredictedCommunity]=find(A>t); 
%       X=[USER , PredictedCommunity];
%       result=sortrows(X,1);
     
     
     
     
     