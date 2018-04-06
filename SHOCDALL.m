classdef SHOCDALL
    methods(Static)

function [labels_i , A]=SHOCD(X,L,R)
%Ekta Gujral and Vagelis Papalexakis - University of
%California,Riverside,CA.Computer Science (2017-2018)
%SMACD: Semi-supervised Multi-Aspect Community Detection
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% INPUT:
% X: Tensor or sptensor. [I x J x K]
% L: Matrix with p% of known communities.[I x R]
% R: Number of communities .
% OUTPUT
% labels_i: Final community assignment for matrix A. By default gives hard
% assignment of community
% A : factorized matrix A for finding overlapping communities.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
D= size(X);
I=D(1);J=D(2);K=D(3);
Atrue{1}=L; %  matrix
Atrue{2}=zeros(I,R);  
Atrue{3}=zeros(K,R);
lambda=0.01;
%lambda = SHOCDALL.SelSPF(X,L,R);
[A, B ,C ,outmatrices] = SHOCDALL.SHOCD_internal(X,Atrue,R,lambda);
 %% find the labels
 labels_i = zeros(size(A,1),1);
 [labels_i]= SHOCDALL.getLabels(I,labels_i,A,R);
end

function [A B C outmatrices] = SHOCD_internal(X,matrices,F,lambda)
%ALS algorithm for Non-Negative Sparse Coupled Matrix-Tensor Factorization
%Example: [A B C outmatrices] = NNSCMTF(X,matrices,F,lambda)
%   X: input tensor
%   matrices{1}: coupled matrix on 1st dimension
%   matrices{2}: coupled matrix on 2nd dimension
%   matrices{3}: coupled matrix on 3rd dimension
%   F: number of components
%   lambda: sparsity penalty regularizing parameter
%   [A B C]: Factors of tensor X
%   outmatrices{i}: i-th latent factor of coupled matrix sharing i-th
%   dimension
% Implementation of the SHOCD Algorithm

MAXNUMITER = 1000; 
SMALLNUMBER = 10^-5;
s = size(X); I=s(1);J=s(2);K=s(3);
mat = {};
for i = 1:length(matrices)
   mat{i} = matrices{i}; 
end
for i = length(matrices)+1 : 3
    mat{i} = [];
end
matrices = mat;
X1 = matrices{1}; X2 = matrices{2}; X3 = matrices{3};

if size(X1,1)~=I
   error('X1 must have I rows'); 
end
if size(X2,1)~=J
   error('X2 must have J rows'); 
end
if size(X3,1)~=K
   error('X3 must have K rows'); 
end

D = sprand(size(X1,2),F,1); E = sprand(size(X2,2),F,1); G = sprand(size(X3,2),F,1);
UA = (sptenmat(X,1,'bc')); UA = sparse(UA.subs(:,1), UA.subs(:,2),UA.vals,size(UA,1),size(UA,2));UA = UA';
UB = (sptenmat(X,2,'bc'));  UB = sparse(UB.subs(:,1), UB.subs(:,2),UB.vals,size(UB,1),size(UB,2));UB = UB';
UC = (sptenmat(X,3,'bc'));  UC = sparse(UC.subs(:,1), UC.subs(:,2),UC.vals,size(UC,1),size(UC,2));UC = UC';
        
A = sprand(I,F,1) ; B = sprand(J,F,1); C = sprand(K,F,1);

cost = 0;
if ~isempty(X1)
    cost = cost + norm(X1 - A*D','fro')^2;
end
if ~isempty(X2)
    cost = cost + norm(X2 - B*E','fro')^2;
end
if ~isempty(X3)
    cost = cost + norm(X3 - C*G','fro')^2;
end

cost = cost + norm(UA - khatrirao(B,C)*A' ,'fro')^2 + lambda*( sum(sum(abs(A))) + sum(sum(abs(B))) + sum(sum(abs(C))) + sum(sum(abs(D))) + sum(sum(abs(E))) + sum(sum(abs(G))));
costold = 2*cost;
it = 0;
while abs((cost-costold)/costold) > SMALLNUMBER && it < MAXNUMITER && cost > 10^5*eps
    
        it = it+1;
        A = SHOCDALL.NNSMREW( [UA ; X1'], [sparse(khatrirao(B,C)) ; D] ,A,lambda);
        % re-estimate B:
        B = SHOCDALL.NNSMREW( [UB ; X2'],[sparse(khatrirao(C,A)) ; E],B,lambda);
        % re-estimate C:
        C = SHOCDALL.NNSMREW([UC ; X3'], [sparse(khatrirao(A,B)); G ],C,lambda);    
        
       
        costold = cost;
        
        cost = 0;
        if ~isempty(X1)
            D = SHOCDALL.NNSMREW(X1,A,D,lambda);
            cost = cost + norm(X1 - A*D','fro')^2;
        else
            D = [];
        end
        
        if ~isempty(X2)
            E = SHOCDALL.NNSMREW(X2,B,E,lambda);
            cost = cost + norm(X2 - B*E','fro')^2;
        else
            E = [];
        end
    
        if ~isempty(X3)
            G = SHOCDALL.NNSMREW(X3,C,G,lambda);
            cost = cost + norm(X3 - C*G','fro')^2;
        else
            G = [];
        end
        
        cost = cost + norm(UA - khatrirao(B,C)*A' ,'fro')^2 + lambda*( sum(sum(abs(A))) + sum(sum(abs(B))) + sum(sum(abs(C))) + sum(sum(abs(D))) + sum(sum(abs(E))) + sum(sum(abs(G))));
        fprintf('iteration: %d cost: %12.10f diff: %.12f\n',it,cost,abs((cost-costold)/costold));

end
outmatrices{1} = sparse(D); outmatrices{2} = sparse(E); outmatrices{3} = sparse(G);
end
function B = NNSMREW(X,A,B,lambda)

[I,J]=size(X);
[I,F]=size(A);

DontShowOutput = 1;
maxit=100;
convcrit = 1e-9;
showfitafter=1;
it=0;
Oldfit=1e100;
Diff=1e100;

while Diff>convcrit && it<maxit
    it=it+1;
    for j=1:J,
        for f=1:F,
            data = X(:,j) - A*B(j,:).' + A(:,f)*B(j,f);
            alpha = A(:,f);
          
            if ((alpha.'*data - lambda/2) > 0)
                B(j,f) = (alpha.'*data - lambda/2)/(alpha.'*alpha);
            else
                B(j,f) = 0;
            end
        end
    end

    fit=norm(X-A*B.','fro')^2+lambda*sum(sum(abs(B)));
    if Oldfit < fit
%         disp(['*** bummer! *** ',num2str(Oldfit-fit)])
    end
    Diff=abs(Oldfit-fit);
    Oldfit=fit;

    if ~DontShowOutput
        % Output text
        if rem(it,showfitafter)==0
            disp([' NNSMREW Iterations:', num2str(it),' fit: ',num2str(fit)])
        end
    end
end
B = sparse(B);
end
function lambdaFinal = SelSPF(X,V,R)
%Sparsity Penalty Factor selection method
%Example: [lambda] = SelSPF(X,L,R)
%   X: input tensor
%   R: number of components
%   V: Vector matrix
%   lambda: sparsity penalty regularizing parameter
%   Implementation of the SHOCD Algorithm
%% Load parameters 
Lambdafound=false;
lambda=10^4;
found=false;
D= size(X);
I=D(1);J=D(2);K=D(3);
 %% create the tensor and matraix
Atrue{1}=V; %  matrix
Atrue{2}=zeros(I,R);  
Atrue{3}=zeros(K,R);
while(lambda>10^-4)
   [A, B, C, outmatrices] = SHOCDALL.SHOCD_internal(X,Atrue,R,lambda);
   labels_i = zeros(I,1);
   [labels_i,labelMat]= SHOCDALL.getLabels(I,labels_i,A,R);
   sumVec=sum(labelMat);
   lambda
   sumVec
   for i2=1:size(sumVec,2)
       if(sumVec(i2)==0)
           found=false;
           lambda=lambda/10;
           break;
       else
           found=true;
       end
    end
   if(found ==true && (size(sumVec,2)==R ||size(sumVec,2)==R+1) )
     
       disp(lambda);
       lambdaLow=lambda;
       lambdaHigh=lambda*10;
       rangeLambda=lambdaLow:lambdaLow:lambdaHigh;
       k=9;
       while(k>0)
            [A B C outmatrices] = SHOCDALL.SHOCD_internal(X,Atrue,R,rangeLambda(k));
            labels_i = zeros(I,1);
            [labels_i,labelMat]= SHOCDALL.getLabels(I,labels_i,A,R);
             sumVec=sum(labelMat);
			 l=rangeLambda(k)
			 sumVec
             cal=true;
             for i=1:size(sumVec,2)
                    if(sumVec(i)==0)
                        cal=false;
                        break;
                    end
             end
            if(cal==true  && (size(sumVec,2)==R ||size(sumVec,2)==R+1))
              lambdaFinal=rangeLambda(k);
              Lambdafound=true;
              lambda=10^-9;
                 break;
            else
                 lambdaFinal=rangeLambda(k);
           end
            k=k-1;
       end
       
       if(Lambdafound==false)
           lambdaFinal=lambda;
          break;
       end

   end

end
  disp('lambda found.');
  disp(lambdaFinal);
end
%% helper functions
function labels=permuteLabels(result,true_idx)

best_purity=SHOCDALL.purity(result, true_idx);
predictedM=SHOCDALL.createVector(result);
ActualM=SHOCDALL.createVector(true_idx);
x=max(true_idx);
best_perm=1:x;
for i=1:20
    y= randperm(x);
    NN=abs(predictedM(:,y));
    labels=zeros(size(NN,1),1);
    [labels]=SHOCDALL.getLabels(size(NN,1),labels,NN,x);
    p=purity(true_idx, labels);
    if(best_purity<p)
        best_perm=y;
        best_purity=p;        
      end
     if(p==1)
        best_perm=y;
        best_purity=p;  
         break;
     end
end

NN=abs(predictedM(:,best_perm));
[labels]=SHOCDALL.getLabels(size(NN,1),labels,NN,x);
end
function vectMat=createVector(label)

maxVal=max(label);
vectMat=zeros(size(label,1),maxVal);
for i=1:size(label,1)
    for j=1:maxVal
        if(label(i)==j)
           vectMat(i,j)=1;
        else
           vectMat(i,j)=0;
        end
    end
    
    
end
end

function v=purity(label, result)
I=size(label,1);
matched=0;
for i = 1:I
    if(label(i)==result(i)) 
        matched=matched+1;

    end
end
    v=matched/I;
end

function [labels_i,labelMat]=getLabels(rowcnt,labels_i,matrix,R)
labelMat=zeros(size(labels_i,1),R);
for i = 1:rowcnt
      a = matrix(i,:);
    [junk, idx] = max(a);
    matrix(i,:) = 0;
   matrix(i,idx) = junk;
    labels_i(i) = idx;
    if sum(a) == 0
        labels_i(i) = R+1;
    end
end
maxLabel=max(labels_i);
 for i=1:size(labels_i,1)
          for j=1:maxLabel
             if(labels_i(i)==j)
                 labelMat(i,j)=1;
             else
                 labelMat(i,j)=0;
             end
          end
 end
end
function v = nmi(label, result)
% Nomalized mutual information
assert(length(label) == length(result));

label = label(:);
result = result(:);

n = length(label);

label_unique = unique(label);
result_unique = unique(result);

 %check the integrity of result
if length(label_unique) ~= length(result_unique)
   error('The clustering result is not consistent with label.');
end

c = length(label_unique);

% distribution of result and label
Ml = double(repmat(label,1,c) == repmat(label_unique',n,1));
Mr = double(repmat(result,1,c) == repmat(result_unique',n,1));
Pl = sum(Ml)/n;
 Pr = sum(Mr)/n;

% entropy of Pr and Pl
Hl = -sum( Pl .* log2( Pl + eps ) );
Hr = -sum( Pr .* log2( Pr + eps ) );


M = Mr'*Ml/n;
Hlr = -sum( M(:) .* log2( M(:) + eps ) );

% mutual information
MI = Hl + Hr - Hlr;

% normalized mutual information
v = sqrt((MI/Hl)*(MI/Hr)) ;
end
    end
 end