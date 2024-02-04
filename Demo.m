clear all
clc
close all
%% Example: Regression
load ExampleData_regression
%%
% data0: training data
% y0: ground truth of data0
% data1: testing data
% y1: group truth of data1
%%
mode='R' %% Regression
Wo=3;
niter=200; %% number of iteration
lr=1; %% learning rate
[Ye]=MEFNN2(data0,y0,data1,mode,Wo,niter,lr); % train a two-layer MEFNN
RMSE=sqrt(mean((Ye-y1).^2)) % root mean square error
NDEI=sqrt(mean((Ye-y1).^2))/std(y1) % non-dimensional error index

%% Example: Classification
load ExampleData_classification
mode='C' %% Classification
[Ye]=MEFNN2(data0,y0,data1,mode,Wo,niter,lr);
CL=length(unique(y0));
CM=confusionmat(y1,Ye) % confusion matrix
Accuracy=sum(CM.*eye(CL),'all')/sum(CM,'all') % classification accuracy
