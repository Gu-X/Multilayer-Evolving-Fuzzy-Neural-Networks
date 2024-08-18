function [Ye,MN,tt]=MEFNN2(data0,y0,data1,mode,Wo,iter,lr)
Input1.Sys=[];
Input2.Sys=[];
Input1.lr=lr;
Input2.lr=lr;
[L,W]=size(data0);
if strcmp(mode,'C')==1
    CL=length(unique(y0));
    Y0=full(ind2vec(y0',CL))';
    tic
    for jj=1:1:iter
        seq=randperm(L);
        for ii=seq
            Input1.data=data0(ii,:);Input1.YS=Wo*CL;
            [Input1]=EFSTra_Forward(Input1);
            Input2.data=Input1.Ye;Input2.YS=CL;
            [Input2]=EFSTra_Forward(Input2);
            Grad=(Y0(ii,:)-Input2.Ye);
            Input2.data=Input1.Ye;Input2.Grad=Grad;Input2.RG=1;
            [Input2]=EFSTra_Backward(Input2);
            Input1.data=data0(ii,:);Input1.Grad=Input2.Grad;Input1.RG=0;
            [Input1]=EFSTra_Backward(Input1);
        end
    end
    tt(1)=toc;
    MN(1,1)=Input1.Sys.ModelNumber;
    MN(2,1)=Input2.Sys.ModelNumber;
    [L,~]=size(data1);
    Ye=zeros(L,CL);
    tic
    for ii=1:1:L
        Input1.data=data1(ii,:);
        [Input1]=EFSTes(Input1);
        Input2.data=Input1.Ye;
        [Input2]=EFSTes(Input2);
        Ye(ii,:)=Input2.Ye;
    end
    tt(2,1)=toc;
    [~,Ye]=max(Ye,[],2);
end
if strcmp(mode,'R')==1
    CL=1;
    Y0=y0;
    tic
    for jj=1:1:iter
        seq=randperm(L);
        for ii=seq
            Input1.data=data0(ii,:);Input1.YS=Wo*CL;
            [Input1]=EFSTra_Forward(Input1);
            Input2.data=Input1.Ye;Input2.YS=CL;
            [Input2]=EFSTra_Forward(Input2);
            Grad=(Y0(ii,:)-Input2.Ye);
            Input2.data=Input1.Ye;Input2.Grad=Grad;Input2.RG=1;
            [Input2]=EFSTra_Backward(Input2);
            Input1.data=data0(ii,:);Input1.Grad=Input2.Grad;Input1.RG=0;
            [Input1]=EFSTra_Backward(Input1);
        end
    end
    tt(1)=toc;
    MN(1,1)=Input1.Sys.ModelNumber;
    MN(2,1)=Input2.Sys.ModelNumber;
    [L,~]=size(data1);
    Ye=zeros(L,CL);
    tic
    for ii=1:1:L
        Input1.data=data1(ii,:);
        [Input1]=EFSTes(Input1);
        Input2.data=Input1.Ye;
        [Input2]=EFSTes(Input2);
        Ye(ii,:)=Input2.Ye;
    end
    tt(2,1)=toc;
end
end
function [Output]=EFSTes(Input)
datain=Input.data;
Output.Sys=Input.Sys;
L1=Input.Sys.L;
prototypes=Input.Sys.prototypes;
local_delta=Input.Sys.local_delta;
Global_mean=Input.Sys.Global_mean;
Global_X=Input.Sys.Global_X;
ModelNumber=Input.Sys.ModelNumber;
A=Input.Sys.A;
W=Input.Sys.W;
CL=Input.Sys.CL;
Global_mean1=Global_mean.*L1./(L1+1)+datain./(L1+1);
Global_X1=Global_X.*L1./(L1+1)+datain.^2./(L1+1);
Global_Delta1=abs(Global_X1-Global_mean1.^2);
[centerlambda,LocalDensity,~]=firingstrength(datain,ModelNumber,prototypes,local_delta,Global_Delta1,W);
[Ye1,~]=OutputGeneration(datain,A,centerlambda,LocalDensity,ModelNumber,CL);
Output.Ye=Ye1;
    %%
end
function [Output]=EFSTra_Backward(Input)
datain=Input.data;
Grad=Input.Grad;
Output.Sys=Input.Sys;
ModelNumber=Input.Sys.ModelNumber;
centerlambda=Input.Sys.centerlambda;
LocalDensity=Input.Sys.LocalDensity;
prototypes=Input.Sys.prototypes;
Global_Delta1=Input.Sys.Global_Delta1;
W=Input.Sys.W;
YeL=Input.Sys.YeL;
lr=Input.lr;
A=Input.Sys.A;
X=[1,datain];
%%
sigd=YeL.*(1-YeL);
if Input.RG==1
    D1=sum(LocalDensity);
    C0=2*(prototypes-datain)./repmat(Global_Delta1,1,W); %NxW
    C0(isnan(C0))=0;
    C00=LocalDensity'*C0; 
    C1=repmat(centerlambda,1,W).*C0-centerlambda/D1*C00;
    Xgrad=zeros(1,W);
    for jj=1:1:ModelNumber
        Xgrad=Xgrad+centerlambda(jj)*(Grad.*sigd(jj,:))*A(:,2:end,jj)+Grad*(YeL(jj,:)'*C1(jj,:));
    	A(:,:,jj)=A(:,:,jj)+lr*centerlambda(jj)*(Grad.*sigd(jj,:))'*X;
    end
    Output.Grad=Xgrad;
end
if Input.RG==0
    for jj=1:1:ModelNumber
        A(:,:,jj)=A(:,:,jj)+lr*centerlambda(jj)*(Grad.*sigd(jj,:))'*X;
    end
end
Output.Sys.A=A;
Output.lr=lr;
end
function [Output]=EFSTra_Forward(Input)
datain=Input.data;
CL=Input.YS;
if isempty(Input.Sys)
    W=length(datain);
    Output.Sys.center=datain;
    Output.Sys.prototypes=datain;
    Output.Sys.local_X=datain.^2;
    Output.Sys.local_delta=zeros(1,W);
    Output.Sys.Global_mean=datain;
    Output.Sys.Global_X=datain.^2;
    Output.Sys.Support=1;
    Output.Sys.ModelNumber=1;
    Output.Sys.A=(round(rand(CL,W+1,1)))/(W+1);%round(rand(CL,W+1,1))/(W+1);
    Output.Ye=1./(1+exp(-1*[1,datain]*Output.Sys.A'));
    Output.Sys.YeL=Output.Ye;
    Output.Sys.L=1;
    Output.Sys.W=W;
    Output.Sys.CL=CL;
    Output.Sys.threshold1=exp(-3);
    Output.lr=Input.lr;
    Output.Sys.centerlambda=1;
    Output.Sys.LocalDensity=1;
    Output.Sys.Global_Delta1=0;
else
    Input.Sys.L=Input.Sys.L+1;
    Output.Sys=Input.Sys;
    ii=Input.Sys.L;
    lr=Input.lr;
    center=Input.Sys.center;
    prototypes=Input.Sys.prototypes;
    local_X=Input.Sys.local_X;
    local_delta=Input.Sys.local_delta;
    Global_mean=Input.Sys.Global_mean;
    Global_X=Input.Sys.Global_X;
    Support=Input.Sys.Support;
    ModelNumber=Input.Sys.ModelNumber;
    threshold1=Input.Sys.threshold1;
    A=Input.Sys.A;
    W=Input.Sys.W;
    CL=Input.Sys.CL;
    Global_mean=Global_mean.*(ii-1)./ii+datain./ii;
    Global_X=Global_X.*(ii-1)./ii+datain.^2./ii;
    Global_Delta=abs(Global_X-Global_mean.^2);
    [~,LocalDensity,Global_Delta1]=firingstrength(datain,ModelNumber,prototypes,local_delta,Global_Delta,W);
    LocalDensity(isnan(LocalDensity))=1;
    centerlambda=LocalDensity./sum(LocalDensity);
    [Ye1,~]=OutputGeneration(datain,A,centerlambda,LocalDensity,ModelNumber,CL);
    if max(LocalDensity)<threshold1
        %% new_cloud_add
        ModelNumber=ModelNumber+1;
        center(ModelNumber,:)=datain;
        prototypes(ModelNumber,:)=datain;
        local_X(:,:,ModelNumber)=datain.^2;
        Support=[Support,1];
        local_delta(:,:,ModelNumber)=zeros(1,W);
        Global_Delta1(ModelNumber,1)=sum(Global_Delta,'all')./2;
        A(:,:,ModelNumber)=(round(rand(CL,W+1,1)))/(W+1);%round(rand(CL,W+1,1))/(W+1);
        LocalDensity=[LocalDensity;1];
    else
        %% local_parameters_update
        [~,label0]=max(LocalDensity);
        Support(label0)=Support(label0)+1;
        center(label0,:)=((Support(label0)-1)*center(label0,:)+datain)/Support(label0);
        local_X(:,:,label0)=((Support(label0)-1)*local_X(:,:,label0)+datain.^2)/Support(label0);
        local_delta(:,:,label0)=abs(local_X(:,:,label0)-center(label0,:).^2);
        Global_Delta1(label0,1)=(sum(Global_Delta+local_delta(:,:,label0),'all')./2);
        LocalDensity(label0,1)=exp(-1*sum((datain-prototypes(label0,:)).^2,'all')/Global_Delta1(label0,1));
    end
    LocalDensity(isnan(LocalDensity))=1;
    centerlambda=LocalDensity./sum(LocalDensity);
    [~,YeL]=OutputGeneration(datain,A,centerlambda,LocalDensity,ModelNumber,CL);
    %%
    Output.lr=lr;
    Output.Sys.centerlambda=centerlambda;
    Output.Sys.LocalDensity=LocalDensity;
    Output.Sys.center=center;
    Output.Sys.prototypes=prototypes;
    Output.Sys.local_X=local_X;
    Output.Sys.local_delta=local_delta;
    Output.Sys.Global_mean=Global_mean;
    Output.Sys.Global_Delta1=Global_Delta1;
    Output.Sys.Global_X=Global_X;
    Output.Sys.Support=Support;
    Output.Sys.ModelNumber=ModelNumber;
    Output.Sys.A=A;
    Output.Ye=Ye1;
    Output.Sys.YeL=YeL;
end
end
function [centerlambda,LocalDensity,Global_Delta1]=firingstrength(datain,ModelNumber,center,local_delta,Global_Delta,W)
LocalDensity=zeros(ModelNumber,1);
Global_Delta1=zeros(ModelNumber,1);
for ii=1:1:ModelNumber
    datain1=sum((datain-center(ii,:)).^2,'all');
    Global_Delta1(ii,1)=sum((Global_Delta+local_delta(:,:,ii))/2,'all');
    LocalDensity(ii,1)=exp(-1*datain1./Global_Delta1(ii,1));
end
LocalDensity(isnan(LocalDensity))=1;
centerlambda=LocalDensity./sum(LocalDensity);
end
function [Ye,YeL]=OutputGeneration(datain,A,centerlambda,LocalDensity,ModelNumber,CL)
Ye=zeros(1,CL);
YeL=zeros(ModelNumber,CL);
for ii=1:1:ModelNumber
    YeL(ii,:)=1./(1+exp(-1*[1,datain]*A(:,:,ii)'));
    Ye=Ye+YeL(ii,:)*centerlambda(ii);
end
end
