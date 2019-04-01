DataSet=readDataSet('train_set.txt');
[Dimension,SampleSize]=size(DataSet);

Theta=ones(17,1,26);
for i=1:26
    theta=ones(17,1);
    theta=train(DataSet,theta,i);
    Theta(:,:,i)=theta;
end
%Theta=train(DataSet,Theta,1);
disp(Theta);
TestSet=readDataSet('test_set.txt');
[Dimension,TestSampleSize]=size(TestSet);
Result=zeros(TestSampleSize,26);
for i=1:26
    Result(:,i)=test(TestSet,Theta(:,:,i),i);
end
%disp(Result);
RESULT=zeros(TestSampleSize,1);
for i=1:TestSampleSize
    class =1;
    for j=2:26
        if Result(i,j)>Result(i,class)
            class=j;
        end
    end
    RESULT(i,1)=class;
end
%disp(RESULT);
error=0;
for i=1:TestSampleSize
   error=error+1-(RESULT(i,1)==TestSet(18,i)); 
end
disp(error);

function [Result]=test(TestSet,Theta,Class)
    [Dimension,SampleSize]=size(TestSet);
    X=TestSet(1:Dimension-1,:);
    Z=X'*Theta;
    %H=zeros(1,SampleSize);
    H=zeros(SampleSize,1);
    for i=1:SampleSize
        H(i,1)=1/(1+exp(-Z(i,1)));
    end
    Result=H;
    Y=zeros(SampleSize,1);
    for i=1:SampleSize
        Y(i,1)=(TestSet(Dimension,i)==Class);
    end
    
    for i=1:SampleSize
        H(i,1)=(H(i,1)>0.5);
    end
    %disp(H);
    R=zeros(SampleSize,1);
    error=0;
    for i=1:SampleSize
        R(i,1)=(H(i,1)==Y(i,1));
        error=error+1-(H(i,1)==Y(i,1));
    end
    %disp(R);
    disp(error);
end
function [newTheta]=train(DataSet,Theta,Class)
    alpha=0.0005;
    oldcost=0;
    times=0;
    while 1
        times=times+1;
        oldTheta=Theta;
        [Theta,alpha,oldcost]=gradientDescent(DataSet,Theta,Class,alpha,oldcost);
        disp(times);
        %disp(alpha);
        disp(Theta);
       if times>2000
           newTheta = Theta;
           break;
       end
    end
end

function [newTheta,alpha,cost]=gradientDescent(DataSet,Theta,Class,alpha,oldcost)
%ÌÝ¶ÈÏÂ½µ
    [Dimension,SampleSize]=size(DataSet);
    X=DataSet(1:Dimension-1,:);
    Z=X'*Theta;
    H=zeros(SampleSize,1);
    for i=1:SampleSize
        H(i,1)=1/(1+exp(-Z(i,1)));
    end
    Y=zeros(SampleSize,1);
    for i=1:SampleSize
        Y(i,1)=(DataSet(Dimension,i)==Class);
    end

    cost = H-Y;
    Theta = Theta - alpha.*X*cost;

    newTheta=Theta;
end
function [output]=readDataSet(fileName)
    a=importdata(fileName,',',0);
    [row,column]=size(a);
    regularization=a(:,1:(column-1));
    for i=1:column-1
        max=regularization(1,i);
        min=max;
        for j=2:row
            if regularization(j,i)>max
                max=regularization(j,i);
            end
            if regularization(j,i)<min
                min=regularization(j,i);
            end
        end
        for j=1:row
            regularization(j,i)=(regularization(j,i)-min)./(max-min);
        end
    end
    b=ones(row,1);%x0 always 1
    output=[b,a]';
end