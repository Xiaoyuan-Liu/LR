DataSet=readDataSet('train_set.txt');
[Dimension,SampleSize]=size(DataSet);

Theta=zeros(17,26);
for i=1:26
    Theta(:,i)=train(DataSet,theta,i);

end
%Theta=train(DataSet,Theta,1);
disp(Theta);
TestSet=readDataSet('test_set.txt');
[Dimension,TestSampleSize]=size(TestSet);
Result=zeros(TestSampleSize,26);
for i=1:26
    Result(:,i)=test(TestSet,Theta(:,i),i);
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
disp(RESULT);
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
    T=0;
    N=0;
    for i=1:SampleSize
        Y(i,1)=(TestSet(Dimension,i)==Class);
        T=T+(TestSet(Dimension,i)==Class);
        N=N+1-(TestSet(Dimension,i)==Class);
    end
    
    for i=1:SampleSize
        H(i,1)=(H(i,1)>0.5);
    end
    %disp(H);
    R=zeros(SampleSize,1);
    error=0;
    TP=0;
    TN=0;
    FP=0;
    FN=0;
    for i=1:SampleSize
        R(i,1)=(H(i,1)==Y(i,1));
        error=error+1-(H(i,1)==Y(i,1));
        if H(i,1)==Y(i,1)
           TP=TP+H(i,1);
           TN=TN+1-H(i,1);
           continue;
        end
        FP=FP+H(i,1);
        FN=FN+1-H(i,1);
    end
    %disp(R);
    disp(TP/(TP+FP));
    disp(TP/T);
end
function [newTheta]=train(DataSet,Theta,Class)
    alpha=0.001;%步长
    
    times=0;%训练次数
    while 1
        times=times+1;
        [Theta,alpha,loss]=gradientDescent(DataSet,Theta,Class,alpha);
        %disp(times);
        %disp(alpha);
        %disp(Theta);
        disp(loss);
       if times>10000
           newTheta = Theta;%训练结束，返回参数
           break;
       end
    end
end
function [newTheta,alpha,loss]=gradientDescent(DataSet,Theta,Class,alpha)
%梯度下降
    [Dimension,SampleSize]=size(DataSet);
    X=DataSet(1:Dimension-1,:);%每列是一个sample，第1-18行是特征，第19行是类别
    Z=Theta'*X;
    H=zeros(SampleSize,1);
    
    for i=1:SampleSize
        H(i,1)=1./(1+exp(-Z(1,i)));%计算sigmoid函数
    end
    
    Y=zeros(SampleSize,1);
    for i=1:SampleSize
        Y(i,1)=(DataSet(Dimension,i)==Class);
    end
    loss=0;
    for i=1:SampleSize
        loss=loss+Y(i,1).*log(H(i,1))+(1-Y(i,1)).*log(1-H(i,1));
    end
    %cost = H-Y;
    for i=1:Dimension-1
        for j=1:SampleSize
            Theta(i,1) = Theta(i,1) - alpha.*(H(j,1)-Y(j,1)).*X(i,j);
        end
    end
    newTheta=Theta;
end

function [output]=readDataSet(fileName)
    a=importdata(fileName,',',0);
    [row,column]=size(a);
    for i=1:column-1%进行正则化，新特征值等于 （原特征值-最小特征值）/（最大特征值-最小特征值）
        max=a(1,i);
        min=max;
        for j=2:row
            if a(j,i)>max
                max=a(j,i);
            end
            if a(j,i)<min
                min=a(j,i);
            end
        end
        for j=1:row
            a(j,i)=(a(j,i)-min)./(max-min);
        end
    end
    b=ones(row,1);%x0 always 1
    output=[b,a]';%数据正则化后进行转置，这样后面处理的数据是每列是一个sample，每行是一种特征值
end