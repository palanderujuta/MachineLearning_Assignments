clear
load MNIST_digit_data
whos

%%% randomly permute data points
rand('seed', 1); %%just to make all random sequences on all computers the same.
inds = randperm(size(images_train, 1));
images_train = images_train(inds, :);
labels_train = labels_train(inds, :);

inds = randperm(size(images_test, 1));
images_test = images_test(inds, :);
labels_test = labels_test(inds, :);
images_train = images_train(1:1000, :);
labels_train = labels_train(1:1000, :);
 
t=zeros(1000,10);
data_train=bsxfun(@minus,images_train,mean(images_train));

data_test=bsxfun(@minus,images_test,mean(images_test));


for i=1:1000
    for j=1:10
      if j==labels_train(i) +1
          t(i,j)=1;
      else 
          t(i,j)=0;
      end
    end;
end;

t = t';
dimensions=[2, 5, 10, 20, 30 ,50 ,70 ,100, 150, 200, 250, 300, 400, 500];
acc = zeros(1,size(dimensions,2));
 
for i=1:size(dimensions,2)
    acc(i)=neuralnet(data_train,data_test, t,labels_test,dimensions(i));
end;   

plot(dimensions,acc);
xlabel('dimensions');
ylabel('accuracy');


function [accuracy]=neuralnet(data_train,data_test,t,labels_test,dim)
[U,S,V] = svds(data_train,dim);
data_train=data_train * V;
 x=data_train';
data_test=data_test * V; 
x1=data_test';
net = patternnet(100);
net = train(net,x,t);
view(net)
y = net(x1);
[~,predicted]=max(y);

cnt=0;
for i=1:10000
    if predicted(i)-1 == labels_test(i)
        cnt=cnt+1;
    end
end
accuracy=(cnt/10000)*100;
disp(accuracy)
%%perf = perform(net,y,t);
end