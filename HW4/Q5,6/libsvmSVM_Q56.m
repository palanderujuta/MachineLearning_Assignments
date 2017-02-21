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
 
%%finding the centered data 
data_train=bsxfun(@minus,images_train,mean(images_train));
data_test=bsxfun(@minus,images_test,mean(images_test));
accn=svmFun(data_train,data_test, labels_train,labels_test,50); 
disp(accn);

dimensions=[2, 5, 10, 20, 30 ,50 ,70 ,100, 150, 200, 250, 300, 400, 500];
acc = zeros(1,size(dimensions,2));
 
for i=1:size(dimensions,2)
    acc(i)=svmFun(data_train,data_test, labels_train,labels_test,dimensions(i));
 end;   
 plot(dimensions,acc);
 xlabel('dimensions');
 ylabel('accuracy');

function [acc]=svmFun(data_train,data_test,labels_train, labels_test,dim)
    [U,S,V] = svds(data_train,dim);
    newdata=data_train * V;
    newdata1=data_test * V;
    model = svmtrain(labels_train, newdata); %train the data
    [predict_label, accuracy, dec_values] = svmpredict(labels_test, newdata1, model);
    acc=accuracy(1);
    disp(accuracy);
end