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
 
model = svmtrain(labels_train, images_train); %train the data
[predict_label, accuracy, dec_values] = svmpredict(labels_test, images_test, model); % test the training data

disp('Calculated accuracy');
%%calculating the accuracy
cnt=0;
 for i=1:10000
    if labels_test(i)==predict_label(i)
        cnt=cnt+1;
    end;
 end
 acc=(cnt/10000)*100;
 disp(acc);
 %%end of calculating the accuracy
 
 %%finding the mean and subtracting it
 data=bsxfun(@minus,images_train,mean(images_train));
 %%end of finding the mean and subtracting it
 
 dimensions=logspace(log10(1),log10(500),50);
 meanFun = zeros(1,size(dimensions,2));
 for i=1:size(dimensions,2)
     [U,S,V] = svds(data,round(dimensions(i)));
     newdata=data * V;
     newdata1=newdata *V';
     meanFun(i) = immse(data, newdata1);
 end;   
 plot(dimensions,meanFun)
 xlabel('Dimensions');
 ylabel('Mean squared error');
 
 
 %%visualization for dimension 50
 [U,S,V] = svds(data,50);
 V=V';
 for i = 1 : 10 
    subplot(2,5,i);
    imagesc(reshape(V(i, :), [28 28]));
 end
 