clear;

load MNIST_digit_data
ind_1 = find(labels_train==1,1000);
img_train1 = images_train(ind_1,:);
label_train1 = labels_train(ind_1,:);

ind_6 = find(labels_train==6,1000);
img_train6 = images_train(ind_6,:);
label_train6 = labels_train(ind_6,:);

label_train6=label_train6-label_train6-1;
 % indices = find(labels_train(:, 1) == 1 | labels_train(:, 1) == 6);
   % sample = indices(randsample(1:10000,1000),:)  ;
   % images_train = images_train(sample, :);
   % labels_train = labels_train(sample);

%temp_train = [img_train1;img_train6];
%temp_train_labels=[label_train1' label_train6'];
%sorted_train=temp_train;
%sorted_train_labels=temp_train_labels;

sorted_train = [img_train1;img_train6];
sorted_train_labels=[label_train1' label_train6'];

M = numel(sorted_train_labels);
randIndex = ceil(rand(1,M)*M);
train_images1 = sorted_train(randIndex,:); 
train_labels1 =sorted_train_labels(randIndex);

ind_1 = find(labels_test==1,500);
img_test1 = images_test(ind_1,:);
label_test1 = labels_test(ind_1,:);

ind_6 = find(labels_test==6,500);
img_test6 = images_test(ind_6,:);
label_test6 = labels_test(ind_6,:);
label_test6=label_test6-label_test6-1;

   % indices_test = find(labels_test(:, 1) == 1 | labels_test(:, 1) == 6);
   % sample_test = indices_test(randsample(1:2093,500),:)  ;
   % images_test = images_test(sample_test, :);
   % labels_test = labels_test(sample_test);
    

temp_test = [img_test1;img_test6];
temp_test_labels=[label_test1' label_test6'];

M = numel(temp_test_labels);
randIndex = ceil(rand(1,M)*M); 
test_images1 = temp_test(randIndex,:); 
test_labels1 =temp_test_labels(randIndex);
[weight,accuracy] = SVM(sorted_train,sorted_train_labels,test_images1,test_labels1,1,0.00001);
figure();
plot (accuracy);
title('Accuracy with sorted training and C = 0.00001')
xlabel('data');
ylabel('accuracy');

[weight,accuracy] = SVM(sorted_train,sorted_train_labels,test_images1,test_labels1,1,10);
figure();
plot (accuracy);
title('Accuracy with sorted training and C = 10')
xlabel('data');
ylabel('accuracy');
   

function [weight,accuracy] = SVM(image_train,label_train,image_test,label_test,acc,C)
bias = 0;
    weight = zeros(1,size(image_train,2));
    accuracy = zeros(size(image_train,1),1);
    iter=1;
     
    for iteration=1:iter
        for i=1:size(image_train,1)     
            lr=1/i; 
            if ((image_train(i,:)*weight'*label_train(i)+bias) < 1 )
                bias=bias - lr * (bias-C*label_train(i)*image_train(i,:));
                weight = weight - lr * (weight-C*label_train(i)*image_train(i,:));
                
            else
                bias=bias-lr*bias;
                weight = weight - lr*weight;
                
            end   
         if(acc==1)
             estimate = sign(image_test*weight');
             acca = sum(estimate' == label_test);
             accuracy(i) = acca/numel(estimate);
         end
        end   
    end
end 
