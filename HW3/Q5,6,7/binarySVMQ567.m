
clear;
load MNIST_digit_data
[weight,confMat,imgs] = MultiwaySVM(images_train,labels_train,images_test,labels_test);
confMat=normr(confMat);
averageAccuracy = trace(confMat)/10;
disp(averageAccuracy)
figure();
hold on
for i=1:10   
im = reshape(images_test(imgs(i,1), :), [28 28]);
subplot(2,5,i),imshow(im);
title(strcat('T ',num2str(i-1),'-','P',num2str(imgs(i,3))));
end 
hold off

function [weight,conf,img] = MultiwaySVM(image_train,label_train,image_test,label_test)
    weight = zeros(10,size(image_train,2));
    for i=1:10
        img_train=image_train;
        lbl_train=label_train;
        lbl_train(lbl_train~=i-1)=-1;
        lbl_train(lbl_train==i-1)=1;
        weight(i,:) = SVM(img_train,lbl_train,img_train,lbl_train',0,0.3);
    end
    
    conf = zeros (10,10);
    img = zeros(10,3);
    for t=1:size(image_test,1)
        result = image_test(t,:)*weight';
        [sorted, index]= sort(result,'descend');
        conf((label_test(t)+1),index(1))=conf((label_test(t)+1),index(1))+1;
        if ((index(1)-1)~=label_test(t))
            if(img(label_test(t)+1,2)>sorted(1))
            img(label_test(t)+1,1)=t;
            img(label_test(t)+1,2)=sorted(1);
            img(label_test(t)+1,3)=index(1)-1;

            end
        end
    end 
    
end

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