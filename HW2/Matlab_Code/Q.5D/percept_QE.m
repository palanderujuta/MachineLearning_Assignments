clear;

load MNIST_digit_data
    indices = find(labels_train(:, 1) == 1 | labels_train(:, 1) == 6);
    sample = indices(randsample(1:10000,500),:)  ;
    images_train = images_train(sample, :);
    labels_train = labels_train(sample);

    indices_test = find(labels_test(:, 1) == 1 | labels_test(:, 1) == 6);
    sample_test = indices_test(randsample(1:2093,500),:)  ;
    images_test = images_test(sample_test, :);
    labels_test = labels_test(sample_test);

    W1=zeros(1,784);
   perceptronFun(images_train,labels_train,images_test,labels_test,W1);
    

function perceptronFun(images_train, labels_train,images_test,labels_test,W1)%, images_test, labels_test,iterations)
    %accr=zeros(1,2500);
    %itr=zeros(1,2500);
    %c=1;
    %W1=zeros(1,784);
    for m=1:5
   
    %disp('in m');
    for i=1:500
        %disp('in i');
        
        W=W1';
        if(images_train(i,:)*W>0)
            y_hat=1;
        else
            y_hat=-1;
        end
        actual_train=labels_train(i);
        if actual_train==1
           
            actual_train=1;
        else
           
            actual_train=-1;
        end
        
        if actual_train==y_hat
           
        else
            
            W1=W1+actual_train.*images_train(i,:);
            
        end
        
        W=W1';
       
        %disp(W);
       
    end
     C = W;
        C(C < 0) = 0;
        B = W;
        B(B > 0) = 0;
        figure();
        imshow([reshape(C, [28 28]),reshape(-1.*B, [28 28])])
    end
end



