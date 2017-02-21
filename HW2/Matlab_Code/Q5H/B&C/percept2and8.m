clear;

load MNIST_digit_data
    indices = find(labels_train(:, 1) == 2 | labels_train(:, 1) == 8);
    sample = indices(randsample(1:10000,1000),:)  ;
    images_train = images_train(sample, :);
    labels_train = labels_train(sample);

    indices_test = find(labels_test(:, 1) == 2 | labels_test(:, 1) == 8);
    sample_test = indices_test(randsample(1:2000,1000),:)  ;
    images_test = images_test(sample_test, :);
    labels_test = labels_test(sample_test);

    W1=zeros(1,784);
    [accuracy,iter]=perceptronFun(images_train,labels_train,images_test,labels_test,W1);
    plot(iter,accuracy);
    xlabel('iterations');
    ylabel('accuracy');
   

function [accr,itr] =  perceptronFun(images_train, labels_train,images_test,labels_test,W1)%, images_test, labels_test,iterations)
    accr=zeros(1,5000);
    itr=zeros(1,5000);
    c=1;
    %W1=zeros(1,784);
    for m=1:5
   
    %disp('in m');
    for i=1:1000
        %disp('in i');
        
        W=W1';
        if(images_train(i,:)*W>0)
            y_hat=1;
        else
            y_hat=-1;
        end
        actual_train=labels_train(i);
        if actual_train==2
           
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
        [accuracy]=perceptronTest(images_test,labels_test,W); 
        accr(c)=accuracy;
   
        itr(c)=c;
        c=c+1;
    end
    end
end


function [cnt] =  perceptronTest(images_test, labels_test,W)%, images_test, labels_test,iterations)
    cnt=0;
    %disp(W);
         for j=1:1000 
            if(images_test(j,:)*W>0)
                y_hat_test=1;
            else
                y_hat_test=-1;
            end
            
            actual_test=labels_test(j);
            
            if actual_test==2
                actual_test=1;
            else
                actual_test=-1;
            end
      
            if (actual_test==y_hat_test)
                cnt=cnt+1;
                
            else
                %disp('wrong prediction');
            end
         end
         
        disp(cnt);
        cnt=cnt/1000;
 
end

