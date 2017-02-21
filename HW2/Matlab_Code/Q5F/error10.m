clear;

load MNIST_digit_data
    indices = find(labels_train(:, 1) == 1 | labels_train(:, 1) == 6);
    sample = indices(randsample(1:12660,12660),:)  ;
    sample1=sample(randsample(1:12660,1266),:);
    images_train = images_train(sample, :);
     labels_trainerror=labels_train(sample1);

    labels_train = labels_train(sample);
   
    indices_test = find(labels_test(:, 1) == 1 | labels_test(:, 1) == 6);
    sample_test = indices_test(randsample(1:2093,1000),:)  ;
    images_test = images_test(sample_test, :);
    labels_test = labels_test(sample_test);

    W1=zeros(1,784);
    [accuracy,iter,k]=perceptronFun(images_train,labels_train,images_test,labels_test,W1,sample1,sample);
    plot(iter,accuracy);
    xlabel('iterations');
    ylabel('accuracy');
    disp(k);
   

function [accr,itr,k] =  perceptronFun(images_train, labels_train,images_test,labels_test,W1,sample1,sample)%, images_test, labels_test,iterations)
   
     accr=zeros(1,63300);
    itr=zeros(1,63300);
    c=1;
    k=1;
    %W1=zeros(1,784);
    for m=1:5
   
    %disp('in m');
    for i=1:12660
        %disp('in i');
        
        W=W1';
        
        
            if(images_train(i,:)*W>0)
                y_hat=1;
            else
                y_hat=-1;
            end
   
       
        actual_train=labels_train(i);
       % for j=1:200
       %     if labels_trainerror(j)==i
       %         flag=1;
       %         break;
       %     else
       %         flag=0;
       %     end;
       % end
             
        if ismember(sample(i),sample1)
            k=k+1;
            if actual_train==1
           
                actual_train=-1;
            else
           
                actual_train=1;
            end
            flag=0;
        else
            
            if actual_train==1
           
                actual_train=1;
            else
           
                actual_train=-1;
            end 
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
            
            if actual_test==1
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

