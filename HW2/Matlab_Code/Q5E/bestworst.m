clear;

load MNIST_digit_data
    indices = find(labels_train(:, 1) == 1 | labels_train(:, 1) == 6);
    sample = indices(randsample(1:10000,1000),:)  ;
    images_train = images_train(sample, :);
    labels_train = labels_train(sample);

    indices1 = find(labels_test(:, 1) == 1 );%| labels_train(:, 1) == 6);
    sample1 = indices1(randsample(1:1135,1135),:)  ;
    images_test1 = images_test(sample1, :);
    labels_test1 = labels_test(sample1);
    indices2 = find(labels_test(:, 1) == 6 );%| labels_train(:, 1) == 6);
    sample2 = indices2(randsample(1:958,958),:)  ;
    images_test2 = images_test(sample2, :);
    labels_test2 = labels_test(sample2);
    images_test=[images_test1;images_test2];
    labels_test=[labels_test1;labels_test2];

    %indices_test = find(labels_test(:, 1) == 1 | labels_test(:, 1) == 6);
    %sample_test = indices_test(randsample(1:2093,500),:)  ;
    %images_test = images_test(sample_test, :);
    %labels_test = labels_test(sample_test);

    W1=zeros(1,784);
    [W]=perceptronFun(images_train,labels_train,images_test,labels_test,W1);
    arr1=zeros(1,1135);
    for i=1:1135
        arr1(i)=images_test1(i,:) * W;
    end
    
    %newarr1=[arr1',labels_test1];
    [~,val1]=sort(arr1);
    
     figure();
    for i=1:20    
        %[X(i),Y(i)]=imread(images_test1(val1(i),:));
        subplot(4,6,i),imshow((reshape(images_test1(val1(i),:),[28 28])));
    end
    
    figure();
    k=1;
    for i=1116:1135
       % figure();
       subplot(4,5,k),imshow(reshape(images_test1(val1(i),:),[28 28]))
       k=k+1;
    end
    
    arr6=zeros(1,958);
    
    
    for i=1:958
        arr6(i)=images_test2(i,:) * W;
    end
    
     [~,val6]=sort(arr6);
     figure();
    for i=1:20
       % figure();
       subplot(4,5,i),imshow(reshape(images_test2(val6(i),:),[28 28]))
    end
    figure();
    k=1;
    for i=939:958
      %  figure();
      subplot(4,5,k),imshow(reshape(images_test2(val6(i),:),[28 28]))
      k=k+1;
    end
    %arr6=images_test2 * W';
   

function [W] =  perceptronFun(images_train, labels_train,images_test,labels_test,W1)%, images_test, labels_test,iterations)
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
        %C = W;
        %C(C < 0) = 0;
        %B = W;
        %B(B > 0) = 0;
        %figure();
        %imshow([reshape(C, [28 28]),reshape(-1.*B, [28 28])])
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

