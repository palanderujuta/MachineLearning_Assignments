clear
load MNIST_digit_data
whos

images_train = images_train(1:10000, :);
images_test = images_test(1:10000, :);
labels_test(1:10000, :);
labels_train(1:10000, :);
knum=1;

%Q. 7 And B

    knum=1;
    [index,accrcy]= KNNfun(images_train, labels_train, images_test, labels_test, knum);
    disp('index matrix');
    disp(index);
    disp('accuracy data');
    disp(accrcy);
    
     knum=10;
    [index,accrcy]= KNNfun(images_train, labels_train, images_test, labels_test, knum);
    disp('index matrix');
    disp(index);
    disp('accuracy data');
    disp(accrcy);


points=[40,50,60,70,80,90,100,150,200,700];


%Q. 7 C
knum=1;
indx=1;
points=[1000,2000,3000,4000,5000,6000,7000,8000,9000,10000];
accr=zeros(10,1);
for j=points
    [ind,accr(indx)]= KNNfun(images_train(1:j, :), labels_train,images_test, labels_test, knum);
    indx=indx+1;
end;
disp('accuracy vector');
disp(accr);
plot(points, accr);
xlabel('Points');
ylabel('Accuracy');


%Q 7 d.

kArr=[1,2,3,5,10];
C=zeros(5,10);
disp(C);

for larr=kArr
    indx=1;
    accr=zeros(10,1);
    for j=points
        [ind,accr(indx)]= KNNfun(images_train(1:j, :), labels_train,images_test, labels_test, larr);
        C(larr,indx)= accr(indx);
        indx=indx+1;
    end;
    
    disp(points);
    disp(accr);
    
end;
for i=1:size(C,1)
    plot(points,C(i,:));
    hold on;
end;
xlabel('Points');
ylabel('Accuracy');
legend('k=1','k=2','k=3','k=5','k=10');


%Q 7 e.
images_train_new = images_train(1:1000, :);
images_test_new=images_train(1001:2000, :);
label_train_new = labels_train(1:1000, :);
label_test_new = labels_train(1001:2000, :);
indx=1;
accr=zeros(5,1);
kArr=[1,2,3,5,10];
for knum=kArr  
  [ind,accr(indx)]= KNNfunNew(images_train_new, label_train_new,images_test_new, label_test_new, knum);
  indx=indx+1;
end;
 plot(kArr, accr);
 xlabel('K');
 ylabel('Accuracy');
  


function [ind,acc]= KNNfun(images_train, labels_train,  images_test, labels_test, knum)
prediction=zeros(10,1);
actual=zeros(10,1);
for images_test_ind=1:1000
sum=zeros(size(images_train,1),784);
for K= 1: size(images_train,1)
        for L=1 : 784
            sum(K,L)= abs(images_train(K,L)^2-images_test(images_test_ind,L)^2);
        end;
    
end;

total=zeros(size(images_train,1),1);
sqrtmatrix=zeros(size(images_train,1),1);
for K=1:size(images_train,1)
    for L=1:784
        total(K)=total(K)+sum(K,L);
    end;
    sqrtmatrix(K)=sqrt(total(K));
end;

[sorted, indx]=sort(sqrtmatrix);
%disp(sorted);
%value = mode(sqrtmatrix);
gettingKLabels=zeros(knum,1);
for k = 1:knum

    gettingKLabels(k) = labels_train(indx(k));
end

maxNum = mode(gettingKLabels);
    sortedK=zeros(knum,1);
for K=1:knum
    sortedK(K)=sorted(K);
end;


indMatrix=zeros(knum,1);
for L=1:knum
    for K=1:size(images_train)
        if sortedK(L)==sqrtmatrix(K)
            disp('in index');
            indMatrix(L)=K;
            break;
        end;
    end;
end;

disp('ind matrix');
disp(indMatrix);
disp('at loation');
disp(K);
disp('knum');
disp(knum);

%indMatrix=zeros(knum,1);
tempArr=zeros(knum,1);
for L=1:knum   
     tempArr(L)=labels_train(indMatrix(L));    
end;

disp('tempArr');
disp(tempArr);
disp(indMatrix);
 maxnum=mode(tempArr);

disp('MaxNum');
disp(maxnum);

im = reshape(images_test(images_test_ind, :), [28 28]);
imshow(im)



if maxNum==labels_test(images_test_ind)
    prediction((maxNum)+1)=prediction((maxNum)+1)+1;
end;

disp(prediction);

actual(labels_test(images_test_ind)+1)=actual(labels_test(images_test_ind)+1)+1;
disp(actual);

end;
accuracy_vector=zeros(10,1);

for L=1:10
        if actual(L)~=0
            accuracy_vector(L)=prediction(L)/actual(L);
        end;
end;



acc=mean(accuracy_vector);
disp(acc);
disp('function call');
ind=accuracy_vector;
end


function [ind,acc]= KNNfunNew(images_train, labels_train,  images_test, labels_test, knum)
prediction=zeros(10,1);
actual=zeros(10,1);

for images_test_ind=1:1000
sum=zeros(size(images_train,1),784);
for K= 1: size(images_train,1)
        for L=1 : 784
            sum(K,L)= abs(images_train(K,L)^2-images_test(images_test_ind,L)^2);
        end;
    
end;

total=zeros(size(images_train,1),1);
sqrtmatrix=zeros(size(images_train,1),1);
for K=1:size(images_train,1)
    for L=1:784
        total(K)=total(K)+sum(K,L);
    end;
    sqrtmatrix(K)=sqrt(total(K));
end;

[sorted, indx]=sort(sqrtmatrix);
%disp(sorted);
%value = mode(sqrtmatrix);
gettingKLabels=zeros(knum,1);
for k = 1:knum

    gettingKLabels(k) = labels_train(indx(k));
end

maxNum = mode(gettingKLabels);
sortedK=zeros(knum,1);

for K=1:knum
    sortedK(K)=sorted(K);
end;


indMatrix=zeros(knum,1);
for L=1:knum
    for K=1:size(images_train)
        if sortedK(L)==sqrtmatrix(K)
            disp('in index');
            indMatrix(L)=K;
            break;
        end;
    end;
end;

disp('ind matrix');
disp(indMatrix);
disp('at loation');
disp(K);
disp('knum');
disp(knum);

%indMatrix=zeros(knum,1);
tempArr=zeros(knum,1);
for L=1:knum   
     tempArr(L)=labels_train(indMatrix(L));    
end;

disp('tempArr');
disp(tempArr);
disp(indMatrix);
 maxnum=mode(tempArr);

disp('MaxNum');
disp(maxnum);

im = reshape(images_test(images_test_ind, :), [28 28]);
imshow(im)



if maxNum==labels_test(images_test_ind)
    prediction((maxNum)+1)=prediction((maxNum)+1)+1;
end;

disp(prediction);

actual(labels_test(images_test_ind)+1)=actual(labels_test(images_test_ind)+1)+1;
disp(actual);

end;
accuracy_vector=zeros(10,1);

for L=1:10
    
        if actual(L)~=0
            accuracy_vector(L)=prediction(L)/actual(L);
        end;
    
end;



acc=mean(accuracy_vector);
disp(acc);
disp('function call');
ind=accuracy_vector;
end
