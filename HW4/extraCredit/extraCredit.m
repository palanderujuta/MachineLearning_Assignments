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
 
model = svmtrain(labels_train, images_train,'-t 2 -c 100'); %train the data
w = (model.sv_coef' * full(model.SVs));
bias = -model.rho;
predictions = sign(images_test * w' );
%%[predict_label, accuracy, dec_values] = svmpredict(labels_test, images_test, model); % test the training data
%
%w1=w(1,:)';
%estimate = sign(images_test*w1);
%acca = sum(estimate' == labels_test);
%acca = acca/numel(estimate);
%disp(acca)