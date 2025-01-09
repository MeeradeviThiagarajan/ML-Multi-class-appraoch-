# ML-Multi-class-appraoch-
Machine learning models for multi class
Performance of the classifiers for the lung disease multiclass classification problem. 

Prepare data
load lungdisease
Divide the data into training and validation sets
rng(1234)
part = cvpartition(lungdataNumMulti.LungDisease,'Holdout',0.2);
tridx = training(part);
tdata = lungdataNumMulti(tridx,:);
vdata = lungdataNumMulti(~tridx,:);
tdataAll = lungdataAllMulti(tridx,:);
vdataAll = lungdataAllMulti(~tridx,:);
Create a table to hold the results
mdlnames = {'kNN','kNN k=5','Weighted kNN k=5','Tree','Pruned tree',...
    'Tree (all predictors)','Pruned tree (all predictors)',...
    'NB','NB kernel','NB (all predictors)','NB kernel (all predictors)',...
    'Linear DA','Quadratic DA','Linear SVM','Gaussian SVM',...
    'Linear SVM (all predictors)','Gaussian SVM (all predictors)'};
results = table(zeros(17,1),zeros(17,1),...
    'RowNames',mdlnames,'VariableNames',{'ResubLoss','Loss'});

kNN models
m = fitcknn(tdata,'LungDisease');
results{1,:} = [resubLoss(m) loss(m,vdata)];

m.NumNeighbors = 5;
results{2,:} = [resubLoss(m) loss(m,vdata)];

m.DistanceWeight = 'squaredinverse';
results{3,:} = [resubLoss(m) loss(m,vdata)];

Tree models
m = fitctree(tdata,'LungDisease');
results{4,:} = [resubLoss(m) loss(m,vdata)];

m = prune(m,'Level',3);
results{5,:} = [resubLoss(m) loss(m,vdata)];

m = fitctree(tdataAll,'LungDisease');
results{6,:} = [resubLoss(m) loss(m,vdataAll)];

m = prune(m,'Level',3);
results{7,:} = [resubLoss(m) loss(m,vdataAll)];

Naive Bayes models
m = fitcnb(tdata,'LungDisease');
results{8,:} = [resubLoss(m) loss(m,vdata)];

m = fitcnb(tdata,'LungDisease','Distribution','kernel');
results{9,:} = [resubLoss(m) loss(m,vdata)];

m = fitcnb(tdataAll,'LungDisease');
results{10,:} = [resubLoss(m) loss(m,vdataAll)];

dists = [repmat({'kernel'},1,11),repmat({'mvmn'},1,10)];
m = fitcnb(tdataAll,'LungDisease','Distribution',dists);
results{11,:} = [resubLoss(m) loss(m,vdataAll)];

SVM models
template = templateSVM;
m = fitcecoc(tdata,'LungDisease','Learners',template);
results{14,:} = [resubLoss(m) loss(m,vdata)];

template = templateSVM('KernelFunction','gaussian');
m = fitcecoc(tdata,'LungDisease','Learners',template);
results{15,:} = [resubLoss(m) loss(m,vdata)];

template = templateSVM;
m = fitcecoc(tdataAll,'LungDisease','Learners',template);
results{16,:} = [resubLoss(m) loss(m,vdataAll)];

template = templateSVM('KernelFunction','gaussian');
m = fitcecoc(tdataAll,'LungDisease','Learners',template);
results{17,:} = [resubLoss(m) loss(m,vdataAll)];

View results
disp(results)

disp(sortrows(results,'Loss'))

