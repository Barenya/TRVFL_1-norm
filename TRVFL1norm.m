function [r,time]=TRVFL1NORM(train,test,L,c1,w_vec,b_vec)
    [no_input,no_col]=size(train);
    obs = train(:,no_col); 
    %w_vec is weight vector
    %b_vec is bias vector
    %c is penalty parameter 
    %L is hidden layer nodes

    x1 = train(:,1:no_col-1);
    y1 = train(:,no_col);
    [no_test,no_col] = size(test);
    xtest = test(:,1:no_col-1);
    ytest = test(:,no_col);
    xtest0 = xtest;

    train1 = [];
    train2 = [];

    for i = 1:no_input
        if(obs(i) == 1)
            train1 = [train1;x1(i,1:no_col-1)];
        else
            train2 = [train2;x1(i,1:no_col-1)];
        end;
    end;

    x1 = [train1 ; train2];    

    c2 = c1;
    c3=10^-4;
	c4 = c3;

    [m1,n_att] = size(train1);
    U = zeros(m1,L); 
    tic
     for i=1:m1
        for j=1:L
            prod = train1(i,:) * w_vec(:,j) + b_vec(j);
            U(i,j) = U(i,j) + 1.0 / ( 1.0 + exp(-prod) );    % using sigmoid function
        end
    end
    U=[U train1];
    [m2,n_att] = size(train2);
    V = zeros(m2,L);
    for i=1:m2
        for j=1:L
            prod = train2(i,:) * w_vec(:,j) + b_vec(j);
            V(i,j) = V(i,j) + 1.0 / ( 1.0 + exp(-prod) );    % using sigmoid function
        end
    end
	
	V=[V train2];
 	[m,n_att] = size(x1);

    e1 = ones(m1,1);
    e2 = ones(m2,1);
   
    lowb1=zeros(m2,1);
    lowb2=zeros(m1,1);
    upb1 = c1*e2;
    upb2 = c2*e1;
    m=m1+m2;

    H=U;
    G=V;
    HTH = H' * H;
    invHTH = inv(HTH + c3 * speye(L+n_att) );
    GINVGT = G * invHTH * G';
    GINVGT=(GINVGT+GINVGT')/2;
    GTG = G' * G;
    invGTG = inv (GTG + c4 * speye(L+n_att));
    HINVHT = H * invGTG * H';
    HINVHT = (HINVHT+HINVHT')/2;

       f1 = -e2';
       f2 = -e1';
    
    u1 = quadprog(GINVGT,f1,[],[],[],[],lowb1,upb1);
    u2 = quadprog(HINVHT,f2,[],[],[],[],lowb2,upb2);
    time= toc
    beta1 = - invHTH * G' *u1;
    beta2 =  invGTG * H' *u2;
    time=toc;

    [n_att,L] = size(w_vec);
    [no_test_input,n_att] = size(xtest0);
 
   I=zeros(no_test_input,L)
    for i=1:no_test_input
         for j=1:L
             prod = xtest0(i,:) * w_vec(:,j) + b_vec(j);
             I(i,j)=I(i,j) + 1.0 / ( 1.0 + exp(-prod) );  % formula of sigmoid function
         end
    end 
    I=[I xtest0]
    ytest1 = I * beta1 ;
    ytest2 = I * beta2;
    
    for i = 1 : size(ytest1,1)
        if abs(ytest1(i)) < abs(ytest2(i))
            classifier(i) = 1;
        else
            classifier(i) = 0;
        end;
    end;
%-----------------------------
match = 0.;
classifier = classifier';
for i = 1:size(ytest1,1)
    if(classifier(i) == ytest(i))
        match = match+1;
    end;
end;
confmat=confusionmat(y1,classifier,'order',[1,0,-1]);
TP=confmat(1,1);
TN=confmat(2,2);
FP=confmat(2,1);
FN=confmat(1,2);

TPR=TP/(TP+FN);
FPR=FP/(FP+TN);

accuracy=(TP+TN)/(TP+TN+FP+FN)*100;
AUC=((1+TPR-FPR)/2)*100;
recall=TP/(TP+FN);
precision=TP/(TP+FP);
f1=2*(precision*recall)/(precision+recall);
gmean=sqrt(precision*recall);
MCC=(TP*TN-FP*FN)/sqrt((TP+FP)*(TP+FN)*(TN+FP)*(TN+FN));
r=[accuracy;AUC;recall;precision;f1;gmean;MC]    
    
    
