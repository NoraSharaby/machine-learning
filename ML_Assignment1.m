clear all
ds = datastore('house_prices_data_training_data.csv','TreatAsMissing','NA',.....
    'MissingValue',0,'ReadSize',25000);
T = read(ds);                  % Data in a form of table
size(T);                      
Alpha=0.01;
lamda=0.001;

m=length(T{:,1});
U0=T{:,2};
U=T{:,4:10};
% U=T{:,4:6};
% U=T{:,4:7};
% U=T{:,4:8};
U1=T{:,20:21};
X=[ones(m,1) U U1 U.^2 U.^3];  % Adding different Us or changing the power of the U give different hypothesis
% X=[ones(m,1) U];
% X=[ones(m,1) U U.^2];
% X=[ones(m,1) U U1 U.^2 U.^3 U.^4];
n=length(X(1,:));              %number of columns
for w=2:n                      % Normalization
    if max(abs(X(:,w)))~=0
    X(:,w)=(X(:,w)-mean((X(:,w))))./std(X(:,w));
    end
end

Y=T{:,3}/mean(T{:,3});         % Price column
Theta=zeros(n,1);
k=1;

E(k)=(1/(2*m))*sum((X*Theta-Y).^2);

R=1;
while R==1                     % Gradient decent
Alpha=Alpha*1;
Theta=Theta-(Alpha/m)*X'*(X*Theta-Y);
% Theta=Theta*(1-(lamda*Alpha/m))-(Alpha/m)*X'*(X*Theta-Y); %Regularization
k=k+1;
E(k)=(1/(2*m))*sum((X*Theta-Y).^2);
% E(k)=(1/(2*m))*sum((X*Theta-Y).^2)+(lamda/2*m)sum(Theta); %Regularization
if E(k-1)-E(k)<0
    break
end 
q=(E(k-1)-E(k))./E(k-1);
if q <.000001;
    R=0;
end
end
plot(E)

Theta_noramal=inv(X'*X)*X'*Y;

%training...................................
m_train=ceil(0.6*m);
U0_train=T{1:m_train,2};
U_train=T{1:m_train,4:10};
% U=T{:,4:6};
% U=T{:,4:7};
% U=T{:,4:8};
U1_train=T{1:m_train,20:21};
X_train=[ones(m_train,1) U_train U1_train U_train.^2 U_train.^3];  % Adding different Us or changing the power of the U give different hypothesis
% X=[ones(m,1) U];
% X=[ones(m,1) U U.^2];
% X=[ones(m,1) U U1 U.^2 U.^3 U.^4];

Y_train=T{1:m_train,3}/mean(T{1:m_train,3});         % Price column
Theta_train=zeros(n,1);
k_train=1;

E_train(k_train)=(1/(2*m_train))*sum((X_train*Theta_train-Y_train).^2);

R_train=1;
while R_train==1                     % Gradient decent
Alpha=Alpha*1;
Theta_train=Theta_train-(Alpha/m_train)*X_train'*(X_train*Theta_train-Y_train);
k_train=k_train+1;
E_train(k_train)=(1/(2*m_train))*sum((X_train*Theta_train-Y_train).^2);
if E_train(k_train-1)-E_train(k_train)<0
    break
end 
q=(E_train(k_train-1)-E_train(k_train))./E_train(k_train-1);
if q <.000001;
    R_train=0;
end
end
%test...................................
m_test=(m-m_train)/2;


U0_test=T{m_train+1:m_train+m_test,2};
U_test=T{m_train+1:m_train+m_test,4:10};
% U=T{:,4:6};
% U=T{:,4:7};
% U=T{:,4:8};
U1_test=T{m_train+1:m_train+m_test,20:21};
X_test=[ones(m_test,1) U_test U1_test U_test.^2 U_test.^3];  % Adding different Us or changing the power of the U give different hypothesis
% X=[ones(m,1) U];
% X=[ones(m,1) U U.^2];
% X=[ones(m,1) U U1 U.^2 U.^3 U.^4];

Y_test=T{m_train+1:m_train+m_test,3}/mean(T{m_train+1:m_train+m_test,3});         % Price column
E_test=(1/(2*m_test))*sum((X_test*Theta_train-Y_test).^2);

% Theta_test=zeros(n,1);
% k_test=1;
% 
% E_test(k_test)=(1/(2*m_test))*sum((X_test*Theta_test-Y_test).^2);
% 
% R_test=1;
% while R_test==1                     % Gradient decent
% Alpha=Alpha*1;
% Theta_test=Theta_test-(Alpha/m_test)*X_test'*(X_test*Theta_test-Y_test);
% k_test=k_test+1;
% E_test(k_test)=(1/(2*m_test))*sum((X_test*Theta_test-Y_test).^2);
% if E_test(k_test-1)-E_test(k_test)<0
%     break
% end 
% q=(E_test(k_test-1)-E_test(k_test))./E_test(k_test-1);
% if q <.000001;
%     R_test=0;
% end
% end

%CV...................................
m_cv=(m-m_train)/2;


U0_cv=T{m_train+m_test+1:m_train+m_test+m_cv,2};
U_cv=T{m_train+m_test+1:m_train+m_test+m_cv,4:10};
% U=T{:,4:6};
% U=T{:,4:7};
% U=T{:,4:8};
U1_cv=T{m_train+m_test+1:m_train+m_test+m_cv,20:21};
X_cv=[ones(m_cv,1) U_cv U1_cv U_cv.^2 U_cv.^3];  % Adding different Us or changing the power of the U give different hypothesis
% X=[ones(m,1) U];
% X=[ones(m,1) U U.^2];
% X=[ones(m,1) U U1 U.^2 U.^3 U.^4];

Y_cv=T{m_train+m_test+1:m_train+m_test+m_cv,3}/mean(T{m_train+m_test+1:m_train+m_test+m_cv,3});         % Price column
E_cv=(1/(2*m_cv))*sum((X_cv*Theta_train-Y_cv).^2);
% Theta_cv=zeros(n,1);
% k_cv=1;
% 
% E_cv(k_cv)=(1/(2*m_cv))*sum((X_cv*Theta_cv-Y_cv).^2);

% R_cv=1;
% while R_cv==1                     % Gradient decent
% Alpha=Alpha*1;
% Theta_cv=Theta_cv-(Alpha/m_cv)*X_cv'*(X_cv*Theta_cv-Y_cv);
% k_cv=k_cv+1;
% E_cv(k_cv)=(1/(2*m_cv))*sum((X_cv*Theta_cv-Y_cv).^2);
% if E_cv(k_cv-1)-E_cv(k_cv)<0
%     break
% end 
% q=(E_cv(k_cv-1)-E_cv(k_cv))./E_cv(k_cv-1);
% if q <.000001;
%     R_cv=0;
% end
% end


