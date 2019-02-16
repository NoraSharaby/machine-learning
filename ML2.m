clear all
ds = datastore('heart_DD.csv','TreatAsMissing','NA',.....
    'MissingValue',0,'ReadSize',25000);
T = read(ds);
Alpha=0.01;
m=length(T{:,1});
U=T{:,1:5};
% U=T{:,1:6};
% U=T{:,1:7};
% U=T{:,1:8};

X=[ones(m,1) U U.^2 U.^3 U.^4];% Adding different Us or changing the power of the U give different hypothesis
% X=[ones(m,1) U U.^2 U.^3];
% X=[ones(m,1) U U.^2];
% X=[ones(m,1) U]

n=length(X(1,:));              %number of columns

for w=2:n                      % Normalization
    if max(abs(X(:,w)))~=0
    X(:,w)=(X(:,w)-mean((X(:,w))))./std(X(:,w));
    end
end
Y=T{:,14}/mean(T{:,14});         % Target column
Theta=zeros(n,1);
k=1;
hb=log(X*Theta);
E(k)=(1/m)*sum((-Y'*log(X*Theta))-((1-Y)'*log(1-X*Theta)));

R=1;
while R==1                     % Gradient decent
Alpha=Alpha*1;
Theta=Theta-(Alpha/m)*X'*(X*Theta-Y);
k=k+1;
E(k)=(1/(m))*sum((X*Theta-Y).^2);
if E(k-1)-E(k)<0
    break
end 
q=(E(k-1)-E(k))./E(k-1);
if q <.000001;
    R=0;
end
end
