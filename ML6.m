clc
clear all
close all

%---------------------------Principle Component Analysis---------------------------
%---------------------------reading data---------------------------
ds = datastore('house_prices_data_training_data.csv','TreatAsMissing','NA',.....
    'MissingValue',0,'ReadSize',25000);
T = read(ds);
X=T{:,4:21};
n=length(X(1,:));
% ---------------------------Normalization---------------------------
% for w=1:n
%     if max(abs(x(:,w)))~=0;
%         x(:,w)=(x(:,w)-mean((x(:,w))))./std(x(:,w));
%         
%     end
% end

%---------------------------correlation matrix---------------------------
corr_x=corr(X);

%---------------------------covariance matrix---------------------------
cov_x=cov(X);

%---------------------------SVD function---------------------------
[U,S,V]=svd(cov_x);

%---------------------------Eigen Value and alpha---------------------------
eigen_values=diag(S);
m=length(eigen_values);
for i=1:m
    alpha=1-(sum(eigen_values(1:i))/sum(eigen_values));
    if (alpha<=0.001)
        break
    end
end
K=i;

%---------------------------Reduced data---------------------------
reduced_data=(U(:,1:K)')*(X');

%---------------------------Multiplying with the eigen vector---------------------------
approx_data=U(:,1:K)*reduced_data;

%---------------------------Error---------------------------
error=(1/m)*(sum(approx_data-X'));

%---------------------------Linear Regression---------------------------
h=1;
theta=zeros(m,1);
k=1;
y=X(:,3)/mean(X(:,3));
E(k)=(1/(2*m))*sum((approx_data'*theta-y).^2); %cost function
lamda=0.001;
alpha=0.01;
while h==1
    alpha=alpha*1;
    theta=theta-(alpha/m)*approx_data*(approx_data'*theta-y);
    k=k+1;
    E(k)=(1/(2*m))*sum((approx_data'*theta-y).^2);
    Reg(k)=(1/(2*m))*sum((approx_data'*theta-y).^2)+(lamda/(2*m))*sum(theta.^2); %regularized cost function
    
    if E(k-1)-E(k)<0;
        break
    end
    q=(E(k-1)-E(k))./E(k-1);
    if q <.000001;
        h=0;
    end
end

%---------------------------K-MEANS---------------------------
cost_Func = zeros(1,15); 
[m n]=size(X);
centroid=zeros(m,n);
K=3;
for q = 1:5
    initial_index=randperm(m);
    centroid=X(initial_index(1:q),:);
    old_centroids=zeros(size(centroid));
    indices=zeros(size(X,1), 1);
    dist=zeros(m,q);
    No_Stop=true;
    iterations=0;
    while(No_Stop)
        for i = 1:m
            for j = 1:q
                dist(i, j) = sum((X(i,:) - centroid(j,:)).^2);
            end
        end
        for i = 1:m
            indices(i) = find(dist(i,:)==min(dist(i,:)));
        end
        for i= 1:q
            clustering = X(find(indices == i), :);
            centroid(i, :) = mean(clustering);
            cost = 0; %costfunction
            for z = 1 : size(clustering,1)
                cost = cost + (1/m)*sum((clustering(z,:) - centroid(i,:)).^2);
            end
            cost_Func(1,q) = cost;
        end
         if old_centroids == centroid
            No_Stop = false;
         end
        old_centroids = centroid;
        iterations = iterations + 1;
        end
    end
[o,K_Optimal] = min(cost_Func);
no_oF_Clusters = 1:15;
plot(no_oF_Clusters, cost_Func);
%---------------------------Anamoly Detection---------------------------
mean_data=mean(X);
s_d=std(X);
pdf_data=[];
for i=1:18
   pdf_data=[pdf_data normcdf(X(50,i),mean_data(i),s_d(i))];
end

if prod(pdf_data)>0.999
    anamoly=1;
else 
    if prod(pdf_data)<0.001
        anamoly=0;
    end
end




