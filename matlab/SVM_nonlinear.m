%% Clear
clear all
clc

%% Import data

opts = delimitedTextImportOptions("NumVariables", 5);

% Specify range and delimiter
opts.DataLines = [1, Inf];
opts.Delimiter = ",";

% Specify column names and types
opts.VariableNames = ["x1", "x2", "x3", "x4", "y"];
opts.VariableTypes = ["double", "double", "double", "double", "double"];
opts.ExtraColumnsRule = "ignore";
opts.EmptyLineRule = "read";

% Import the data
databanknoteauthentication = readtable("/Users/diego/Documents/MATLAB/data_banknote_authentication.txt", opts);
clear opts

%% Input and ouput variables

x=[databanknoteauthentication.x1 databanknoteauthentication.x2...
    databanknoteauthentication.x3 databanknoteauthentication.x4];
y=2*databanknoteauthentication.y-1;

%% Scatter plot of the data

figure(1)
gscatter(x(:,1),x(:,2),y)

%% Training and test data

% Training
n=1200;
x_train=x(1:n,1:2);
y_train=y(1:n);

%% Kernel matrix

K=zeros(n);
sigma=5;
for i=1:n
    for j=1:n
        %K(i,j)=klin(x_train(i,:),x_train(j,:));%Linear kernel
        K(i,j)=krbf(x_train(i,:),x_train(j,:),sigma); %RBF kernel
    end
end

%% Set variables for optimization

Ym=y_train*y_train';
H=K.*Ym;
onev=ones(1,n)';
Aeq=y_train';
beq=0;
c=10;
lb=zeros(1,n);
ub=c*ones(1,n);

%% Optimal alpha values

alpha = quadprog(H,-onev,[],[],Aeq,beq,lb,ub);

%% Support values and support vectors

ind=alpha>1e-10;
alpha_sv=alpha(ind);
n_sv=length(alpha_sv);
x_sv=x_train(ind,:);
y_sv=y_train(ind);
K_sv=K(ind,ind);

%% Calculation of w and b
%  The calculation of w is only possible for some particular cases

%  The weights w for the linear case
%w=sum([alpha_sv.*y_sv alpha_sv.*y_sv].*x_sv)';

b=mean(y_sv-K_sv*(alpha_sv.*y_sv));

%% Prediction model

x_p=[-6 -15];
K_pred=zeros(n_sv,1);
for i=1:n_sv
     %K_pred(i)= klin(x_p,x_sv(i,:));
     K_pred(i)= krbf(x_p,x_sv(i,:),sigma);
end

y_pred= sign(sum(alpha_sv.*y_sv.*K_pred)+b);

%% Set values for plots

% x values for plotting
n1=100;
n2=100;
x1plot=linspace(-8,8,n1);
x2plot=linspace(-15,15,n2);
[X1,X2]=meshgrid(x1plot,x2plot);
x1m=reshape(X1,n1*n2,1);
x2m=reshape(X2,n1*n2,1);
Xm=[x1m x2m];

% y values (from the model)
ym = zeros(n1*n2,1);

for j=1:n1*n2
    x_p=Xm(j,:);
    K_pred=zeros(n_sv,1);
    for i=1:n_sv
         %K_pred(i)= klin(x_p,x_sv(i,:));
         K_pred(i)= krbf(x_p,x_sv(i,:),sigma);
    end
    ym(j)= sign(sum(alpha_sv.*y_sv.*K_pred)+b);  
end

yplot=reshape(ym,n1,n2);

%% Plots

figure(2)
gscatter(x(:,1),x(:,2),y)
hold on
contour(x1plot,x2plot,yplot)
title('Complete dataset')
hold off

%% Kernel functions

function K=klin(x,y)
    x=x(:);
    y=y(:);
    K=x'*y;
end

function K=krbf(x,y,sigma)
    x=x(:);
    y=y(:);
    K=exp(-norm(x-y)^2/(2*sigma^2));
end


