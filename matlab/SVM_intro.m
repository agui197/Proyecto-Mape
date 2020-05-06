%% Clear
clear all
clc

%% Simulate data

% Simulated dataset
n=20;
x1=linspace(-10,10,n);
x2=linspace(-10,10,n);
[X1,X2]=meshgrid(x1,x2);
Y=2*(X1^3-3*X1>X2)-1;

% Reshape step
x1m=reshape(X1,n^2,1);
x2m=reshape(X2,n^2,1);
Xm=[x1m x2m];
y=reshape(Y,n^2,1);
Ym=y*y';
Ver=[Xm y];

%% Scatter plot of the data

figure(1)
gscatter(x1m,x2m,y)

%% Kernel matrix

K=zeros(n^2);
for i=1:n^2
    for j=1:n^2
        K(i,j)=Xm(i,:)*Xm(j,:)';
    end
end

%% Set variables for optimization

H=K.*Ym;
onev=ones(1,n^2)';
Aeq=y';
beq=0;
c=0.1;
lb=zeros(1,n^2);
ub=c*ones(1,n^2);

%% Optimal alpha values

alpha = quadprog(H,-onev,[],[],Aeq,beq,lb,ub);

%% Support values and support vectors

ind=alpha>1e-10;
alpha_sv=alpha(ind);
x_sv=Xm(ind,:);
y_sv=y(ind);

%% Calculation of w and b

w=sum([alpha_sv.*y_sv alpha_sv.*y_sv].*x_sv)';
b=mean(1./y_sv-x_sv*w);

%% Prediction model

x=[10 1]';
y_est=sign(w'*x+b);

%% Plots

figure(2)
gscatter(x1m,x2m,y)
hold on
plot(x1,(-w(1)*x1-b)/w(2))
axis([-10 10 -10 10])
hold off
