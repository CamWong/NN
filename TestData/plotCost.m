% Read and plot NN_cost
fid = fopen('NN_cost.txt','r');
data = fscanf(fid,'%f');
fclose(fid);
lr = data(1);
data = data(2:end);
data = reshape(data,2,[]);
close all
scatter(data(1,:),data(2,:),'b.')
title(['Cost Over Iterations. \alpha = ' num2str(lr)])
ylabel('Cost - [rms]')
xlabel('Iteration Number')
grid minor