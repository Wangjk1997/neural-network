data;%training_Set的最后一列为结果
% stop_condition = 0.5;
hidden = 10;
i = 0;
times =3000;
W = rand(hidden,1)- 0.5;
V = rand(30,hidden) -0.5;
theta = rand(1,1)-0.5;
r = rand(1,hidden);
learning_rate = 0.5;
route = zeros(1,times);
precision = zeros(1,times);
result = zeros(1,times);
for j = 1:times
    for i = 1:size(r_training_set,1)
        alpha = r_training_set(i,1:size(r_training_set,2) -1) * V;
        b = sigmoid(alpha - r);
        beta = b * W;
        output = sigmoid(beta - theta);
%         if i == 300
%             i;
%             output;
%             r_training_set(i,size(r_training_set,2));
%         end
%         if i == 469
%             i;
%             output;
%             r_training_set(i,size(r_training_set,2));
%         end
        g = output * (1 - output) * (r_training_set(i,size(r_training_set,2)) - output);
        delta_W = learning_rate * g * b;
        delta_theta = - learning_rate * g;
        e = b.*(1 - b).*W'*g;
        delta_V = learning_rate * r_training_set(i,1:size(r_training_set,2)-1)' *e;
        delta_r = -learning_rate * e;
        %更新
        W = W + delta_W';
        V = V + delta_V;
        r = r + delta_r;
        theta = theta + delta_theta;
    end
    tmp = 0;
    for i = 1:size(r_training_set,1)
        alpha = r_training_set(i,1:size(r_training_set,2) -1) * V;
        b = sigmoid(alpha - r);
        beta = b * W;
        output = sigmoid(beta - theta);
        tmp = tmp + 0.5 * (r_training_set(i,size(r_training_set,2)) - output)^2;
    end
    route(j) = tmp;
    num_right = 0;
    for k= 1:size(r_validation_set,1)
        alpha_v = r_validation_set(k,1:size(r_validation_set,2) -1) * V;
        b_v = sigmoid(alpha_v - r);
        beta_v = b_v * W;
        output_v = sigmoid(beta_v - theta);
        if output_v >0.5
            output_v = 1;
        else
            output_v = 0;
        end
        if output_v == r_validation_set(k,size(r_validation_set,2))
            num_right = num_right + 1;
        end
    end
    precision(j) = num_right / size(r_validation_set,1);
end
figure(1)
plot(1:times,precision)
xlabel('times')
ylabel('precision')
figure(2)
plot(1:times,route)
xlabel('times')
ylabel('loss')