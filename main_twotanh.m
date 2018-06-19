data;%training_Set的最后一列为结果
% stop_condition = 0.5;
i = 0;
hidden = 22;
times =3000;
W = rand(hidden,2)- 0.5;
V = rand(30,hidden) -0.5;
theta = rand(1,2)-0.5;
r = rand(1,hidden);
learning_rate = 0.1;
route = zeros(1,times);
precision = zeros(1,times);
result = zeros(1,times);
for j = 1:times
    for i = 1:size(r_training_set,1)
        alpha = r_training_set(i,1:size(r_training_set,2) -1) * V;
        b = tanh(alpha - r);
        beta = b * W;
        output = tanh(beta - theta);
        if r_training_set(i,size(r_training_set,2)) == 0
            label = [1,-1];
        end
        if r_training_set(i,size(r_training_set,2)) == 1
            label = [-1,1];
        end
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
        g = (ones(size(output)) - output.*output).* (label - output);
        delta_W = learning_rate * b' * g;
        delta_theta = - learning_rate * g;
        e = (ones(size(b)) - b.*b).*(g * W');
        delta_V = learning_rate * r_training_set(i,1:size(r_training_set,2)-1)' *e;
        delta_r = -learning_rate * e;
        %更新
        W = W + delta_W;
        V = V + delta_V;
        r = r + delta_r;
        theta = theta + delta_theta;
    end
    tmp = 0;
    for i = 1:size(r_training_set,1)
        alpha = r_training_set(i,1:size(r_training_set,2) -1) * V;
        b = tanh(alpha - r);
        beta = b * W;
        output = tanh(beta - theta);
        if r_training_set(i,size(r_training_set,2)) == 0
            label = [1,-1];
        end
        if r_training_set(i,size(r_training_set,2)) == 1
            label = [-1,1];
        end
        tmp = tmp + 0.5 * (label - output) * (label - output)';
    end
    route(j) = tmp;
    num_right = 0;
    for k= 1:size(r_validation_set,1)
        alpha_v = r_validation_set(k,1:size(r_validation_set,2) -1) * V;
        b_v = tanh(alpha_v - r);
        beta_v = b_v * W;
        output_v = tanh(beta_v - theta);
        if output_v(1) > output(2)
            label = 0;
        else
            label = 1;
        end
        if label == r_validation_set(k,size(r_validation_set,2))
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