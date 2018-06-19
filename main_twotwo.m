data;%training_Set的最后一列为结果
% stop_condition = 0.5;
i = 0;
times =2000;
W = rand(20,2)- 0.5;
V = rand(40,20) - 0.5;
O = rand(30,40) - 0.5;
theta = rand(1,2)-0.5;
gamma = rand(1,20) - 0.5;
sigma = rand(1,40) - 0.5;
learning_rate = 0.3;
route = zeros(1,times);
precision = zeros(1,times);
result = zeros(1,times);
for j = 1:times
    for i = 1:size(r_training_set,1)
        r = r_training_set(i,1:size(r_training_set,2) -1) * O;
        c = sigmoid(r - sigma);
        alpha = c * V;
        b = sigmoid(alpha - gamma);
        beta = b * W;
        output = sigmoid(beta - theta);
        if r_training_set(i,size(r_training_set,2)) == 0
            label = [1,0];
        end
        if r_training_set(i,size(r_training_set,2)) == 1
            label = [0,1];
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
        g = output .* (ones(size(output)) - output) .* (label - output);
        delta_W = learning_rate * b' * g;
        delta_theta = - learning_rate * g;
        e = b.*(1 - b).*(g * W');
        delta_V = learning_rate * c' *e;
        delta_gamma = -learning_rate * e;
        f = c.*(1 - c).*(e *V');
        delta_O = learning_rate * r_training_set(i,1:size(r_training_set,2)-1)' * f;
        delta_sigma = -learning_rate * f;
        %更新
        W = W + delta_W;
        V = V + delta_V;
        O = O + delta_O;
        gamma = gamma + delta_gamma;
        theta = theta + delta_theta;
        sigma = sigma + delta_sigma;
    end
%     tmp = 0;
%     for i = 1:size(r_training_set,1)
%         r = r_training_set(i,1:size(r_training_set,2) -1) * O;
%         c = sigmoid(r - sigma);
%         alpha = c * V;
%         b = sigmoid(alpha - gamma);
%         beta = b * W;
%         output = sigmoid(beta - theta);
%         tmp = tmp + 0.5 * (r_training_set(i,size(r_training_set,2)) - output)^2;
%     end
    route(j) = tmp;
    num_right = 0;
    for k= 1:size(r_validation_set,1)
        r_v = r_validation_set(k,1:size(r_validation_set,2) -1) * O;
        c_v = sigmoid(r_v - sigma);
        alpha_v = c_v * V;
        b_v = sigmoid(alpha_v - gamma);
        beta_v = b_v * W;
        output_v = sigmoid(beta_v - theta);
%         alpha_v = r_validation_set(k,1:size(r_validation_set,2) -1) * V;
%         b_v = sigmoid(alpha_v - gamma);
%         beta_v = b_v * W;
%         output_v = sigmoid(beta_v - theta);
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
% figure(2)
% plot(1:times,route)
% xlabel('times')
% ylabel('loss')