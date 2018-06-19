% Irisdata;
times =3000;
hidden = 5;
loss = zeros(1,5);
error = zeros(1,5);
m = 1;
for hidden = 5:5:25
W = rand(hidden,30);
V = rand(30,hidden);
theta = rand(1,30);
r = rand(1,hidden);
learning_rate = 0.05;

for j = 1:times
    for i = 1:size(r_training_set,1)
        alpha = r_training_set(i,1:size(r_training_set,2) -1) * V;
        b = sigmoid(alpha - r);
        beta = b * W;
        output = sigmoid(beta - theta);

        g = output .* (ones(size(output)) - output) .* (r_training_set(i,1:size(r_training_set,2) -1) - output);
        delta_W = learning_rate * b' * g;
        delta_theta = - learning_rate * g;
        e = b.*(1 - b).*(g * W');
        delta_V = learning_rate * r_training_set(i,1:size(r_training_set,2)-1)' *e;
        delta_r = -learning_rate * e;
        %¸üÐÂ
        W = W + delta_W;
        V = V + delta_V;
        r = r + delta_r;
        theta = theta + delta_theta;
    end
end
    tmp = 0;
    for i = 1:size(r_training_set,1)
        alpha = r_training_set(i,1:size(r_training_set,2) -1) * V;
        b = sigmoid(alpha - r);
        beta = b * W;
        output = sigmoid(beta - theta);
        tmp = tmp + 0.5 * (output - r_training_set(i,1:size(r_training_set,2) -1))*(output - r_training_set(i,1:size(r_training_set,2) -1))';
    end
    loss(m) = tmp;
    tmp = 0;
    for k= 1:size(r_validation_set,1)
        alpha_v = r_validation_set(k,1:size(r_validation_set,2) -1) * V;
        b_v = sigmoid(alpha_v - r);
        beta_v = b_v * W;
        output_v = sigmoid(beta_v - theta);
        tmp = tmp + 0.5 * (output_v - r_validation_set(k,1:size(r_validation_set,2) - 1)) * (output_v - r_validation_set(k,1:size(r_validation_set,2) - 1))';
    end
    error(m) = tmp;
    m = m+1
end
figure(1)
plot(5:5:25,loss)
xlabel('hidden')
ylabel('loss')
figure(2)
plot(5:5:25,error)
xlabel('hidden')
ylabel('error')