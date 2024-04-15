from mlops import dotproduct

inputs = [0.1, 0.5]
weights = [[0.1,0.3], [0.2,0.4]]
biases = [0.25, 0.25]

layer_out = []
for w, b in zip(weights, biases):
    output = 0
    for i, w1 in zip(inputs, w):
        output += i * w1
    output += b
    layer_out.append(output)


print(layer_out)


inputs = [1.0, 2.0, 3.0, 2.5]
weights = [0.2, 0.8, -0.5, 1.0]
bias = 2.0

output = dotproduct(inputs, weights) + bias
print(output)