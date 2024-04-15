inputs = [1, 2, 3, 2.5]
weights = [[0.2, 0.8, -0.5, 1],
 [0.5, -0.91, 0.26, -0.5],
 [-0.26, -0.27, 0.17, 0.87]]
biases = [2, 3, 0.5]

layer_out = []
for w, b in zip(weights, biases):
    output = 0
    for i, w1 in zip(inputs, w):
        output += i * w1
    output += b
    layer_out.append(output)


print(layer_out)