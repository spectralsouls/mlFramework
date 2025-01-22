# mlFramework

tensor.py contains the front-end

functions.py is the layer underneath tensor.py that actually calls the functions (all base functions are implemented by numpy)

layers.py represent neural network layers, which are essentially more complex functions made up of a certain combination of base functions

(the above files are all in the mlf folder)

mnist.py in the examples folder is an example implementation in pytorch

Base functions are listed below (unaryops, binaryops, movementops

## UnaryOps
EXP2: 2^x  
LOG2: log base 2 (x)   
SIN: sin(x) Note: x needs to be in radians  
SQRT: sqrt(x)    
NEG: x * -1  
RECIP: 1 / x  

## BinaryOps
ADD: x + y  
MUL: x * y  
DIV: int(x / y)  

## ReduceOps
SUM: [i_1, i_2, i_3, ...] -> i_1 + i_2 + i_3 + ... -> i_sum  

## MovementOps
Expand: [[a], [b], [c]] -> [[a,a,a], [b,b,b], [c,c,c]]  
Reshape:  [a, b, c, d] -> [[a. b], [c, d]]  
Pad: [[a], [b]] -> [[0, 0, a, 0, 0], [0, 0, b, 0, 0]]  
Shrink: [[a, b], [c, d], [e, f]] -> [[b], [d], [f]]   
Flip: [[a, b, c], [d, e, f]] -> [[d, e, f], [a, b, c]]  
Permute: [a, b, c] -> [b, c, a]  
