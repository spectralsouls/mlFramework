# mlFramework

## UnaryOps
EXP2: 2^x  
LOG2: log base 2 (x)  
CAST:  
BITCAST:  
SIN: sin(x) Note: x needs to be in radians  
SQRT: sqrt(x)    
NEG: x * -1  
RECIP: 1 / x  

## BinaryOps
ADD: x + y  
MUL: x * y  
IDIV: int(x / y)  
MAX: max(x ,y)  
MOD: x % y  
CMPLT: x < y  
CMPNE: !=  
XOR: x ^ y  
SHR: x >> y  
SHL: x << y  

## TernaryOps
WHERE: y if x else z  
MULACC: x * y + z  

## ReduceOps
SUM: [i_1, i_2, i_3, ...] -> i_1 + i_2 + i_3 + ... -> i_sum  
MAX: [i_1, i_2, i_3, ...] -> i_2 > i_3 > i_1 > ... -> i_max  

## LoadOps
EMPTY: [] -> [ , , ,]  
CONST: [] -> [i_1]   
COPY: [] -> [i_1, i_2, i_3]  
CONTIGUOUS:  
CUSTOM: [] -> [i_rndm, i_rndm, i_rndm], i_rndm is random   
ASSIGN:  
VIEW:  

## BufferOps
LOAD: 
CONST:
STORE: