import numpy as np

A = np.array([[10, 20],
              [30, 40]])

B = np.array([[1, 2],
              [3, 4]])

print("A=")
print(A)

print("\n B=")
print(B)

#加算
add_result = A + B
print("\n A + B =")
print(add_result)

#減算
sub_result = A - B
print("\n A - B =")
print(sub_result)

#乗算
mul_result = A * B
print("\n A * B =")
print(mul_result)

#行列積
dot_result = np.dot(A, B)
print("\n A ・ B =")
print(dot_result)

#除算
div_result = A / B
print("\n A / B =")
print(div_result)