T = [-2, -10, -5, 14; 12, -6, 10, -2; -6, 11, 3, -11; 14, 2, -7, -11]; # создание случайной матрицы
T
det(T)
B = diag ([-10,-5,-1,-0.5]) #создание диагональной матрицы
A = inv(T) * B * T # inv - инверсная матрица
[eigvec, eigval] = eig(A)
A * eigvec(:,4)
B(4,4) * eigvec(:,4)



x0 = [2,-1.5, 1, -2.5] # вектор начальных условий
c = inv(eigvec) * x0' # вектор-столбец коэффициентов C
H =  eigvec * diag(c)
diag(c)
H_inv = inv(H)

