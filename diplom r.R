# Генерация матрицы T
set.seed(123)  # Для воспроизводимости
# Данные
data <- c(-2, -10, -5, 14, 12, -6, 10, -2, -6, 11, 3, -11, 14, 2, -7, -11)

# Формирование матрицы 4х4
T <- matrix(data, nrow = 4, byrow = TRUE)


print("матрица T:")
print (T)
# Формирование матрицы B
B <- matrix(0, nrow = 4, ncol = 4)
diag(B) <- c(-10, -5, -1, -0.5)
print("матрица B:")
print (B)


# Формирование матрицы A
A <- solve(T) %*% B %*% T

print("матрица A:")
print (A)

# Находим собственные значения и собственные векторы
eigen_result <- eigen(A)

# Собственные значения
eigenvalues <- eigen_result$values
# Собственные векторы
eigenvectors <- eigen_result$vectors

# Проверка: A * v = λ * v для каждого собственного вектора v и соответствующего собственного значения λ
check_eigen <- function(A, eigenvalues, eigenvectors) {
  for (i in 1:length(eigenvalues)) {
    lambda <- eigenvalues[i]
    v <- eigenvectors[, i]
    # Преобразуем результат в вектор
    if (!all.equal(as.vector(A %*% v), as.vector(lambda * v))) {
      return(FALSE)
    }
  }
  return(TRUE)
}

# Выполняем проверку
is_correct <- check_eigen(A, eigenvalues, eigenvectors)

# Задаем вектор начальных условий
x0 <- t(c(2, -1.5, 1, -2.5))
x0_tr <- t(x0)
print(x0)
# Нахождение коэффициентов C
C <- solve(eigenvectors)%*%t(x0)

# Выводим результаты
print(paste("Собственные значения:", toString(eigenvalues)))
print("Собственные векторы:")
print(eigenvectors)
print(paste("Проверка корректности:", is_correct))
print("Коэффициенты C:")
print(C)
print(diag(C))
# Вычисляем обратную матрицу A
#H <- eigenvectors %*% 

# Печатаем обратную матрицу
print ("Обратная матрица А:")
print(A_inv)