# Генерация матрицы T
set.seed(123)  # Для воспроизводимости
T <- matrix(sample(-15:15, 16, replace = TRUE), nrow = 4)

# Проверка определителя матрицы T
det_T <- det(T)
print(paste("Определитель матрицы T:", det_T))
print("матрица T:")
print (T)
# Формирование матрицы B
B <- matrix(0, nrow = 4, ncol = 4)
diag(B) <- c(-10, -5, -1, -0.5)
print("матрица B:")
print (B)


# Формирование матрицы A
A <- solve(T) %*% B %*% T
# Проверка определителя матрицы A
det_A <- det(A)
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
x0 <- c(2, -1.5, 1, -2.5)

# Нахождение коэффициентов C
C <- solve(eigenvectors, x0)

# Выводим результаты
print(paste("Собственные значения:", toString(eigenvalues)))
print("Собственные векторы:")
print(eigenvectors)
print(paste("Проверка корректности:", is_correct))
print("Коэффициенты C:")
print(C)
# Вычисляем обратную матрицу A
A_inv <- solve(A)

# Печатаем обратную матрицу
print ("Обратная матрица А:")
print(A_inv)