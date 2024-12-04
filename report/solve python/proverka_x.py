import sympy as sp

# Определяем переменные
x1, x2, x3, x4, y = sp.symbols('x1 x2 x3 x4 y')

# Выражаем одну из переменных из уравнения y
xi = sp.solve(0.068966*x1+0.344828*x2+0.172414*x3-0.482759*x4, x1)[0]

# Система уравнений
equations = [
-3.0504*x1 - 44.6371*x2 - 14.7046*x3 + 56.9748*x4,
-1.4677*x1 - 89.0323*x2 - 25.8790*x3 + 110.2661*x4,
-3.4798*x1 - 10.1452*x2 - 6.7681*x3 + 15.2601*x4,
-1.2984*x1 - 66.4516*x2 - 19.4315*x3 + 82.3508*x4]


# Подставляем выраженное xi в систему и приводим подобные члены
substitutedequations = [eq.subs(x1, xi) for eq in equations]

simplifiedequations = [sp.simplify(eq) for eq in substitutedequations]

# Выводим результаты
print("Выраженное xi:")
sp.pprint(xi)

for eq in substitutedequations:
    sp.pprint(eq)
    print()


print("\nПолученные уравнения:")
for eq in simplifiedequations:
    sp.pprint(eq)
    print()
