from sympy import symbols, Eq, solve
from sympy.stats import Normal, density

# Define symbolic variable x
x = symbols('x')

# Define normal distributions
y1 = density(Normal('N1', 5, 0.5))(x)  # PDF for N(5, 0.5^2)
y2 = density(Normal('N2', 7, 1))(x)    # PDF for N(7, 1^2)

# Solve the equation 2 * y1 == y2 for x
solution = solve(Eq(2 * y1, y2), x)

# Display the numerical solutions
print(solution)