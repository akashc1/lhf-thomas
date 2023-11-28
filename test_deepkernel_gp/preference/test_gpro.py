from GPro.preference import ProbitPreferenceGP
import numpy as np
import matplotlib.pyplot as plt

X = np.array([2, 1]).reshape(-1, 1)

M = np.array([0, 1]).reshape(-1, 2)

gpr = ProbitPreferenceGP()

gpr.fit(X, M, f_prior=None)

X_new = np.linspace(-6, 9, 100).reshape(-1, 1)
predicted_values, predicted_deviations = gpr.predict(X_new, return_y_std=True)

plt.plot(X_new, np.zeros(100), 'k--', label='GP predictive prior')
plt.plot(X_new, predicted_values, 'r-', label='GP predictive posterior')
plt.plot(X.flat, gpr.predict(X).flat, 'bx', label='Preference')
plt.ylabel('f(X)')
plt.xlabel('X')
# the predicted s.d. is divided for an aesthetic purpose.
plt.gca().fill_between(X_new.flatten(),
                       (predicted_values - predicted_deviations / 50).flatten(),
                       (predicted_values + predicted_deviations / 50).flatten(),
                       color="#b0e0e6", label='GP predictive posterior s.d.')
plt.legend()
plt.savefig('test_gpro.png')