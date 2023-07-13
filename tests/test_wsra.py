
theta = np.deg2rad(45)
R = rotation_matrix(theta)

x = np.array([1, 2, 3, 4])
y = np.array([0, 0, 0, 0])

X = np.vstack([x, y])
X_rot = R @ X



#%%
theta = np.deg2rad(45)
R = rotation_matrix(theta)

x = np.array([1, 2, 3, 4])
y = np.array([0, 0, 0, 0])

X = np.vstack([x, y])
X_rot = R @ X

x_rot, y_rot = X_rot

fig, ax = plt.subplots()
ax.plot(x, y)
for theta in np.linspace(0, 180, 10):
    theta = np.deg2rad(theta)
    R = rotation_matrix(theta)
    X_rot = R @ X
    x_rot, y_rot = X_rot
    ax.plot(x_rot, y_rot)
ax.set_xlim([-5, 5])
ax.set_ylim([-5, 5])
ax.set_aspect('equal')

#%%