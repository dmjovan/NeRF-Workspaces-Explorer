import matplotlib.pyplot as plt
import numpy as np

data = np.loadtxt(
    r"C:\Users\jdmitrovic\Desktop\NeRF-Workspaces-Explorer\Replica_Dataset\office_1\Sequence_1\traj_w_c.txt",
    delimiter=" ").reshape(-1, 4, 4)

x_arr = []
y_arr = []
z_arr = []

for idx in range(data.shape[0]):
    x, y, z = data[idx, :3, 3]
    x_arr.append(x)
    y_arr.append(y)
    z_arr.append(z)
    print(f"{idx} ------> x = {x:3f}, y = {y:3f}, z = {z:3f}")

x_arr = np.array(x_arr)
z_arr = np.array(z_arr)
y_arr = np.array(y_arr)

# fig = plt.figure()
# plt.plot(z_arr, x_arr)
# plt.scatter(z_arr[0], x_arr[0])
# plt.scatter(z_arr[18], x_arr[18])
# plt.scatter(z_arr[49], x_arr[49])
# plt.scatter(z_arr[105], x_arr[105])
# plt.scatter(z_arr[204], x_arr[204])
# plt.scatter(z_arr[220], x_arr[220])
# plt.scatter(z_arr[272], x_arr[272])
# plt.scatter(z_arr[445], x_arr[445])
# plt.scatter(z_arr[545], x_arr[545])
# plt.scatter(z_arr[745], x_arr[745])
# plt.xlabel("z")
# plt.ylabel("x")
# plt.grid()
# plt.show()

fig = plt.figure()
plt.plot(z_arr, x_arr)
plt.scatter(z_arr[0], x_arr[0], label="0")
plt.scatter(z_arr[105], x_arr[105], label="105")
plt.scatter(z_arr[236], x_arr[236], label="236")
plt.scatter(z_arr[370], x_arr[370], label="370")
plt.scatter(z_arr[417], x_arr[417], label="417")
plt.scatter(z_arr[481], x_arr[481], label="481")
plt.scatter(z_arr[550], x_arr[550], label="550")
plt.scatter(z_arr[690], x_arr[690], label="690")
plt.scatter(z_arr[779], x_arr[779], label="779")
plt.scatter(z_arr[840], x_arr[840], label="840")
plt.scatter(z_arr[899], x_arr[899], label="899")
plt.xlabel("z")
plt.ylabel("x")
plt.legend()
plt.grid()
plt.show()

# fig = plt.figure()
# plt.plot(x_arr, z_arr)
# plt.scatter(x_arr[0], z_arr[0])
# plt.xlabel("x")
# plt.ylabel("z")
# plt.grid()
# plt.show()
