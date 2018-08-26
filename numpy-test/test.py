import numpy as np

if __name__ == '__main__':
    a = np.zeros(shape=(2, 4, 3))
    b = np.empty(shape=(2, 4, 3))
    c = np.ones(shape=(2, 4, 3))

    d = c.reshape(shape=(-1, 1))
    print("end")



