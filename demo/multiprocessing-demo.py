#!/usr/bin/env python
# -*- coding: UTF-8 -*-

# xpecgen demo: calculate multiple distributions using multiple processes by means of the command line interface.
# The distributions are stored in independent files which can be loaded later.


from __future__ import print_function
import subprocess
import os
from multiprocessing import Process, Queue

import numpy as np

# Electron energies
e_0_list = np.linspace(10, 300, num=10)
# Use the same x-ray mesh in all of them
gamma_mesh = np.linspace(3, 300, num=50)
angle = 12
output_dir = "spectra"

num_threads = 2

# Create the directory if needed
try:
    # Will fail either if exists or unable to create it
    os.makedirs(output_dir)
except OSError:
    if os.path.exists(output_dir):
        # Directory did [probably] exist
        pass
    else:
        raise OSError("Unable to create directory " + output_dir)


# Create some classes for the multiprocessing
class Calculate:
    """Perform a calculation when called"""

    def __init__(self, e_0, i, n):
        self.e_0 = e_0
        self.i = i
        self.n = n

    def __call__(self):
        print("Starting calculation %d/%d" % (self.i, self.n))
        return subprocess.run(" ".join(["xpecgencli", str(self.e_0), str(angle), "--n_points", "100",
                                        "-o", os.path.join(output_dir, "S%d.pkl" % self.i),
                                        "--mesh" + " " + " ".join(map(lambda x: "%.4f" % x, gamma_mesh)),
                                        "--overwrite",
                                        "--epsrel", "0.1"]),
                              shell=True, stdout=subprocess.DEVNULL)


def _caller(q):
    """Execute calls from a Queue until its value is 'END'"""
    while True:
        data = q.get()
        if data == "END":
            break
        else:
            data()  # Call the supplied argument


class MPCaller:
    def __init__(self, num_threads=2):
        self._queue = Queue()
        self.processes = []
        for _ in range(num_threads):
            t = Process(target=_caller, args=(self._queue,))
            t.daemon = True
            t.start()
            self.processes.append(t)

    def add_call(self, call):
        """Add a call to the instance's stack"""
        self._queue.put(call)

    def wait_calls(self):
        """Ask all processes to consume the queue and stop after that"""
        num_threads = len(self.processes)
        for _ in range(num_threads):
            self._queue.put("END")
        for t in self.processes:
            t.join()
        self.processes = []


caller = MPCaller(num_threads)

for i, e_0 in enumerate(e_0_list, start=1):
    caller.add_call(Calculate(e_0, i, len(e_0_list)))

caller.wait_calls()

# Save the list of electron energies
with open(os.path.join(output_dir, "list.txt"), 'w') as f:
    for e_0 in e_0_list:
        f.write("%.f\n" % e_0)


# Additional example: load and show one of the saved spectra

import pickle
from xpecgen.xpecgen import Spectrum  # Needed to unpickle the spectra

import matplotlib.pyplot as plt

for i, e_0 in enumerate(e_0_list, start=1):
    with open(os.path.join(output_dir, "S%d.pkl" % i), 'rb') as f:
        s = pickle.load(f)
    xx, yy = s.get_points()
    plt.semilogy(xx, yy, label="%.2f"% e_0)

plt.xlim(0, e_0_list[-1])
plt.legend()
plt.show(block=True)
