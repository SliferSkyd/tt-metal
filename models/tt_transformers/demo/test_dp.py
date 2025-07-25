import time

import pytest

import ttnn


@pytest.mark.parametrize("mesh_device", [(1, 2)], indirect=True)
def test_dp(mesh_device):
    print("Testing distributed processing with mesh device:", mesh_device)
    submeshes = mesh_device.create_submeshes(ttnn.MeshShape(1, 1))
    for submesh in submeshes:
        print(submesh)

    # start two processes to test distributed processing
    import multiprocessing
    import os

    def worker(submesh):
        print(f"Worker started on submesh: {submesh}")
        # Simulate some work
        for i in range(5000):
            print(f"Worker {os.getpid()} on submesh {submesh} is working... {i}")
            time.sleep(0.01)
        print(f"Worker finished on submesh: {submesh}")

    processes = []
    for submesh in submeshes:
        p = multiprocessing.Process(target=worker, args=(submesh,))
        processes.append(p)
        p.start()

    for p in processes:
        p.join()

    print("All workers finished.")
