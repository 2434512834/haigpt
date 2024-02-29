import multiprocessing

def print_cpu_cores():
    print(f"CPU has {multiprocessing.cpu_count()} cores")

print_cpu_cores()