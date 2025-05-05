import matplotlib
matplotlib.use('Agg')  # <-- to rozwiązuje problem z init.tcl
import matplotlib.pyplot as plt
import math
from collections import Counter

def read_binary_file(path):
    with open(path, 'rb') as file:
        return file.read()

def calculate_entropy(byte_data):
    byte_counts = Counter(byte_data)
    total_bytes = len(byte_data)
    entropy = -sum((count / total_bytes) * math.log2(count / total_bytes) 
                   for count in byte_counts.values())
    return entropy

def plot_histogram(byte_data, output_path='histogram.png'):
    byte_counts = Counter(byte_data)
    values = [byte_counts.get(i, 0) for i in range(256)]
    
    plt.figure(figsize=(12, 6))
    plt.bar(range(256), values, width=1.0, edgecolor='black')
    plt.title('Histogram częstości bajtów (0-255)')
    plt.xlabel('Bajt')
    plt.ylabel('Częstość')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(output_path)
    print(f"Histogram zapisany jako: {output_path}")

def main():
    path = 'output.bin'  # <- Zamień na ścieżkę do swojego pliku
    data = read_binary_file(path)
    entropy = calculate_entropy(data)
    print(f"Entropia pliku: {entropy:.10f} bitów na bajt")
    plot_histogram(data)

    path2 = 'aes_256_output.bin'  # <- Zamień na ścieżkę do swojego pliku
    data2 = read_binary_file(path2)
    entropy = calculate_entropy(data2)
    print(f"Entropia pliku: {entropy:.10f} bitów na bajt")
    plot_histogram(data2, "aes_histogram.png")

if __name__ == "__main__":
    main()
