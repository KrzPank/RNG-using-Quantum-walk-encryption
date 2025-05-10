import matplotlib
matplotlib.use('Agg')
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
    plt.title('Empiryczny rozklad zmiennych losowych po post-procesingu')
    plt.xlabel('Bajt')
    plt.ylabel('Częstość')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(output_path)
    print(f"Histogram zapisany jako: {output_path}")

def main():
    path = 'source.bin'
    data = read_binary_file(path)
    entropy = calculate_entropy(data)
    print(f"Entropia pliku: {entropy:.10f} bitów na bajt")
    plot_histogram(data, "images/source_histogram.png")
    
    path = 'post.bin'
    data = read_binary_file(path)
    entropy = calculate_entropy(data)
    print(f"Entropia pliku: {entropy:.10f} bitów na bajt")
    plot_histogram(data, "images/post_histogram.png")

    path2 = 'aes.bin'
    data2 = read_binary_file(path2)
    entropy = calculate_entropy(data2)
    print(f"Entropia pliku: {entropy:.10f} bitów na bajt")
    plot_histogram(data2, "images/aes_histogram.png")



if __name__ == "__main__":
    main()
