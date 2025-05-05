import numpy as np
from PIL import Image
from scipy.ndimage import zoom
from qiskit import QuantumCircuit, transpile
from qiskit.quantum_info import Statevector, Operator
import os


def placholder_quantum_walks(T, r, alpha, beta):   # Quantum walks TO DO
    np.random.seed(0)                       # rand for tests
    P = np.random.rand(T)
    P /= np.sum(P)
    return P


def quantum_walks(T, r, alpha, beta):
    from math import ceil, log2, pi

    assert T % 2 == 1, "T must be odd"
    n_pos = ceil(log2(T))  # Number of position qubits
    n_total = 1 + n_pos     # 1 coin qubit + position qubits

    qc = QuantumCircuit(n_total)

    qc.ry(2 * alpha, 0)

    def apply_coin(qc, coin_qubit):
        qc.ry(2 * beta, coin_qubit)  # Simple rotation coin

    def apply_shift(qc, coin_qubit, pos_qubits):
        for i in range(len(pos_qubits)):
            qc.cx(coin_qubit, pos_qubits[i])
            if i < len(pos_qubits) - 1:
                qc.ccx(coin_qubit, pos_qubits[i], pos_qubits[i+1])

    pos_qubits = list(range(1, n_total))  # position qubits (after coin)

    for _ in range(r):
        apply_coin(qc, 0)
        apply_shift(qc, 0, pos_qubits)

    # Simulate final state
    final_state = Statevector.from_instruction(qc)

    probs = final_state.probabilities_dict()

    # Aggregate probabilities over position basis (ignoring coin qubit)
    position_probs = np.zeros(T)
    for bitstr, prob in probs.items():
        pos_idx = int(bitstr[1:], 2)  # skip coin qubit
        if pos_idx < T:
            position_probs[pos_idx] += prob

    position_probs /= np.sum(position_probs)
    return position_probs


def generate_permutation_boxes(BP):
    PL = BP[:128]
    PR = BP[128:256]

    L = np.argsort(PL)
    R = np.argsort(PR)

    P_box1 = np.argsort(L)
    P_box2 = np.argsort(R)
    return P_box1, P_box2


def create_keys(BP, P, h, w):
    K = np.fix(BP * 1e8).astype(int) % 256
    KeyP = zoom(P, (h * w / len(P)))
    Key = np.fix(KeyP * 1e8).astype(int) % 256
    Key = Key.reshape(h, w)
    return K, Key


def divide_into_blocks(img_array, block_size=16):
    h, w = img_array.shape
    blocks = []
    for i in range(0, h, block_size):
        for j in range(0, w, block_size):
            blocks.append(img_array[i:i+block_size, j:j+block_size])
    return blocks


def encrypt_block(block, K, P_box1, P_box2):
    flat_block = block.flatten()
    LB = flat_block[:128]
    RB = flat_block[128:]

    KR = K[128:]
    KL = K[:128]

    RB1 = np.bitwise_xor(RB, KR)

    LB1 = LB[P_box1]
    RB2 = RB1[P_box2]

    LB2 = np.bitwise_xor(LB1, KL)

    Encblock = np.concatenate((LB2, RB2)).reshape((16, 16))
    return Encblock


def generate_binary_data_from_image(image_path, T, r, alpha, beta):
    image = Image.open(image_path).convert("L")
    img_array = np.array(image)
    h, w = img_array.shape

    P = quantum_walks(T, r, alpha, beta)

    BP = zoom(P, (256 / len(P)))

    P_box1, P_box2 = generate_permutation_boxes(BP)
    K, Key = create_keys(BP, P, h, w)
    blocks = divide_into_blocks(img_array)

    encrypted_blocks = []
    for block in blocks:
        enc_block = encrypt_block(block, K, P_box1, P_box2)
        encrypted_blocks.append(enc_block)

    encrypted_image = np.zeros((h, w), dtype=np.uint8)
    idx = 0
    for i in range(0, h, 16):
        for j in range(0, w, 16):
            encrypted_image[i:i+16, j:j+16] = encrypted_blocks[idx]
            idx += 1

    ciphered_image = np.bitwise_xor(encrypted_image, Key)
    return ciphered_image.astype(np.uint8).flatten().tobytes()

#binary_output = generate_binary_data_from_image("tiles_output/tile_0_0.png", T=51, r=100, alpha=np.pi/4, beta=np.pi/3)
#print(binary_output[:512])  # Show first 512 bits
#print(f"Total length: {len(binary_output)/8} bytes")


def chop_image_to_tiles(image_path, output_folder, offset_x=0, offset_y=0, tile_size=512):
    image = Image.open(image_path).convert("L")
    width, height = image.size

    num_tiles_x = width // tile_size
    num_tiles_y = height // tile_size

    tile_count = 0
    for y in range(num_tiles_y):
        for x in range(num_tiles_x):
            left = x * tile_size
            top = y * tile_size
            right = left + tile_size
            bottom = top + tile_size
            tile = image.crop((left, top, right, bottom))
            tile.save(os.path.join(output_folder, f"tile_{y + offset_y}_{x + offset_x}.png"))
            tile_count += 1
    
    print(f"Saved {tile_count} tiles of size {tile_size}x{tile_size}, from: tile_{offset_y}_{offset_x}, to: tile_{num_tiles_y + offset_y}_{num_tiles_x + offset_x} ")
    return num_tiles_x + offset_x, num_tiles_y + offset_y


def save_images_to_bin(folder_path, output_folder, T, r, alpha, beta):
    for filename in os.listdir(folder_path):
        if filename.lower().endswith('.png'):
            input_path = os.path.join(folder_path, filename)
            print(f"Processing: {filename}")
            base_name = os.path.splitext(filename)[0]
            out_path = os.path.join(output_folder, f"{base_name}.bin")
            with open(out_path, "wb") as f_out:
                binary_data = generate_binary_data_from_image(input_path, T, r, alpha, beta)
                f_out.write(binary_data)


def merge(input_folder, output_file):
    with open(output_file, "wb") as out_f:
        for filename in sorted(os.listdir(input_folder)):
            if filename.lower().endswith('.bin'):
                file_path = os.path.join(input_folder, filename)
                print(f"Merging: {filename}")

                with open(file_path, "rb") as in_f:
                    data = in_f.read()
                    out_f.write(data)

    print(f"\n Merged")

#'''
off_x1, off_y1 = chop_image_to_tiles("images/cat1.jpg", "tiles_output/")
off_x2, off_y2 = chop_image_to_tiles("images/cat2.jpg", "tiles_output/", off_x1 + 1, off_y1 + 1 )
#chop_image_to_tiles("images/cat3.jpg", "tiles_output/", off_x2 + 1, off_y2 + 1)

save_images_to_bin("tiles_output/", "output/", T=51, r=100, alpha=np.pi/4, beta=np.pi/3)
merge("output/", "output.bin")
#'''
