from Crypto.Cipher import AES
from Crypto.Random import get_random_bytes
import os

def encrypt_file_aes_ctr(input_path, output_path, key_path='key.bin', nonce_path='nonce.bin'):
    with open(input_path, 'rb') as f:
        data = f.read()

    key = get_random_bytes(32)
    nonce = get_random_bytes(8)

    cipher = AES.new(key, AES.MODE_CTR, nonce=nonce)
    ciphertext = cipher.encrypt(data)

    with open(output_path, 'wb') as f:
        f.write(ciphertext)

def main():
    input_file = 'output.bin'      # <-- zamień na swój plik
    encrypted_file = 'aes_256_output.bin'
    encrypt_file_aes_ctr(input_file, encrypted_file)

if __name__ == '__main__':
    main()
