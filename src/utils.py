from os import urandom
from config import default_para as config
from hashlib import sha256
import platform

if platform.system() == 'Linux':
    from Cryptodome.Cipher import AES
elif platform.system() == 'Darwin':  # For macOS
    from Crypto.Cipher import AES
else:
    from Crypto.Cipher import AES

KEY_SIZE = 32
BLOCK_SIZE = 4096

output_dir = config['output_dir']

# Pseudo Random Function used for generating service ID or secret key
def PRF(seed):
    if type(seed) == str:
        return sha256(seed.encode('utf8')).digest()
    elif type(seed) == bytes:
        return sha256(seed).digest()
    else:
        raise ValueError(f'Invalid type of seed:\t{type(seed)}')


def Hash(input):
    assert type(input) == bytes
    return sha256(input).digest()


def XOR_blocks(blocks):
    result = blocks[0]
    for i in range(1, len(blocks)):
        result = bytes(a ^ b for a, b in zip(result, blocks[i]))
    return result


def Encrypt(key_bytes, bytes):
    myCipher = AES.new(key_bytes, AES.MODE_CTR)
    encrypted = myCipher.encrypt(bytes)
    nonce = myCipher.nonce
    return nonce + encrypted


def Decrypt(key_bytes, bytes):
    nonce = bytes[:8]
    cipher_text = bytes[8:]
    myCipher = AES.new(key_bytes, AES.MODE_CTR, nonce=nonce)
    decrypted = myCipher.decrypt(cipher_text)
    return decrypted

# Encoder class for encoding and decoding data, basically includes all above functions
class Encoder():
    supported_sids = {
        PRF('path'): 'path',
        PRF('ring'): 'ring',
        PRF('concur'): 'concur'
    }

    def __init__(self, mk=urandom(32)):
        self.mk = mk

    def judge_sid(self, sid):
        if sid not in self.supported_sids.keys():
            raise ValueError(f'Unsupported sid:\t{sid}')

    def Enc(self, data, sid):
        self.judge_sid(PRF(sid))
        sk = PRF(self.mk + PRF(sid))
        return PRF(sid) + Encrypt(sk, data)

    def Dec(self, cipher):
        sid = cipher[:32]
        self.judge_sid(sid)
        sk = PRF(self.mk + sid)
        data = cipher[32:]
        return Decrypt(sk, data)

    def service_of(self, cipher):
        return self.supported_sids[cipher[:32]]

if __name__ == '__main__':
    data = urandom(4096)
    mk = urandom(32)
    encoder = Encoder(mk)
    block = encoder.Enc(data, 'path')
    print(encoder.service_of(block))
    result = encoder.Dec(block)
    assert result == data
    new_block = encoder.Enc(result, 'concur')
    print(encoder.service_of(new_block))
