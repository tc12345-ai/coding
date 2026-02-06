"""
核心功能测试脚本
"""
import sys
sys.path.insert(0, '.')
import numpy as np
np.random.seed(42)

print('=' * 50)
print('信道编码核心功能测试')
print('=' * 50)

# 测试卷积码
print('\n1. 卷积码测试')
from encoders import ConvolutionalEncoder
from decoders import ViterbiDecoder

encoder = ConvolutionalEncoder(7, [0o171, 0o133])
decoder = ViterbiDecoder(7, [0o171, 0o133])

bits = np.random.randint(0, 2, 50)
coded = encoder.encode(bits)
print(f'   编码: {len(bits)} 信息比特 -> {len(coded)} 编码比特')
print(f'   码率: {encoder.rate:.3f}')

# 无噪声译码
symbols = (1 - 2 * coded).astype(float)
decoded, stats = decoder.decode(symbols, mode='soft')
ber = np.mean(bits != decoded[:len(bits)])
print(f'   无噪声 BER: {ber}')

# 有噪声译码
noise = np.random.randn(len(symbols)) * 0.5
received = symbols + noise
decoded_noisy, _ = decoder.decode(received, mode='soft')
ber_noisy = np.mean(bits != decoded_noisy[:len(bits)])
print(f'   有噪声 BER (SNR~6dB): {ber_noisy}')

# 测试 LDPC
print('\n2. LDPC 码测试')
from encoders import LDPCCode
from decoders import BeliefPropagationDecoder

ldpc = LDPCCode(n=96, k=48, seed=123)
bp_decoder = BeliefPropagationDecoder(ldpc.H, max_iter=50)

print(f'   码参数: n={ldpc.n}, k={ldpc.k}, rate={ldpc.rate:.3f}')

info = np.random.randint(0, 2, ldpc.k)
codeword = ldpc.encode(info)
check_ok = ldpc.check(codeword)
print(f'   编码校验: {"通过" if check_ok else "失败"}')

# BP 译码 (高 SNR)
llr = (1 - 2 * codeword) * 8.0
decoded, history = bp_decoder.decode(llr)
bp_ok = ldpc.check(decoded)
print(f'   BP 译码 (高SNR): 迭代{len(history)}次, 校验{"通过" if bp_ok else "失败"}')

# BP 译码 (中等 SNR)
noise = np.random.randn(ldpc.n)
llr_noisy = (1 - 2 * codeword) * 4.0 + noise
decoded_noisy, history_noisy = bp_decoder.decode(llr_noisy)
bp_ok_noisy = ldpc.check(decoded_noisy)
ber_ldpc = np.mean(info != decoded_noisy[:ldpc.k])
print(f'   BP 译码 (中SNR): 迭代{len(history_noisy)}次, BER={ber_ldpc:.4f}')

print('\n' + '=' * 50)
print('核心功能测试完成!')
print('=' * 50)
