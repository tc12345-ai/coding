"""
Convolutional Encoder
卷积码编码器

支持任意约束长度和码率的卷积码编码
"""

import numpy as np
from typing import List, Tuple


class ConvolutionalEncoder:
    """
    卷积码编码器
    
    Parameters:
        constraint_length: 约束长度 K
        generators: 生成多项式列表（八进制表示）
        
    Example:
        # (2,1,7) 卷积码，生成多项式 [171, 133]
        encoder = ConvolutionalEncoder(7, [0o171, 0o133])
        encoded = encoder.encode(data_bits)
    """
    
    def __init__(self, constraint_length: int = 7, 
                 generators: List[int] = None):
        """
        初始化卷积码编码器
        
        Args:
            constraint_length: 约束长度 K (默认 7)
            generators: 生成多项式列表，八进制 (默认 [171, 133])
        """
        self.K = constraint_length
        self.generators = generators if generators else [0o171, 0o133]
        self.n_outputs = len(self.generators)
        self.rate = 1.0 / self.n_outputs
        self.n_states = 2 ** (self.K - 1)
        
        # 预计算状态转移表和输出表
        self._build_trellis()
        
    def _build_trellis(self):
        """构建网格图（状态转移表和输出表）"""
        self.next_state = np.zeros((self.n_states, 2), dtype=np.int32)
        self.output = np.zeros((self.n_states, 2, self.n_outputs), dtype=np.int32)
        
        for state in range(self.n_states):
            for input_bit in range(2):
                # 计算下一状态
                next_s = (state >> 1) | (input_bit << (self.K - 2))
                self.next_state[state, input_bit] = next_s
                
                # 计算输出
                register = (state << 1) | input_bit
                for i, gen in enumerate(self.generators):
                    out = 0
                    temp = register & gen
                    while temp:
                        out ^= (temp & 1)
                        temp >>= 1
                    self.output[state, input_bit, i] = out
                    
    def encode(self, bits: np.ndarray, terminate: bool = True) -> np.ndarray:
        """
        编码输入比特序列
        
        Args:
            bits: 输入比特数组
            terminate: 是否添加尾比特使编码器回到零状态
            
        Returns:
            编码后的比特数组
        """
        bits = np.asarray(bits, dtype=np.int32)
        
        # 添加尾比特
        if terminate:
            tail = np.zeros(self.K - 1, dtype=np.int32)
            bits = np.concatenate([bits, tail])
            
        # 编码
        n_bits = len(bits)
        output = np.zeros(n_bits * self.n_outputs, dtype=np.int32)
        
        state = 0
        for i, bit in enumerate(bits):
            for j in range(self.n_outputs):
                output[i * self.n_outputs + j] = self.output[state, bit, j]
            state = self.next_state[state, bit]
            
        return output
    
    def get_trellis_info(self) -> dict:
        """
        获取网格图信息（用于可视化）
        
        Returns:
            包含状态转移和输出信息的字典
        """
        return {
            'n_states': self.n_states,
            'next_state': self.next_state.copy(),
            'output': self.output.copy(),
            'constraint_length': self.K,
            'generators': self.generators,
            'rate': self.rate
        }
    
    def __repr__(self):
        gens_oct = [oct(g) for g in self.generators]
        return f"ConvolutionalEncoder(K={self.K}, generators={gens_oct}, rate=1/{self.n_outputs})"
