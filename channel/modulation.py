"""
Modulation Schemes
调制模块
"""

import numpy as np
from typing import Tuple


class BPSKModulator:
    """
    BPSK 调制器
    
    0 -> +1
    1 -> -1
    
    Example:
        mod = BPSKModulator()
        symbols = mod.modulate(bits)
        bits_hat = mod.demodulate(received)
    """
    
    def __init__(self):
        """初始化 BPSK 调制器"""
        self.bits_per_symbol = 1
        self.constellation = np.array([1, -1])
        
    def modulate(self, bits: np.ndarray) -> np.ndarray:
        """
        BPSK 调制
        
        Args:
            bits: 输入比特 (0/1)
            
        Returns:
            symbols: BPSK 符号 (+1/-1)
        """
        bits = np.asarray(bits, dtype=np.int32)
        return 1 - 2 * bits  # 0 -> +1, 1 -> -1
    
    def demodulate(self, received: np.ndarray) -> np.ndarray:
        """
        BPSK 硬判决解调
        
        Args:
            received: 接收信号
            
        Returns:
            bits: 解调比特
        """
        return (received < 0).astype(np.int32)
    
    def soft_demodulate(self, received: np.ndarray, 
                        noise_var: float = 1.0) -> np.ndarray:
        """
        BPSK 软解调，返回 LLR
        
        Args:
            received: 接收信号
            noise_var: 噪声方差
            
        Returns:
            llr: 对数似然比
        """
        # LLR = 2y / sigma^2
        return 2 * received / noise_var
    
    def __repr__(self):
        return "BPSKModulator()"


class QPSKModulator:
    """
    QPSK 调制器
    
    使用 Gray 映射:
    00 -> +1+1j
    01 -> +1-1j  
    10 -> -1+1j
    11 -> -1-1j
    
    归一化功率为 1
    """
    
    def __init__(self):
        """初始化 QPSK 调制器"""
        self.bits_per_symbol = 2
        # Gray 编码星座点（归一化）
        self.constellation = np.array([
            1 + 1j,   # 00
            1 - 1j,   # 01
            -1 + 1j,  # 10
            -1 - 1j   # 11
        ]) / np.sqrt(2)
        
    def modulate(self, bits: np.ndarray) -> np.ndarray:
        """
        QPSK 调制
        
        Args:
            bits: 输入比特（长度必须为偶数）
            
        Returns:
            symbols: QPSK 符号（复数）
        """
        bits = np.asarray(bits, dtype=np.int32)
        
        if len(bits) % 2 != 0:
            raise ValueError("QPSK 需要偶数个比特")
            
        # 每两个比特一组
        n_symbols = len(bits) // 2
        indices = bits[0::2] * 2 + bits[1::2]
        
        return self.constellation[indices]
    
    def demodulate(self, received: np.ndarray) -> np.ndarray:
        """
        QPSK 硬判决解调
        
        Args:
            received: 接收复数信号
            
        Returns:
            bits: 解调比特
        """
        bits = np.zeros(len(received) * 2, dtype=np.int32)
        
        # 实部决定第一个比特
        bits[0::2] = (np.real(received) < 0).astype(np.int32)
        # 虚部决定第二个比特
        bits[1::2] = (np.imag(received) < 0).astype(np.int32)
        
        return bits
    
    def soft_demodulate(self, received: np.ndarray,
                        noise_var: float = 1.0) -> np.ndarray:
        """
        QPSK 软解调，返回 LLR
        
        Args:
            received: 接收复数信号
            noise_var: 噪声方差
            
        Returns:
            llr: 对数似然比（每个符号两个 LLR）
        """
        llr = np.zeros(len(received) * 2)
        
        # 实部和虚部独立解调
        # 归一化后符号为 ±1/sqrt(2)，等效 SNR 需要调整
        llr[0::2] = 2 * np.sqrt(2) * np.real(received) / noise_var
        llr[1::2] = 2 * np.sqrt(2) * np.imag(received) / noise_var
        
        return llr
    
    def __repr__(self):
        return "QPSKModulator()"
