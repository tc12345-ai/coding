"""
AWGN Channel
加性高斯白噪声信道
"""

import numpy as np
from typing import Tuple


class AWGNChannel:
    """
    AWGN 信道
    
    y = x + n, where n ~ N(0, sigma^2)
    
    Parameters:
        snr_db: 信噪比 (dB)
        
    Example:
        channel = AWGNChannel(snr_db=5.0)
        received = channel.transmit(symbols)
    """
    
    def __init__(self, snr_db: float = 10.0):
        """
        初始化 AWGN 信道
        
        Args:
            snr_db: 信噪比 (dB)，基于 Eb/N0
        """
        self.snr_db = snr_db
        self._update_noise_power()
        
    def _update_noise_power(self):
        """根据 SNR 更新噪声功率"""
        # SNR = Eb/N0 (dB)
        # 对于 BPSK：Es = Eb, 所以 sigma^2 = N0/2 = Es / (2 * SNR_linear)
        snr_linear = 10 ** (self.snr_db / 10)
        self.noise_variance = 1.0 / (2 * snr_linear)
        self.noise_std = np.sqrt(self.noise_variance)
        
    def set_snr(self, snr_db: float):
        """设置新的 SNR"""
        self.snr_db = snr_db
        self._update_noise_power()
        
    def transmit(self, symbols: np.ndarray) -> np.ndarray:
        """
        通过 AWGN 信道传输
        
        Args:
            symbols: 发送符号（实数或复数）
            
        Returns:
            received: 接收信号
        """
        symbols = np.asarray(symbols)
        
        if np.iscomplexobj(symbols):
            # 复数信号：实部和虚部分别加噪声
            noise = (np.random.randn(*symbols.shape) + 
                    1j * np.random.randn(*symbols.shape)) * self.noise_std / np.sqrt(2)
        else:
            # 实数信号
            noise = np.random.randn(*symbols.shape) * self.noise_std
            
        return symbols + noise
    
    def compute_llr(self, received: np.ndarray, 
                    symbol_power: float = 1.0) -> np.ndarray:
        """
        计算 BPSK 接收信号的 LLR
        
        LLR = log(P(x=0|y) / P(x=1|y)) = 2y / sigma^2
        
        其中 x=0 对应符号 +1，x=1 对应符号 -1
        
        Args:
            received: 接收信号
            symbol_power: 符号功率（默认 1.0）
            
        Returns:
            llr: 对数似然比
        """
        # LLR = 2 * y / sigma^2 (假设符号为 +1/-1)
        llr = 2 * received / self.noise_variance
        return llr
    
    def get_info(self) -> dict:
        """获取信道参数"""
        return {
            'snr_db': self.snr_db,
            'noise_variance': self.noise_variance,
            'noise_std': self.noise_std
        }
    
    def __repr__(self):
        return f"AWGNChannel(SNR={self.snr_db:.1f}dB, σ²={self.noise_variance:.4f})"
