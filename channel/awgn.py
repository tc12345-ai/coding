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
    
    def __init__(self, snr_db: float = 10.0,
                 bits_per_symbol: int = 1,
                 code_rate: float = 1.0,
                 symbol_energy: float = 1.0):
        """
        初始化 AWGN 信道
        
        Args:
            snr_db: 信噪比 (dB)，基于 Eb/N0
            bits_per_symbol: 每个符号携带的比特数
            code_rate: 码率 (k/n)
            symbol_energy: 符号能量 (默认 1.0)
        """
        self.snr_db = snr_db
        self.bits_per_symbol = bits_per_symbol
        self.code_rate = code_rate
        self.symbol_energy = symbol_energy
        self._update_noise_power()
        
    def _update_noise_power(self):
        """根据 SNR 更新噪声功率"""
        # SNR = Eb/N0 (dB)
        # Es/N0 = Eb/N0 * bits_per_symbol / code_rate
        # sigma^2 = Es / (2 * Es/N0)
        snr_linear = 10 ** (self.snr_db / 10)
        esn0_linear = snr_linear * (self.bits_per_symbol / self.code_rate)
        self.noise_variance = self.symbol_energy / (2 * esn0_linear)
        self.noise_std = np.sqrt(self.noise_variance)
        
    def set_snr(self, snr_db: float,
                bits_per_symbol: int = None,
                code_rate: float = None,
                symbol_energy: float = None):
        """设置新的 SNR"""
        self.snr_db = snr_db
        if bits_per_symbol is not None:
            self.bits_per_symbol = bits_per_symbol
        if code_rate is not None:
            self.code_rate = code_rate
        if symbol_energy is not None:
            self.symbol_energy = symbol_energy
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
