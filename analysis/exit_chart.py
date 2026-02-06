"""
EXIT Chart Analyzer
EXIT 曲线分析（简化版）

分析译码器的外部信息特性
"""

import numpy as np
from typing import List, Tuple, Dict, Optional


class EXITChartAnalyzer:
    """
    EXIT 曲线分析器（简化版）
    
    通过蒙特卡洛方法估计互信息
    
    Example:
        analyzer = EXITChartAnalyzer()
        I_A, I_E = analyzer.compute_exit_point(decoder, snr_db=3.0)
    """
    
    def __init__(self, n_samples: int = 10000):
        """
        初始化 EXIT 分析器
        
        Args:
            n_samples: 蒙特卡洛样本数
        """
        self.n_samples = n_samples
        
    def compute_mutual_information(self, llr: np.ndarray, 
                                   transmitted: np.ndarray = None) -> float:
        """
        估计 LLR 与发送比特的互信息
        
        使用公式：I ≈ 1 - E[log2(1 + exp(-s * LLR))]
        其中 s = 1 - 2*x (x 为发送比特)
        
        Args:
            llr: 对数似然比
            transmitted: 发送比特（如果已知）
            
        Returns:
            mutual_information: 互信息 [0, 1]
        """
        llr = np.asarray(llr)
        
        if transmitted is not None:
            # 使用实际发送比特
            s = 1 - 2 * np.asarray(transmitted)
            signed_llr = s * llr
        else:
            # 假设等概率发送
            signed_llr = np.abs(llr)
            
        # 避免数值问题
        signed_llr = np.clip(signed_llr, -50, 50)
        
        # I = 1 - E[log2(1 + exp(-signed_llr))]
        mutual_info = 1 - np.mean(np.log2(1 + np.exp(-signed_llr)))
        
        return float(np.clip(mutual_info, 0, 1))
    
    def generate_apriori_llr(self, bits: np.ndarray, 
                            mutual_info: float) -> np.ndarray:
        """
        生成具有指定互信息的先验 LLR
        
        使用高斯模型：LLR ~ N(σ²/2 * s, σ²)
        其中 σ² = J^(-1)(I_A) 通过查表或迭代获得
        
        Args:
            bits: 发送比特
            mutual_info: 目标互信息 I_A
            
        Returns:
            a_priori_llr: 先验 LLR
        """
        if mutual_info <= 0:
            return np.zeros_like(bits, dtype=np.float64)
        if mutual_info >= 1:
            # 完美信息
            s = 1 - 2 * bits
            return s * 100  # 大 LLR
            
        # 使用 J 函数的近似逆
        # 对于高斯信道：I = J(σ) ≈ 1 - exp(-0.3037*σ² - 0.8935*σ)
        # 反解 σ
        sigma = self._inverse_j_function(mutual_info)
        
        # 生成高斯 LLR
        s = 1 - 2 * np.asarray(bits)
        mean = sigma**2 / 2 * s
        llr = mean + sigma * np.random.randn(len(bits))
        
        return llr
    
    def _inverse_j_function(self, I: float) -> float:
        """
        J 函数的近似逆
        
        J(σ) = 1 - ∫ exp(-LLR) / (1 + exp(-LLR)) * N(LLR; σ²/2, σ²) dLLR
        
        使用数值查找
        """
        if I <= 0:
            return 0.0
        if I >= 1:
            return 20.0
            
        # 二分搜索
        sigma_low, sigma_high = 0.0, 20.0
        
        for _ in range(50):
            sigma_mid = (sigma_low + sigma_high) / 2
            I_mid = self._j_function(sigma_mid)
            
            if I_mid < I:
                sigma_low = sigma_mid
            else:
                sigma_high = sigma_mid
                
            if abs(I_mid - I) < 1e-6:
                break
                
        return sigma_mid
    
    def _j_function(self, sigma: float, n_samples: int = 10000) -> float:
        """
        J 函数：计算高斯 LLR 的互信息
        """
        if sigma <= 0:
            return 0.0
        if sigma >= 20:
            return 1.0
            
        # 蒙特卡洛估计
        mean = sigma**2 / 2
        llr = mean + sigma * np.random.randn(n_samples)
        
        # I = 1 - E[log2(1 + exp(-LLR))]
        llr = np.clip(llr, -50, 50)
        I = 1 - np.mean(np.log2(1 + np.exp(-llr)))
        
        return float(np.clip(I, 0, 1))
    
    def compute_decoder_exit(self, 
                            decoder,
                            code_length: int,
                            info_length: int,
                            snr_db: float,
                            I_A_range: np.ndarray = None,
                            n_trials: int = 100) -> Tuple[np.ndarray, np.ndarray]:
        """
        计算译码器的 EXIT 曲线
        
        Args:
            decoder: BP 译码器
            code_length: 码字长度
            info_length: 信息位长度
            snr_db: 信噪比
            I_A_range: 先验互信息范围
            n_trials: 每个点的试验次数
            
        Returns:
            I_A: 先验互信息数组
            I_E: 外部互信息数组
        """
        if I_A_range is None:
            I_A_range = np.linspace(0.01, 0.99, 20)
            
        I_E_list = []
        
        # 信道 LLR 的 sigma
        snr_linear = 10 ** (snr_db / 10)
        channel_sigma = np.sqrt(2 * snr_linear)
        
        for I_A in I_A_range:
            I_E_samples = []
            
            for _ in range(n_trials):
                # 生成随机比特
                bits = np.random.randint(0, 2, info_length)
                
                # 这里简化处理，直接生成码字的 LLR
                # 实际应用中需要完整的编码过程
                codeword = np.random.randint(0, 2, code_length)  # 简化
                
                # 信道 LLR
                s = 1 - 2 * codeword
                channel_llr = s * channel_sigma**2 / 2 + channel_sigma * np.random.randn(code_length)
                
                # 先验 LLR
                apriori_llr = self.generate_apriori_llr(codeword, I_A)
                
                # 总输入 LLR
                input_llr = channel_llr + apriori_llr
                
                # 一次迭代
                try:
                    decoded, history = decoder.decode(input_llr, early_stop=False)
                    
                    if history and len(history) > 0:
                        # 获取后验 LLR
                        posterior = history[0].get('posterior_llr', input_llr)
                        
                        # 外部信息 = 后验 - 输入
                        extrinsic = posterior - input_llr
                        
                        I_E = self.compute_mutual_information(extrinsic, codeword)
                        I_E_samples.append(I_E)
                except:
                    continue
                    
            if I_E_samples:
                I_E_list.append(np.mean(I_E_samples))
            else:
                I_E_list.append(0)
                
        return I_A_range, np.array(I_E_list)
    
    def compute_channel_exit(self, snr_range: np.ndarray,
                            n_samples: int = 10000) -> Tuple[np.ndarray, np.ndarray]:
        """
        计算 AWGN 信道的 EXIT 曲线（互信息 vs SNR）
        
        Args:
            snr_range: SNR 范围 (dB)
            n_samples: 样本数
            
        Returns:
            snr_range: SNR 数组
            I_ch: 信道互信息数组
        """
        I_ch_list = []
        
        for snr_db in snr_range:
            snr_linear = 10 ** (snr_db / 10)
            sigma = np.sqrt(1 / (2 * snr_linear))
            
            # 生成 BPSK 符号和接收信号
            bits = np.random.randint(0, 2, n_samples)
            symbols = 1 - 2 * bits
            received = symbols + sigma * np.random.randn(n_samples)
            
            # 计算 LLR
            llr = 2 * received / sigma**2
            
            # 计算互信息
            I_ch = self.compute_mutual_information(llr, bits)
            I_ch_list.append(I_ch)
            
        return snr_range, np.array(I_ch_list)
