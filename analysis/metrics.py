"""
Performance Metrics
性能指标计算

BER (Bit Error Rate) 和 BLER (Block Error Rate) 计算
"""

import numpy as np
from typing import List, Dict, Tuple, Callable, Optional
from dataclasses import dataclass


@dataclass
class SimulationResult:
    """仿真结果"""
    snr_db: float
    n_bits: int
    n_errors: int
    n_blocks: int
    n_block_errors: int
    ber: float
    bler: float
    avg_iterations: float = 0.0


class BERCalculator:
    """
    BER 计算器
    
    累积计算误比特率
    """
    
    def __init__(self):
        self.reset()
        
    def reset(self):
        """重置统计"""
        self.total_bits = 0
        self.error_bits = 0
        
    def update(self, transmitted: np.ndarray, received: np.ndarray):
        """
        更新统计
        
        Args:
            transmitted: 发送比特
            received: 接收/译码比特
        """
        transmitted = np.asarray(transmitted)
        received = np.asarray(received)
        
        min_len = min(len(transmitted), len(received))
        errors = np.sum(transmitted[:min_len] != received[:min_len])
        
        self.total_bits += min_len
        self.error_bits += errors
        
    def get_ber(self) -> float:
        """获取当前 BER"""
        if self.total_bits == 0:
            return 0.0
        return self.error_bits / self.total_bits
    
    def get_stats(self) -> dict:
        """获取详细统计"""
        return {
            'total_bits': self.total_bits,
            'error_bits': self.error_bits,
            'ber': self.get_ber()
        }


class BLERCalculator:
    """
    BLER 计算器
    
    累积计算误块率
    """
    
    def __init__(self):
        self.reset()
        
    def reset(self):
        """重置统计"""
        self.total_blocks = 0
        self.error_blocks = 0
        
    def update(self, transmitted: np.ndarray, received: np.ndarray):
        """
        更新统计（一次调用 = 一个块）
        
        Args:
            transmitted: 发送比特
            received: 接收/译码比特
        """
        transmitted = np.asarray(transmitted)
        received = np.asarray(received)
        
        min_len = min(len(transmitted), len(received))
        has_error = np.any(transmitted[:min_len] != received[:min_len])
        
        self.total_blocks += 1
        if has_error:
            self.error_blocks += 1
            
    def get_bler(self) -> float:
        """获取当前 BLER"""
        if self.total_blocks == 0:
            return 0.0
        return self.error_blocks / self.total_blocks
    
    def get_stats(self) -> dict:
        """获取详细统计"""
        return {
            'total_blocks': self.total_blocks,
            'error_blocks': self.error_blocks,
            'bler': self.get_bler()
        }


class PerformanceSimulator:
    """
    性能仿真器
    
    运行完整的编码-信道-译码链路仿真
    
    Example:
        sim = PerformanceSimulator()
        results = sim.run_simulation(
            encoder=conv_encoder,
            decoder=viterbi_decoder,
            snr_range=np.arange(0, 10, 1),
            n_frames=1000
        )
    """
    
    def __init__(self):
        self.results: List[SimulationResult] = []
        self._stop_flag = False
        
    def stop(self):
        """停止仿真"""
        self._stop_flag = True
        
    def run_simulation(self,
                      encode_func: Callable,
                      decode_func: Callable,
                      modulate_func: Callable,
                      demodulate_func: Callable,
                      channel_func: Callable,
                      snr_range: np.ndarray,
                      block_size: int = 100,
                      n_frames: int = 1000,
                      min_errors: int = 100,
                      max_frames: int = 100000,
                      callback: Optional[Callable] = None) -> List[SimulationResult]:
        """
        运行性能仿真
        
        Args:
            encode_func: 编码函数 bits -> codeword
            decode_func: 译码函数 received -> (decoded, stats)
            modulate_func: 调制函数 bits -> symbols
            demodulate_func: 软解调函数 received -> soft_values
            channel_func: 信道函数 (symbols, snr) -> received
            snr_range: SNR 范围 (dB)
            block_size: 每帧信息比特数
            n_frames: 每个 SNR 点的帧数
            min_errors: 最小错误数（统计置信度）
            max_frames: 最大帧数
            callback: 进度回调函数 (snr_idx, frame_idx, current_result)
            
        Returns:
            results: 每个 SNR 点的仿真结果
        """
        self.results = []
        self._stop_flag = False
        
        for snr_idx, snr_db in enumerate(snr_range):
            if self._stop_flag:
                break
                
            ber_calc = BERCalculator()
            bler_calc = BLERCalculator()
            total_iterations = 0
            iteration_count = 0
            
            frame = 0
            while frame < min(n_frames, max_frames):
                if self._stop_flag:
                    break
                    
                # 生成随机信息比特
                info_bits = np.random.randint(0, 2, block_size)
                
                # 编码
                coded = encode_func(info_bits)
                
                # 调制
                symbols = modulate_func(coded)
                
                # 信道
                received = channel_func(symbols, snr_db)
                
                # 软解调
                soft = demodulate_func(received, snr_db)
                
                # 译码
                result = decode_func(soft)
                if isinstance(result, tuple):
                    decoded, stats = result
                    if isinstance(stats, list) and len(stats) > 0:
                        total_iterations += len(stats)
                        iteration_count += 1
                    elif isinstance(stats, dict) and 'iterations' in stats:
                        total_iterations += stats['iterations']
                        iteration_count += 1
                else:
                    decoded = result
                    
                # 更新统计
                ber_calc.update(info_bits, decoded)
                bler_calc.update(info_bits, decoded)
                
                frame += 1
                
                # 回调
                if callback and frame % 10 == 0:
                    callback(snr_idx, frame, {
                        'snr_db': snr_db,
                        'ber': ber_calc.get_ber(),
                        'bler': bler_calc.get_bler()
                    })
                    
                # 提前停止条件
                if ber_calc.error_bits >= min_errors and frame >= 100:
                    break
                    
            # 保存结果
            avg_iter = total_iterations / iteration_count if iteration_count > 0 else 0
            
            result = SimulationResult(
                snr_db=snr_db,
                n_bits=ber_calc.total_bits,
                n_errors=ber_calc.error_bits,
                n_blocks=bler_calc.total_blocks,
                n_block_errors=bler_calc.error_blocks,
                ber=ber_calc.get_ber(),
                bler=bler_calc.get_bler(),
                avg_iterations=avg_iter
            )
            self.results.append(result)
            
        return self.results
    
    def get_results_dict(self) -> Dict:
        """获取结果字典（便于绘图）"""
        if not self.results:
            return {}
            
        return {
            'snr_db': [r.snr_db for r in self.results],
            'ber': [r.ber for r in self.results],
            'bler': [r.bler for r in self.results],
            'avg_iterations': [r.avg_iterations for r in self.results]
        }
