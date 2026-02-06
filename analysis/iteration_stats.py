"""
Iteration Statistics
迭代统计分析

分析 LDPC BP 译码的迭代行为
"""

import numpy as np
from typing import List, Dict, Tuple
from collections import defaultdict


class IterationStatistics:
    """
    迭代统计分析器
    
    收集和分析 BP 译码的迭代次数分布
    """
    
    def __init__(self):
        self.reset()
        
    def reset(self):
        """重置统计"""
        self.iteration_counts = []
        self.convergence_flags = []
        self.snr_values = []
        
    def record(self, n_iterations: int, converged: bool, snr_db: float = None):
        """
        记录一次译码结果
        
        Args:
            n_iterations: 迭代次数
            converged: 是否收敛
            snr_db: SNR 值（可选）
        """
        self.iteration_counts.append(n_iterations)
        self.convergence_flags.append(converged)
        if snr_db is not None:
            self.snr_values.append(snr_db)
            
    def record_batch(self, histories: List[List[Dict]], snr_db: float = None):
        """
        批量记录译码历史
        
        Args:
            histories: BP 译码返回的历史列表
            snr_db: SNR 值
        """
        for history in histories:
            if history:
                n_iter = len(history)
                converged = history[-1].get('syndrome_weight', 1) == 0
                self.record(n_iter, converged, snr_db)
                
    def get_statistics(self) -> Dict:
        """
        获取统计摘要
        
        Returns:
            统计字典
        """
        if not self.iteration_counts:
            return {}
            
        counts = np.array(self.iteration_counts)
        converged = np.array(self.convergence_flags)
        
        return {
            'total_frames': len(counts),
            'mean_iterations': float(np.mean(counts)),
            'std_iterations': float(np.std(counts)),
            'min_iterations': int(np.min(counts)),
            'max_iterations': int(np.max(counts)),
            'median_iterations': float(np.median(counts)),
            'convergence_rate': float(np.mean(converged)),
            'converged_mean_iter': float(np.mean(counts[converged])) if np.any(converged) else 0,
            'failed_mean_iter': float(np.mean(counts[~converged])) if np.any(~converged) else 0
        }
    
    def get_histogram(self, bins: int = 20) -> Tuple[np.ndarray, np.ndarray]:
        """
        获取迭代次数直方图
        
        Args:
            bins: 直方图 bin 数量
            
        Returns:
            hist: 直方图计数
            bin_edges: bin 边界
        """
        if not self.iteration_counts:
            return np.array([]), np.array([])
            
        return np.histogram(self.iteration_counts, bins=bins)
    
    def get_iteration_distribution(self) -> Dict[int, int]:
        """获取迭代次数分布"""
        dist = defaultdict(int)
        for count in self.iteration_counts:
            dist[count] += 1
        return dict(dist)
    
    def get_snr_grouped_stats(self) -> Dict[float, Dict]:
        """
        按 SNR 分组的统计
        
        Returns:
            每个 SNR 的统计字典
        """
        if not self.snr_values:
            return {}
            
        grouped = defaultdict(lambda: {'iterations': [], 'converged': []})
        
        for snr, n_iter, conv in zip(self.snr_values, 
                                      self.iteration_counts, 
                                      self.convergence_flags):
            grouped[snr]['iterations'].append(n_iter)
            grouped[snr]['converged'].append(conv)
            
        result = {}
        for snr, data in grouped.items():
            iters = np.array(data['iterations'])
            convs = np.array(data['converged'])
            result[snr] = {
                'mean_iterations': float(np.mean(iters)),
                'std_iterations': float(np.std(iters)),
                'convergence_rate': float(np.mean(convs))
            }
            
        return result


class ConvergenceAnalyzer:
    """
    收敛行为分析器
    
    分析 BP 译码的收敛曲线
    """
    
    def __init__(self):
        self.syndrome_curves = []
        self.llr_curves = []
        
    def reset(self):
        """重置"""
        self.syndrome_curves = []
        self.llr_curves = []
        
    def add_history(self, history: List[Dict]):
        """
        添加一次译码历史
        
        Args:
            history: BP 译码返回的迭代历史
        """
        if not history:
            return
            
        syndrome = [h.get('syndrome_weight', 0) for h in history]
        llr_mean = [h.get('llr_mean', 0) for h in history]
        
        self.syndrome_curves.append(syndrome)
        self.llr_curves.append(llr_mean)
        
    def get_average_curves(self, max_iter: int = 50) -> Dict:
        """
        获取平均收敛曲线
        
        Args:
            max_iter: 最大迭代次数
            
        Returns:
            平均曲线数据
        """
        if not self.syndrome_curves:
            return {}
            
        # 对齐曲线长度
        syndrome_padded = []
        llr_padded = []
        
        for syn, llr in zip(self.syndrome_curves, self.llr_curves):
            # 填充到 max_iter
            syn_pad = syn + [syn[-1]] * (max_iter - len(syn))
            llr_pad = llr + [llr[-1]] * (max_iter - len(llr))
            syndrome_padded.append(syn_pad[:max_iter])
            llr_padded.append(llr_pad[:max_iter])
            
        syndrome_arr = np.array(syndrome_padded)
        llr_arr = np.array(llr_padded)
        
        return {
            'iterations': list(range(1, max_iter + 1)),
            'syndrome_mean': np.mean(syndrome_arr, axis=0).tolist(),
            'syndrome_std': np.std(syndrome_arr, axis=0).tolist(),
            'llr_mean': np.mean(llr_arr, axis=0).tolist(),
            'llr_std': np.std(llr_arr, axis=0).tolist()
        }
