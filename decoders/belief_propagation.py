"""
Belief Propagation Decoder for LDPC
LDPC 置信传播译码器

实现 Sum-Product 算法，记录迭代过程用于可视化
"""

import numpy as np
from typing import Tuple, List, Dict, Optional


class BeliefPropagationDecoder:
    """
    LDPC 置信传播 (BP) 译码器
    
    使用 Sum-Product 算法（log 域实现以提高数值稳定性）
    
    Parameters:
        H: 校验矩阵
        max_iter: 最大迭代次数
        
    Example:
        decoder = BeliefPropagationDecoder(ldpc.H, max_iter=50)
        decoded, history = decoder.decode(llr)
    """
    
    def __init__(self, H: np.ndarray, max_iter: int = 50):
        """
        初始化 BP 译码器
        
        Args:
            H: (m x n) 校验矩阵
            max_iter: 最大迭代次数
        """
        self.H = np.asarray(H, dtype=np.int32)
        self.m, self.n = self.H.shape
        self.max_iter = max_iter
        
        # 构建邻居关系
        self._build_neighbors()
        
    def _build_neighbors(self):
        """构建变量节点和校验节点的邻居索引"""
        self.var_neighbors = [[] for _ in range(self.n)]  # 每个变量节点连接的校验节点
        self.check_neighbors = [[] for _ in range(self.m)]  # 每个校验节点连接的变量节点
        
        for i in range(self.m):
            for j in range(self.n):
                if self.H[i, j] == 1:
                    self.var_neighbors[j].append(i)
                    self.check_neighbors[i].append(j)
                    
    def decode(self, llr: np.ndarray, 
               early_stop: bool = True,
               record_history: bool = True) -> Tuple[np.ndarray, List[Dict]]:
        """
        BP 译码
        
        Args:
            llr: 信道 LLR (Log-Likelihood Ratio) 值
                 LLR = log(P(x=0|y) / P(x=1|y))
                 正值表示更可能是 0，负值表示更可能是 1
            early_stop: 当校验通过时提前停止
            record_history: 是否记录迭代历史
            
        Returns:
            decoded: 译码后的硬判决比特
            history: 每次迭代的统计信息
        """
        llr = np.asarray(llr, dtype=np.float64)
        
        if len(llr) != self.n:
            raise ValueError(f"LLR 长度必须为 {self.n}，实际为 {len(llr)}")
            
        # 初始化消息
        # var_to_check[i][j]: 变量节点 j 发送给校验节点 i 的消息
        # check_to_var[i][j]: 校验节点 i 发送给变量节点 j 的消息
        var_to_check = {}
        check_to_var = {}
        
        for i in range(self.m):
            for j in self.check_neighbors[i]:
                var_to_check[(i, j)] = llr[j]
                check_to_var[(i, j)] = 0.0
                
        history = []
        
        for iteration in range(self.max_iter):
            iter_stats = {
                'iteration': iteration + 1,
                'llr_mean': [],
                'llr_std': [],
                'error_bits': 0,
                'syndrome_weight': 0
            }
            
            # 校验节点更新 (Check Node Update)
            for i in range(self.m):
                neighbors = self.check_neighbors[i]
                
                for j in neighbors:
                    # 计算除 j 外所有邻居的消息乘积（使用 tanh 规则）
                    product = 1.0
                    for k in neighbors:
                        if k != j:
                            # tanh(x/2) 用于数值稳定性
                            msg = np.clip(var_to_check[(i, k)], -50.0, 50.0)
                            product *= np.tanh(msg / 2)
                            
                    # 限制范围避免数值问题
                    product = np.clip(product, -1 + 1e-12, 1 - 1e-12)
                    check_to_var[(i, j)] = 2 * np.arctanh(product)
                    
            # 变量节点更新 (Variable Node Update)
            posterior_llr = np.zeros(self.n)
            
            for j in range(self.n):
                neighbors = self.var_neighbors[j]
                
                # 计算后验 LLR
                total = llr[j]
                for i in neighbors:
                    total += check_to_var[(i, j)]
                posterior_llr[j] = total
                
                # 更新发送给校验节点的消息
                for i in neighbors:
                    var_to_check[(i, j)] = total - check_to_var[(i, j)]
                    
            # 硬判决
            decoded = (posterior_llr < 0).astype(np.int32)
            
            # 计算校验子
            syndrome = np.dot(self.H, decoded) % 2
            syndrome_weight = np.sum(syndrome)
            
            if record_history:
                iter_stats['llr_mean'] = float(np.mean(np.abs(posterior_llr)))
                iter_stats['llr_std'] = float(np.std(posterior_llr))
                iter_stats['llr_histogram'] = np.histogram(posterior_llr, bins=20)
                iter_stats['syndrome_weight'] = int(syndrome_weight)
                iter_stats['posterior_llr'] = posterior_llr.copy()
                history.append(iter_stats)
                
            # 早停
            if early_stop and syndrome_weight == 0:
                break
                
        return decoded, history
    
    def decode_simple(self, llr: np.ndarray) -> np.ndarray:
        """
        简化的 BP 译码（不记录历史）
        
        Args:
            llr: 信道 LLR 值
            
        Returns:
            decoded: 译码后的比特
        """
        decoded, _ = self.decode(llr, early_stop=True, record_history=False)
        return decoded
    
    def compute_mutual_information(self, llr: np.ndarray) -> float:
        """
        计算 LLR 的互信息（用于 EXIT 图）
        
        Args:
            llr: LLR 值数组
            
        Returns:
            mutual_information: 互信息值 [0, 1]
        """
        # 使用高斯近似计算互信息
        # I ≈ 1 - E[log2(1 + exp(-LLR))]
        llr = np.asarray(llr)
        
        # 避免数值溢出
        llr_clipped = np.clip(llr, -20, 20)
        
        # 对于发送 0 的情况
        I = 1 - np.mean(np.log2(1 + np.exp(-np.abs(llr_clipped))))
        
        return float(np.clip(I, 0, 1))
    
    def get_iteration_stats(self, history: List[Dict]) -> Dict:
        """
        汇总迭代统计信息
        
        Args:
            history: decode 返回的历史信息
            
        Returns:
            汇总统计
        """
        if not history:
            return {}
            
        return {
            'total_iterations': len(history),
            'converged': history[-1]['syndrome_weight'] == 0,
            'llr_evolution': [h['llr_mean'] for h in history],
            'syndrome_evolution': [h['syndrome_weight'] for h in history]
        }
    
    def __repr__(self):
        return f"BeliefPropagationDecoder(m={self.m}, n={self.n}, max_iter={self.max_iter})"
