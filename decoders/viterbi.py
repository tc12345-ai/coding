"""
Viterbi Decoder
Viterbi 译码器

支持硬判决和软判决，记录路径度量用于可视化
"""

import numpy as np
from typing import Tuple, List, Dict, Optional


class ViterbiDecoder:
    """
    Viterbi 译码器
    
    Parameters:
        constraint_length: 约束长度 K
        generators: 生成多项式列表（八进制表示）
        
    Example:
        decoder = ViterbiDecoder(7, [0o171, 0o133])
        decoded, stats = decoder.decode(received_symbols = soft)
    """
    
    def __init__(self, constraint_length: int = 7,
                 generators: List[int] = None):
        """
        初始化 Viterbi 译码器
        
        Args:
            constraint_length: 约束长度 K
            generators: 生成多项式列表
        """
        self.K = constraint_length
        self.generators = generators if generators else [0o171, 0o133]
        self.n_outputs = len(self.generators)
        self.n_states = 2 ** (self.K - 1)
        
        # 构建网格图
        self._build_trellis()
        
    def _build_trellis(self):
        """构建状态转移表和输出表"""
        self.next_state = np.zeros((self.n_states, 2), dtype=np.int32)
        self.prev_state = np.zeros((self.n_states, 2), dtype=np.int32)
        self.output = np.zeros((self.n_states, 2, self.n_outputs), dtype=np.int32)
        
        for state in range(self.n_states):
            for input_bit in range(2):
                # 下一状态
                next_s = (state >> 1) | (input_bit << (self.K - 2))
                self.next_state[state, input_bit] = next_s
                
                # 输出
                register = (state << 1) | input_bit
                for i, gen in enumerate(self.generators):
                    out = 0
                    temp = register & gen
                    while temp:
                        out ^= (temp & 1)
                        temp >>= 1
                    self.output[state, input_bit, i] = out
                    
        # 构建反向索引
        for state in range(self.n_states):
            for input_bit in range(2):
                next_s = self.next_state[state, input_bit]
                if self.prev_state[next_s, 0] == 0 and state != 0:
                    self.prev_state[next_s, 1] = state
                else:
                    self.prev_state[next_s, 0] = state
                    
    def decode(self, received: np.ndarray, 
               mode: str = 'soft',
               return_metrics: bool = True) -> Tuple[np.ndarray, Dict]:
        """
        Viterbi 译码
        
        Args:
            received: 接收信号
                - 软判决: 连续值（正值表示 0，负值表示 1）
                - 硬判决: 0/1 比特
            mode: 'soft' 或 'hard'
            return_metrics: 是否返回路径度量历史
            
        Returns:
            decoded: 译码后的比特
            stats: 统计信息（路径度量历史等）
        """
        n_symbols = len(received) // self.n_outputs
        
        # 初始化路径度量
        INF = 1e10
        path_metric = np.full(self.n_states, INF)
        path_metric[0] = 0
        
        # 存储历史
        survivor = np.zeros((n_symbols, self.n_states), dtype=np.int32)
        metric_history = []
        
        for t in range(n_symbols):
            # 获取当前接收符号
            rx = received[t * self.n_outputs:(t + 1) * self.n_outputs]
            
            new_metric = np.full(self.n_states, INF)
            
            for state in range(self.n_states):
                if path_metric[state] >= INF:
                    continue
                    
                for input_bit in range(2):
                    next_s = self.next_state[state, input_bit]
                    expected = self.output[state, input_bit]
                    
                    # 计算分支度量
                    if mode == 'soft':
                        # 软判决：使用接收值和期望符号的相关性
                        # expected: 0/1 -> BPSK: +1/-1
                        expected_bpsk = 1 - 2 * expected
                        branch_metric = -np.sum(rx * expected_bpsk)
                    else:
                        # 硬判决：汉明距离
                        branch_metric = np.sum(rx != expected)
                        
                    total_metric = path_metric[state] + branch_metric
                    
                    if total_metric < new_metric[next_s]:
                        new_metric[next_s] = total_metric
                        survivor[t, next_s] = state
                        
            path_metric = new_metric
            
            if return_metrics:
                # 记录归一化后的度量（便于可视化）
                valid_metrics = path_metric[path_metric < INF]
                if len(valid_metrics) > 0:
                    metric_history.append({
                        'min': float(np.min(valid_metrics)),
                        'max': float(np.max(valid_metrics)),
                        'mean': float(np.mean(valid_metrics)),
                        'spread': float(np.max(valid_metrics) - np.min(valid_metrics))
                    })
                    
        # 回溯
        # 假设终止于零状态
        state = 0
        if path_metric[0] >= INF:
            # 如果零状态不可达，选择度量最小的状态
            state = np.argmin(path_metric)
            
        decoded = np.zeros(n_symbols, dtype=np.int32)
        for t in range(n_symbols - 1, -1, -1):
            prev = survivor[t, state]
            # 确定输入比特
            if self.next_state[prev, 0] == state:
                decoded[t] = 0
            else:
                decoded[t] = 1
            state = prev
            
        # 移除尾比特
        decoded = decoded[:-(self.K - 1)] if len(decoded) > self.K - 1 else decoded
        
        stats = {
            'final_metric': float(path_metric[0]),
            'metric_history': metric_history,
            'survivor_path': survivor
        }
        
        return decoded, stats
    
    def decode_with_traceback(self, received: np.ndarray,
                               mode: str = 'soft') -> Tuple[np.ndarray, List[Dict]]:
        """
        带详细回溯信息的译码（用于可视化）
        
        Returns:
            decoded: 译码结果
            traceback_info: 每一步的详细信息
        """
        n_symbols = len(received) // self.n_outputs
        
        INF = 1e10
        path_metric = np.full(self.n_states, INF)
        path_metric[0] = 0
        
        traceback_info = []
        survivor = np.zeros((n_symbols, self.n_states), dtype=np.int32)
        
        for t in range(n_symbols):
            rx = received[t * self.n_outputs:(t + 1) * self.n_outputs]
            new_metric = np.full(self.n_states, INF)
            step_info = {'time': t, 'metrics': {}, 'transitions': []}
            
            for state in range(self.n_states):
                if path_metric[state] >= INF:
                    continue
                    
                for input_bit in range(2):
                    next_s = self.next_state[state, input_bit]
                    expected = self.output[state, input_bit]
                    
                    if mode == 'soft':
                        expected_bpsk = 1 - 2 * expected
                        branch_metric = -np.sum(rx * expected_bpsk)
                    else:
                        branch_metric = np.sum(rx != expected)
                        
                    total_metric = path_metric[state] + branch_metric
                    
                    step_info['transitions'].append({
                        'from': int(state),
                        'to': int(next_s),
                        'input': input_bit,
                        'branch_metric': float(branch_metric),
                        'total_metric': float(total_metric)
                    })
                    
                    if total_metric < new_metric[next_s]:
                        new_metric[next_s] = total_metric
                        survivor[t, next_s] = state
                        
            path_metric = new_metric
            step_info['metrics'] = {int(s): float(m) for s, m in enumerate(path_metric) if m < INF}
            traceback_info.append(step_info)
            
        # 回溯
        state = 0 if path_metric[0] < INF else int(np.argmin(path_metric))
        decoded = np.zeros(n_symbols, dtype=np.int32)
        
        for t in range(n_symbols - 1, -1, -1):
            prev = survivor[t, state]
            decoded[t] = 1 if self.next_state[prev, 1] == state else 0
            state = prev
            
        decoded = decoded[:-(self.K - 1)] if len(decoded) > self.K - 1 else decoded
        
        return decoded, traceback_info
    
    def __repr__(self):
        gens_oct = [oct(g) for g in self.generators]
        return f"ViterbiDecoder(K={self.K}, generators={gens_oct})"
