"""
LDPC Encoder
LDPC 码编码器

支持规则 LDPC 码的生成和编码
"""

import numpy as np
from typing import Tuple, Optional
from scipy import sparse


class LDPCCode:
    """
    LDPC 码
    
    使用 Gallager 方法构造规则 LDPC 码
    
    Parameters:
        n: 码字长度
        k: 信息位长度
        wc: 列重（每列 1 的个数）
        wr: 行重（每行 1 的个数）
        
    Example:
        ldpc = LDPCCode(n=100, k=50)
        codeword = ldpc.encode(info_bits)
    """
    
    def __init__(self, n: int = 96, k: int = 48, 
                 wc: int = 3, wr: int = 6,
                 seed: Optional[int] = None):
        """
        初始化 LDPC 码
        
        Args:
            n: 码字长度
            k: 信息位长度  
            wc: 列重 (默认 3)
            wr: 行重 (默认 6)
            seed: 随机种子（可选）
        """
        self.n = n
        self.k = k
        self.m = n - k  # 校验位个数
        self.wc = wc
        self.wr = wr
        self.rate = k / n
        
        if seed is not None:
            np.random.seed(seed)
            
        # 生成校验矩阵 H
        self.H = self._generate_parity_check_matrix()
        
        # 生成生成矩阵 G (系统形式)
        self.G = self._generate_generator_matrix()
        
    def _generate_parity_check_matrix(self) -> np.ndarray:
        """
        使用 Gallager 方法生成规则 LDPC 校验矩阵
        
        Returns:
            H: (m x n) 校验矩阵
        """
        # 计算子矩阵尺寸
        m_sub = self.m // self.wc
        
        # 创建第一个子矩阵（循环形式）
        H_sub = np.zeros((m_sub, self.n), dtype=np.int32)
        for i in range(m_sub):
            for j in range(self.wr):
                col = i * self.wr + j
                if col < self.n:
                    H_sub[i, col] = 1
                    
        # 堆叠并随机置换列
        H_list = [H_sub]
        for _ in range(self.wc - 1):
            perm = np.random.permutation(self.n)
            H_list.append(H_sub[:, perm])
            
        H = np.vstack(H_list)
        
        # 调整大小
        if H.shape[0] > self.m:
            H = H[:self.m, :]
        elif H.shape[0] < self.m:
            # 补充行
            extra = np.zeros((self.m - H.shape[0], self.n), dtype=np.int32)
            H = np.vstack([H, extra])
            
        return H
    
    def _generate_generator_matrix(self) -> np.ndarray:
        """
        从校验矩阵生成系统形式的生成矩阵
        使用改进的高斯消元确保 H * G^T = 0
        
        Returns:
            G: (k x n) 生成矩阵
        """
        H = self.H.copy().astype(np.int32)
        m, n = H.shape
        k = n - m
        
        # 尝试将 H 转换为 [A | I_m] 形式
        # 通过列交换和高斯消元
        
        # 记录列置换
        col_perm = list(range(n))
        
        # 对每一行找主元并消元
        pivot_col = n - m  # 从右侧开始找主元位置
        
        for row in range(m):
            # 在当前行找一个非零元素作为主元
            pivot_found = False
            
            # 首先在右侧 m 列中寻找
            for j in range(pivot_col + row, n):
                if H[row, j] == 1:
                    # 交换列
                    if j != pivot_col + row:
                        H[:, [pivot_col + row, j]] = H[:, [j, pivot_col + row]]
                        col_perm[pivot_col + row], col_perm[j] = col_perm[j], col_perm[pivot_col + row]
                    pivot_found = True
                    break
                    
            # 如果右侧没找到，在左侧找
            if not pivot_found:
                for j in range(pivot_col + row):
                    if H[row, j] == 1:
                        H[:, [pivot_col + row, j]] = H[:, [j, pivot_col + row]]
                        col_perm[pivot_col + row], col_perm[j] = col_perm[j], col_perm[pivot_col + row]
                        pivot_found = True
                        break
                        
            if not pivot_found:
                continue
                
            # 消元：将该列其他行的 1 消去
            for r in range(m):
                if r != row and H[r, pivot_col + row] == 1:
                    H[r, :] = (H[r, :] + H[row, :]) % 2
                    
        # 现在 H 应该是 [A | I_m] 形式（经过列置换后）
        # 提取 A
        A = H[:, :k]
        
        # G = [I_k | A^T] 在置换前的列顺序
        I_k = np.eye(k, dtype=np.int32)
        G_perm = np.hstack([I_k, A.T])
        
        # 应用逆置换恢复原始列顺序
        inv_perm = [0] * n
        for i, p in enumerate(col_perm):
            inv_perm[p] = i
            
        G = G_perm[:, inv_perm]
        
        # 存储列顺序
        self._col_order = col_perm
        
        return G.astype(np.int32)
    
    def encode(self, bits: np.ndarray) -> np.ndarray:
        """
        编码信息比特
        
        Args:
            bits: 长度为 k 的信息比特数组
            
        Returns:
            长度为 n 的码字
        """
        bits = np.asarray(bits, dtype=np.int32)
        
        if len(bits) != self.k:
            raise ValueError(f"输入长度必须为 {self.k}，实际为 {len(bits)}")
            
        # c = m * G mod 2
        codeword = np.dot(bits, self.G) % 2
        
        return codeword.astype(np.int32)
    
    def check(self, codeword: np.ndarray) -> bool:
        """
        检查码字是否有效
        
        Args:
            codeword: 长度为 n 的码字
            
        Returns:
            True 如果是有效码字
        """
        syndrome = np.dot(self.H, codeword) % 2
        return np.all(syndrome == 0)
    
    def get_neighbors(self) -> Tuple[list, list]:
        """
        获取变量节点和校验节点的邻居信息（用于 BP 译码）
        
        Returns:
            var_neighbors: 每个变量节点连接的校验节点列表
            check_neighbors: 每个校验节点连接的变量节点列表
        """
        var_neighbors = [[] for _ in range(self.n)]
        check_neighbors = [[] for _ in range(self.m)]
        
        for i in range(self.m):
            for j in range(self.n):
                if self.H[i, j] == 1:
                    var_neighbors[j].append(i)
                    check_neighbors[i].append(j)
                    
        return var_neighbors, check_neighbors
    
    def get_info(self) -> dict:
        """获取 LDPC 码参数信息"""
        return {
            'n': self.n,
            'k': self.k,
            'm': self.m,
            'rate': self.rate,
            'wc': self.wc,
            'wr': self.wr,
            'H_density': np.sum(self.H) / (self.m * self.n)
        }
    
    def __repr__(self):
        return f"LDPCCode(n={self.n}, k={self.k}, rate={self.rate:.3f})"
