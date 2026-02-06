"""
Soft Information Visualizer
软信息可视化

显示 LLR 分布变化和迭代过程
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.animation import FuncAnimation
from typing import List, Dict, Optional, Tuple


class SoftInfoVisualizer:
    """
    软信息可视化器
    
    展示 BP 译码过程中 LLR 的变化
    """
    
    def __init__(self):
        self.fig = None
        
    def plot_llr_histogram(self, llr: np.ndarray, 
                          title: str = "LLR 分布",
                          bins: int = 50) -> Figure:
        """
        绘制 LLR 直方图
        
        Args:
            llr: LLR 数组
            title: 图标题
            bins: 直方图 bin 数
            
        Returns:
            Figure 对象
        """
        fig, ax = plt.subplots(figsize=(8, 5))
        
        ax.hist(llr, bins=bins, density=True, alpha=0.7, color='steelblue', 
                edgecolor='white', linewidth=0.5)
        
        # 添加统计信息
        mean_llr = np.mean(llr)
        std_llr = np.std(llr)
        
        ax.axvline(mean_llr, color='red', linestyle='--', linewidth=2, 
                  label=f'均值: {mean_llr:.2f}')
        ax.axvline(0, color='black', linestyle='-', linewidth=1, alpha=0.5)
        
        ax.set_xlabel('LLR 值')
        ax.set_ylabel('密度')
        ax.set_title(title)
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 添加文本框
        textstr = f'μ = {mean_llr:.2f}\nσ = {std_llr:.2f}'
        props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
        ax.text(0.95, 0.95, textstr, transform=ax.transAxes, fontsize=10,
               verticalalignment='top', horizontalalignment='right', bbox=props)
        
        fig.tight_layout()
        return fig
    
    def plot_llr_evolution(self, history: List[Dict], 
                          positions: List[int] = None) -> Figure:
        """
        绘制特定位置的 LLR 迭代演变
        
        Args:
            history: BP 译码历史
            positions: 要追踪的比特位置（默认随机选择）
            
        Returns:
            Figure 对象
        """
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        fig.suptitle('LLR 迭代演变', fontsize=14, fontweight='bold')
        
        if not history:
            return fig
            
        # 获取所有 LLR
        all_llr = []
        for h in history:
            posterior = h.get('posterior_llr', None)
            if posterior is not None:
                all_llr.append(posterior)
                
        if not all_llr:
            return fig
            
        all_llr = np.array(all_llr)
        n_iter, n_bits = all_llr.shape
        
        # 选择位置
        if positions is None:
            n_show = min(10, n_bits)
            positions = np.random.choice(n_bits, n_show, replace=False)
            
        iterations = np.arange(1, n_iter + 1)
        colors = plt.cm.tab10(np.linspace(0, 1, len(positions)))
        
        # 绘制 LLR 轨迹
        for i, pos in enumerate(positions):
            axes[0].plot(iterations, all_llr[:, pos], 
                        color=colors[i], linewidth=1.5, alpha=0.7,
                        label=f'Bit {pos}')
                        
        axes[0].axhline(0, color='black', linestyle='--', alpha=0.5)
        axes[0].set_xlabel('迭代次数')
        axes[0].set_ylabel('LLR')
        axes[0].set_title('比特 LLR 轨迹')
        axes[0].grid(True, alpha=0.3)
        if len(positions) <= 10:
            axes[0].legend(fontsize=8, loc='upper left')
            
        # 绘制 LLR 分布随迭代变化
        iter_to_show = [0, n_iter//4, n_iter//2, n_iter-1]
        iter_to_show = [i for i in iter_to_show if i < n_iter]
        
        for i in iter_to_show:
            axes[1].hist(all_llr[i], bins=30, density=True, alpha=0.5,
                        label=f'Iter {i+1}')
                        
        axes[1].set_xlabel('LLR')
        axes[1].set_ylabel('密度')
        axes[1].set_title('LLR 分布演变')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        fig.tight_layout()
        return fig
    
    def plot_llr_heatmap(self, history: List[Dict], 
                        max_bits: int = 100) -> Figure:
        """
        绘制 LLR 热图
        
        Args:
            history: BP 译码历史
            max_bits: 最大显示比特数
            
        Returns:
            Figure 对象
        """
        fig, ax = plt.subplots(figsize=(12, 6))
        
        if not history:
            return fig
            
        # 收集 LLR 数据
        llr_data = []
        for h in history:
            posterior = h.get('posterior_llr', None)
            if posterior is not None:
                llr_data.append(posterior[:max_bits])
                
        if not llr_data:
            return fig
            
        llr_matrix = np.array(llr_data)
        
        # 绘制热图
        im = ax.imshow(llr_matrix.T, aspect='auto', cmap='RdBu_r',
                      vmin=-np.percentile(np.abs(llr_matrix), 95),
                      vmax=np.percentile(np.abs(llr_matrix), 95))
                      
        ax.set_xlabel('迭代次数')
        ax.set_ylabel('比特索引')
        ax.set_title('LLR 演变热图')
        
        cbar = fig.colorbar(im, ax=ax)
        cbar.set_label('LLR 值')
        
        fig.tight_layout()
        return fig
    
    def plot_reliability_evolution(self, history: List[Dict]) -> Figure:
        """
        绘制可靠性（|LLR|）演变
        
        Args:
            history: BP 译码历史
            
        Returns:
            Figure 对象
        """
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        fig.suptitle('可靠性分析', fontsize=14, fontweight='bold')
        
        if not history:
            return fig
            
        # 收集数据
        llr_data = []
        for h in history:
            posterior = h.get('posterior_llr', None)
            if posterior is not None:
                llr_data.append(posterior)
                
        if not llr_data:
            return fig
            
        llr_matrix = np.array(llr_data)
        reliability = np.abs(llr_matrix)
        
        iterations = np.arange(1, len(llr_data) + 1)
        
        # 1. 平均可靠性
        mean_reliability = np.mean(reliability, axis=1)
        axes[0, 0].plot(iterations, mean_reliability, 'b-o', markersize=4)
        axes[0, 0].set_xlabel('迭代次数')
        axes[0, 0].set_ylabel('平均 |LLR|')
        axes[0, 0].set_title('平均可靠性')
        axes[0, 0].grid(True, alpha=0.3)
        
        # 2. 可靠性百分位数
        percentiles = [10, 50, 90]
        for p in percentiles:
            vals = np.percentile(reliability, p, axis=1)
            axes[0, 1].plot(iterations, vals, '-o', markersize=3, 
                           label=f'{p}th percentile')
        axes[0, 1].set_xlabel('迭代次数')
        axes[0, 1].set_ylabel('|LLR|')
        axes[0, 1].set_title('可靠性分布')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # 3. 低可靠性比特数
        thresholds = [0.5, 1.0, 2.0]
        for thresh in thresholds:
            low_rel_count = np.sum(reliability < thresh, axis=1)
            axes[1, 0].plot(iterations, low_rel_count, '-o', markersize=3,
                           label=f'|LLR| < {thresh}')
        axes[1, 0].set_xlabel('迭代次数')
        axes[1, 0].set_ylabel('比特数')
        axes[1, 0].set_title('低可靠性比特')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # 4. 最终 LLR 分布
        final_llr = llr_matrix[-1]
        axes[1, 1].hist(final_llr, bins=40, density=True, 
                       alpha=0.7, color='steelblue', edgecolor='white')
        axes[1, 1].axvline(0, color='red', linestyle='--', linewidth=2)
        axes[1, 1].set_xlabel('LLR')
        axes[1, 1].set_ylabel('密度')
        axes[1, 1].set_title('最终 LLR 分布')
        axes[1, 1].grid(True, alpha=0.3)
        
        fig.tight_layout()
        return fig
    
    def create_animation_data(self, history: List[Dict], 
                             max_bits: int = 100) -> Tuple[List[np.ndarray], List[int]]:
        """
        创建动画数据
        
        Args:
            history: BP 译码历史
            max_bits: 最大比特数
            
        Returns:
            frames: 每帧的 LLR 数据
            iterations: 迭代索引
        """
        frames = []
        iterations = []
        
        for i, h in enumerate(history):
            posterior = h.get('posterior_llr', None)
            if posterior is not None:
                frames.append(posterior[:max_bits])
                iterations.append(i + 1)
                
        return frames, iterations
