"""
Convergence Plotter
收敛曲线绘制

绘制 BP 译码的收敛过程
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from typing import List, Dict, Optional


class ConvergencePlotter:
    """
    收敛曲线绘制器
    
    可视化 BP 译码的迭代收敛过程
    """
    
    def __init__(self, figsize: tuple = (10, 6)):
        """
        初始化绘制器
        
        Args:
            figsize: 图像尺寸
        """
        self.figsize = figsize
        self.fig = None
        self.axes = None
        
    def create_figure(self) -> Figure:
        """创建图形"""
        self.fig, self.axes = plt.subplots(2, 2, figsize=self.figsize)
        self.fig.suptitle('BP 译码收敛分析', fontsize=14, fontweight='bold')
        return self.fig
    
    def plot_convergence(self, history: List[Dict], 
                        clear: bool = True) -> Figure:
        """
        绘制单次译码的收敛曲线
        
        Args:
            history: BP 译码返回的迭代历史
            clear: 是否清除之前的图
            
        Returns:
            Figure 对象
        """
        if self.fig is None:
            self.create_figure()
            
        if clear:
            for ax in self.axes.flat:
                ax.clear()
                
        if not history:
            return self.fig
            
        iterations = list(range(1, len(history) + 1))
        
        # 1. 校验子权重
        syndrome_weights = [h.get('syndrome_weight', 0) for h in history]
        ax1 = self.axes[0, 0]
        ax1.plot(iterations, syndrome_weights, 'b-o', markersize=4, linewidth=1.5)
        ax1.set_xlabel('迭代次数')
        ax1.set_ylabel('校验子权重')
        ax1.set_title('校验子收敛')
        ax1.grid(True, alpha=0.3)
        ax1.set_yscale('symlog', linthresh=1)
        
        # 2. LLR 均值变化
        llr_means = [h.get('llr_mean', 0) for h in history]
        ax2 = self.axes[0, 1]
        ax2.plot(iterations, llr_means, 'g-o', markersize=4, linewidth=1.5)
        ax2.set_xlabel('迭代次数')
        ax2.set_ylabel('|LLR| 均值')
        ax2.set_title('LLR 幅度增长')
        ax2.grid(True, alpha=0.3)
        
        # 3. LLR 标准差
        llr_stds = [h.get('llr_std', 0) for h in history]
        ax3 = self.axes[1, 0]
        ax3.plot(iterations, llr_stds, 'r-o', markersize=4, linewidth=1.5)
        ax3.set_xlabel('迭代次数')
        ax3.set_ylabel('LLR 标准差')
        ax3.set_title('LLR 分布扩展')
        ax3.grid(True, alpha=0.3)
        
        # 4. 收敛指示
        ax4 = self.axes[1, 1]
        converged = syndrome_weights[-1] == 0
        final_iter = len(history)
        
        info_text = f"最终迭代次数: {final_iter}\n"
        info_text += f"收敛状态: {'✓ 成功' if converged else '✗ 未收敛'}\n"
        info_text += f"最终校验子权重: {syndrome_weights[-1]}\n"
        info_text += f"最终 LLR 均值: {llr_means[-1]:.2f}"
        
        ax4.text(0.5, 0.5, info_text, transform=ax4.transAxes,
                fontsize=12, verticalalignment='center', horizontalalignment='center',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        ax4.set_xlim(0, 1)
        ax4.set_ylim(0, 1)
        ax4.axis('off')
        ax4.set_title('译码结果摘要')
        
        self.fig.tight_layout()
        return self.fig
    
    def plot_multi_convergence(self, histories: List[List[Dict]],
                               labels: List[str] = None) -> Figure:
        """
        绘制多次译码的收敛曲线对比
        
        Args:
            histories: 多次译码的历史列表
            labels: 曲线标签
            
        Returns:
            Figure 对象
        """
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        fig.suptitle('多次译码收敛对比', fontsize=14, fontweight='bold')
        
        colors = plt.cm.viridis(np.linspace(0, 1, len(histories)))
        
        for i, history in enumerate(histories):
            if not history:
                continue
                
            iterations = list(range(1, len(history) + 1))
            syndrome = [h.get('syndrome_weight', 0) for h in history]
            llr_mean = [h.get('llr_mean', 0) for h in history]
            
            label = labels[i] if labels and i < len(labels) else f'Frame {i+1}'
            
            axes[0].plot(iterations, syndrome, color=colors[i], 
                        alpha=0.7, label=label)
            axes[1].plot(iterations, llr_mean, color=colors[i], 
                        alpha=0.7, label=label)
                        
        axes[0].set_xlabel('迭代次数')
        axes[0].set_ylabel('校验子权重')
        axes[0].set_title('校验子收敛')
        axes[0].grid(True, alpha=0.3)
        axes[0].set_yscale('symlog', linthresh=1)
        
        axes[1].set_xlabel('迭代次数')
        axes[1].set_ylabel('|LLR| 均值')
        axes[1].set_title('LLR 增长')
        axes[1].grid(True, alpha=0.3)
        
        if len(histories) <= 10:
            axes[0].legend(fontsize=8)
            axes[1].legend(fontsize=8)
            
        fig.tight_layout()
        return fig
    
    def plot_average_convergence(self, avg_data: Dict) -> Figure:
        """
        绘制平均收敛曲线
        
        Args:
            avg_data: ConvergenceAnalyzer.get_average_curves() 返回的数据
            
        Returns:
            Figure 对象
        """
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        fig.suptitle('平均收敛曲线', fontsize=14, fontweight='bold')
        
        iterations = avg_data.get('iterations', [])
        
        # 校验子
        syn_mean = avg_data.get('syndrome_mean', [])
        syn_std = avg_data.get('syndrome_std', [])
        
        if iterations and syn_mean:
            axes[0].plot(iterations, syn_mean, 'b-', linewidth=2, label='均值')
            axes[0].fill_between(iterations, 
                                np.array(syn_mean) - np.array(syn_std),
                                np.array(syn_mean) + np.array(syn_std),
                                alpha=0.3, color='blue', label='±1σ')
            axes[0].set_xlabel('迭代次数')
            axes[0].set_ylabel('校验子权重')
            axes[0].set_title('校验子收敛（平均）')
            axes[0].grid(True, alpha=0.3)
            axes[0].legend()
            axes[0].set_yscale('symlog', linthresh=1)
            
        # LLR
        llr_mean = avg_data.get('llr_mean', [])
        llr_std = avg_data.get('llr_std', [])
        
        if iterations and llr_mean:
            axes[1].plot(iterations, llr_mean, 'g-', linewidth=2, label='均值')
            axes[1].fill_between(iterations,
                                np.array(llr_mean) - np.array(llr_std),
                                np.array(llr_mean) + np.array(llr_std),
                                alpha=0.3, color='green', label='±1σ')
            axes[1].set_xlabel('迭代次数')
            axes[1].set_ylabel('|LLR| 均值')
            axes[1].set_title('LLR 增长（平均）')
            axes[1].grid(True, alpha=0.3)
            axes[1].legend()
            
        fig.tight_layout()
        return fig


class ViterbiPathPlotter:
    """
    Viterbi 路径可视化
    """
    
    def __init__(self):
        pass
    
    def plot_trellis(self, n_states: int, n_stages: int,
                    survivor_path: np.ndarray = None) -> Figure:
        """
        绘制网格图和幸存路径
        
        Args:
            n_states: 状态数
            n_stages: 时间步数
            survivor_path: 幸存路径
            
        Returns:
            Figure 对象
        """
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # 绘制状态节点
        for t in range(n_stages):
            for s in range(n_states):
                ax.plot(t, s, 'ko', markersize=8)
                
        # 绘制幸存路径
        if survivor_path is not None:
            # 回溯路径
            state = 0
            path = [state]
            for t in range(len(survivor_path) - 1, -1, -1):
                state = survivor_path[t, state]
                path.append(state)
            path = path[::-1]
            
            for t in range(len(path) - 1):
                ax.plot([t, t+1], [path[t], path[t+1]], 'r-', linewidth=2)
                
        ax.set_xlabel('时间步')
        ax.set_ylabel('状态')
        ax.set_title('Viterbi 网格图与幸存路径')
        ax.set_yticks(range(n_states))
        ax.grid(True, alpha=0.3)
        
        fig.tight_layout()
        return fig
    
    def plot_path_metrics(self, metric_history: List[Dict]) -> Figure:
        """
        绘制路径度量变化
        
        Args:
            metric_history: Viterbi 译码返回的度量历史
            
        Returns:
            Figure 对象
        """
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        fig.suptitle('Viterbi 路径度量分析', fontsize=14, fontweight='bold')
        
        if not metric_history:
            return fig
            
        times = list(range(1, len(metric_history) + 1))
        
        # 度量范围
        mins = [h['min'] for h in metric_history]
        maxs = [h['max'] for h in metric_history]
        means = [h['mean'] for h in metric_history]
        
        axes[0].fill_between(times, mins, maxs, alpha=0.3, color='blue', label='范围')
        axes[0].plot(times, means, 'b-', linewidth=2, label='均值')
        axes[0].set_xlabel('时间步')
        axes[0].set_ylabel('路径度量')
        axes[0].set_title('度量演变')
        axes[0].grid(True, alpha=0.3)
        axes[0].legend()
        
        # 度量展开
        spreads = [h['spread'] for h in metric_history]
        axes[1].plot(times, spreads, 'g-o', markersize=3, linewidth=1.5)
        axes[1].set_xlabel('时间步')
        axes[1].set_ylabel('度量展开')
        axes[1].set_title('路径分离度')
        axes[1].grid(True, alpha=0.3)
        
        fig.tight_layout()
        return fig
