"""
Main Window
主窗口

信道编码性能分析与译码可视化软件的主界面
"""

import sys
import numpy as np
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QSplitter, QPushButton, QProgressBar, QStatusBar, QTabWidget,
    QMessageBox, QFileDialog, QLabel
)
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QFont, QIcon, QPalette, QColor

import matplotlib
matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure

from .widgets import ParameterPanel, PlotCanvas, SimulationThread, ResultsPanel


# 设置 matplotlib 中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


class MainWindow(QMainWindow):
    """
    主窗口
    """
    
    def __init__(self):
        super().__init__()
        self.sim_thread = None
        self.results = {}
        self.histories = []
        self.init_ui()
        
    def init_ui(self):
        """初始化界面"""
        self.setWindowTitle("信道编码性能分析与译码可视化")
        self.setMinimumSize(1400, 900)
        
        # 设置深色主题
        self.set_dark_theme()
        
        # 中央控件
        central = QWidget()
        self.setCentralWidget(central)
        main_layout = QHBoxLayout(central)
        main_layout.setSpacing(10)
        main_layout.setContentsMargins(10, 10, 10, 10)
        
        # 左侧面板
        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)
        left_panel.setMaximumWidth(320)
        left_panel.setMinimumWidth(280)
        
        # 标题
        title = QLabel("信道编码分析器")
        title.setFont(QFont("Microsoft YaHei", 16, QFont.Bold))
        title.setAlignment(Qt.AlignCenter)
        title.setStyleSheet("color: #4fc3f7; margin: 10px 0;")
        left_layout.addWidget(title)
        
        # 参数面板
        self.param_panel = ParameterPanel()
        left_layout.addWidget(self.param_panel)
        
        # 控制按钮
        btn_layout = QHBoxLayout()
        
        self.start_btn = QPushButton("▶ 开始仿真")
        self.start_btn.setStyleSheet("""
            QPushButton {
                background-color: #4caf50;
                color: white;
                border: none;
                padding: 10px 20px;
                font-size: 14px;
                border-radius: 5px;
            }
            QPushButton:hover {
                background-color: #66bb6a;
            }
            QPushButton:disabled {
                background-color: #666;
            }
        """)
        self.start_btn.clicked.connect(self.start_simulation)
        btn_layout.addWidget(self.start_btn)
        
        self.stop_btn = QPushButton("■ 停止")
        self.stop_btn.setStyleSheet("""
            QPushButton {
                background-color: #f44336;
                color: white;
                border: none;
                padding: 10px 20px;
                font-size: 14px;
                border-radius: 5px;
            }
            QPushButton:hover {
                background-color: #ef5350;
            }
            QPushButton:disabled {
                background-color: #666;
            }
        """)
        self.stop_btn.setEnabled(False)
        self.stop_btn.clicked.connect(self.stop_simulation)
        btn_layout.addWidget(self.stop_btn)
        
        left_layout.addLayout(btn_layout)
        
        # 进度条
        self.progress_bar = QProgressBar()
        self.progress_bar.setStyleSheet("""
            QProgressBar {
                border: 1px solid #555;
                border-radius: 5px;
                text-align: center;
                background-color: #2d2d2d;
            }
            QProgressBar::chunk {
                background-color: #4fc3f7;
            }
        """)
        left_layout.addWidget(self.progress_bar)
        
        # 结果面板
        self.results_panel = ResultsPanel()
        left_layout.addWidget(self.results_panel)
        
        # 导出按钮
        export_btn = QPushButton("📊 导出结果")
        export_btn.setStyleSheet("""
            QPushButton {
                background-color: #ff9800;
                color: white;
                border: none;
                padding: 8px 16px;
                font-size: 12px;
                border-radius: 5px;
            }
            QPushButton:hover {
                background-color: #ffa726;
            }
        """)
        export_btn.clicked.connect(self.export_results)
        left_layout.addWidget(export_btn)
        
        main_layout.addWidget(left_panel)
        
        # 右侧绘图区
        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)
        
        # 绘图标签页
        self.plot_tabs = QTabWidget()
        self.plot_tabs.setStyleSheet("""
            QTabWidget::pane {
                border: 1px solid #444;
                background-color: #1e1e1e;
            }
            QTabBar::tab {
                background-color: #2d2d2d;
                color: #ccc;
                padding: 8px 16px;
                margin-right: 2px;
            }
            QTabBar::tab:selected {
                background-color: #3d3d3d;
                color: #4fc3f7;
            }
        """)
        
        # BER/BLER 曲线
        self.ber_widget = QWidget()
        ber_layout = QVBoxLayout(self.ber_widget)
        self.ber_fig = Figure(figsize=(10, 6), dpi=100, facecolor='#1e1e1e')
        self.ber_canvas = FigureCanvas(self.ber_fig)
        self.ber_toolbar = NavigationToolbar(self.ber_canvas, self)
        ber_layout.addWidget(self.ber_toolbar)
        ber_layout.addWidget(self.ber_canvas)
        self.plot_tabs.addTab(self.ber_widget, "BER/BLER 曲线")
        
        # 收敛曲线
        self.conv_widget = QWidget()
        conv_layout = QVBoxLayout(self.conv_widget)
        self.conv_fig = Figure(figsize=(10, 6), dpi=100, facecolor='#1e1e1e')
        self.conv_canvas = FigureCanvas(self.conv_fig)
        self.conv_toolbar = NavigationToolbar(self.conv_canvas, self)
        conv_layout.addWidget(self.conv_toolbar)
        conv_layout.addWidget(self.conv_canvas)
        self.plot_tabs.addTab(self.conv_widget, "收敛曲线")
        
        # LLR 可视化
        self.llr_widget = QWidget()
        llr_layout = QVBoxLayout(self.llr_widget)
        self.llr_fig = Figure(figsize=(10, 6), dpi=100, facecolor='#1e1e1e')
        self.llr_canvas = FigureCanvas(self.llr_fig)
        self.llr_toolbar = NavigationToolbar(self.llr_canvas, self)
        llr_layout.addWidget(self.llr_toolbar)
        llr_layout.addWidget(self.llr_canvas)
        self.plot_tabs.addTab(self.llr_widget, "软信息可视化")
        
        # 迭代统计
        self.iter_widget = QWidget()
        iter_layout = QVBoxLayout(self.iter_widget)
        self.iter_fig = Figure(figsize=(10, 6), dpi=100, facecolor='#1e1e1e')
        self.iter_canvas = FigureCanvas(self.iter_fig)
        self.iter_toolbar = NavigationToolbar(self.iter_canvas, self)
        iter_layout.addWidget(self.iter_toolbar)
        iter_layout.addWidget(self.iter_canvas)
        self.plot_tabs.addTab(self.iter_widget, "迭代统计")
        
        right_layout.addWidget(self.plot_tabs)
        main_layout.addWidget(right_panel, stretch=1)
        
        # 状态栏
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        self.status_bar.showMessage("就绪")
        
    def set_dark_theme(self):
        """设置深色主题"""
        palette = QPalette()
        palette.setColor(QPalette.Window, QColor(30, 30, 30))
        palette.setColor(QPalette.WindowText, QColor(200, 200, 200))
        palette.setColor(QPalette.Base, QColor(45, 45, 45))
        palette.setColor(QPalette.AlternateBase, QColor(53, 53, 53))
        palette.setColor(QPalette.ToolTipBase, QColor(255, 255, 255))
        palette.setColor(QPalette.ToolTipText, QColor(0, 0, 0))
        palette.setColor(QPalette.Text, QColor(200, 200, 200))
        palette.setColor(QPalette.Button, QColor(53, 53, 53))
        palette.setColor(QPalette.ButtonText, QColor(200, 200, 200))
        palette.setColor(QPalette.BrightText, QColor(255, 0, 0))
        palette.setColor(QPalette.Link, QColor(79, 195, 247))
        palette.setColor(QPalette.Highlight, QColor(79, 195, 247))
        palette.setColor(QPalette.HighlightedText, QColor(0, 0, 0))
        self.setPalette(palette)
        
    def start_simulation(self):
        """开始仿真"""
        params = self.param_panel.get_parameters()
        
        # 重置
        self.results = {'snr_db': [], 'ber': [], 'bler': [], 'avg_iterations': []}
        self.histories = []
        self.clear_plots()
        self.results_panel.clear()
        
        # 创建仿真线程
        self.sim_thread = SimulationThread(params)
        self.sim_thread.progress.connect(self.on_progress)
        self.sim_thread.snr_complete.connect(self.on_snr_complete)
        self.sim_thread.iteration_history.connect(self.on_iteration_history)
        self.sim_thread.finished_signal.connect(self.on_simulation_finished)
        self.sim_thread.error.connect(self.on_error)
        
        # 更新 UI
        self.start_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)
        self.progress_bar.setValue(0)
        self.status_bar.showMessage("仿真进行中...")
        
        self.sim_thread.start()
        
    def stop_simulation(self):
        """停止仿真"""
        if self.sim_thread:
            self.sim_thread.stop()
            self.status_bar.showMessage("正在停止...")
            
    def on_progress(self, value: int, message: str):
        """进度更新"""
        self.progress_bar.setValue(value)
        self.status_bar.showMessage(message)
        
    def on_snr_complete(self, snr: float, ber: float, bler: float):
        """单个 SNR 点完成"""
        self.results['snr_db'].append(snr)
        self.results['ber'].append(ber)
        self.results['bler'].append(bler)
        
        # 实时更新 BER 曲线
        self.update_ber_plot()
        
    def on_iteration_history(self, histories: list):
        """收到迭代历史"""
        self.histories = histories
        self.update_convergence_plot()
        self.update_llr_plot()
        
    def on_simulation_finished(self, results: dict):
        """仿真完成"""
        self.results = results
        self.histories = results.get('histories', [])
        
        # 更新所有图
        self.update_ber_plot()
        self.update_convergence_plot()
        self.update_llr_plot()
        self.update_iteration_plot()
        
        # 更新结果面板
        self.results_panel.update_results(results)
        
        # 更新 UI
        self.start_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        self.progress_bar.setValue(100)
        self.status_bar.showMessage("仿真完成")
        
    def on_error(self, message: str):
        """错误处理"""
        QMessageBox.critical(self, "错误", message)
        self.start_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        self.status_bar.showMessage("仿真出错")
        
    def clear_plots(self):
        """清除所有图"""
        for fig in [self.ber_fig, self.conv_fig, self.llr_fig, self.iter_fig]:
            fig.clear()
            
    def update_ber_plot(self):
        """更新 BER/BLER 曲线"""
        self.ber_fig.clear()
        
        if not self.results.get('snr_db'):
            self.ber_canvas.draw()
            return
            
        ax1 = self.ber_fig.add_subplot(121)
        ax2 = self.ber_fig.add_subplot(122)
        
        snr = self.results['snr_db']
        ber = self.results['ber']
        bler = self.results.get('bler', [])
        
        # 设置深色背景
        for ax in [ax1, ax2]:
            ax.set_facecolor('#2d2d2d')
            ax.tick_params(colors='white')
            ax.spines['bottom'].set_color('white')
            ax.spines['top'].set_color('white')
            ax.spines['left'].set_color('white')
            ax.spines['right'].set_color('white')
            
        # BER 曲线
        # 过滤零值
        valid_ber = [(s, b) for s, b in zip(snr, ber) if b > 0]
        if valid_ber:
            snr_v, ber_v = zip(*valid_ber)
            ax1.semilogy(snr_v, ber_v, 'o-', color='#4fc3f7', linewidth=2, 
                        markersize=6, label='BER')
        ax1.set_xlabel('Eb/N0 (dB)', color='white')
        ax1.set_ylabel('BER', color='white')
        ax1.set_title('误比特率 (BER)', color='white', fontweight='bold')
        ax1.grid(True, alpha=0.3, color='gray')
        ax1.legend(facecolor='#2d2d2d', edgecolor='white', labelcolor='white')
        
        # BLER 曲线
        if bler:
            valid_bler = [(s, b) for s, b in zip(snr, bler) if b > 0]
            if valid_bler:
                snr_v, bler_v = zip(*valid_bler)
                ax2.semilogy(snr_v, bler_v, 's-', color='#ff7043', linewidth=2,
                            markersize=6, label='BLER')
        ax2.set_xlabel('Eb/N0 (dB)', color='white')
        ax2.set_ylabel('BLER', color='white')
        ax2.set_title('误块率 (BLER)', color='white', fontweight='bold')
        ax2.grid(True, alpha=0.3, color='gray')
        ax2.legend(facecolor='#2d2d2d', edgecolor='white', labelcolor='white')
        
        self.ber_fig.tight_layout()
        self.ber_canvas.draw()
        
    def update_convergence_plot(self):
        """更新收敛曲线"""
        self.conv_fig.clear()
        
        if not self.histories:
            self.conv_canvas.draw()
            return
            
        ax1 = self.conv_fig.add_subplot(121)
        ax2 = self.conv_fig.add_subplot(122)
        
        for ax in [ax1, ax2]:
            ax.set_facecolor('#2d2d2d')
            ax.tick_params(colors='white')
            for spine in ax.spines.values():
                spine.set_color('white')
                
        colors = plt.cm.viridis(np.linspace(0, 1, min(10, len(self.histories))))
        
        for i, history in enumerate(self.histories[:10]):
            if not history:
                continue
            iterations = list(range(1, len(history) + 1))
            syndrome = [h.get('syndrome_weight', 0) for h in history]
            llr_mean = [h.get('llr_mean', 0) for h in history]
            
            ax1.plot(iterations, syndrome, color=colors[i], alpha=0.7, linewidth=1.5)
            ax2.plot(iterations, llr_mean, color=colors[i], alpha=0.7, linewidth=1.5)
            
        ax1.set_xlabel('迭代次数', color='white')
        ax1.set_ylabel('校验子权重', color='white')
        ax1.set_title('校验子收敛', color='white', fontweight='bold')
        ax1.grid(True, alpha=0.3, color='gray')
        ax1.set_yscale('symlog', linthresh=1)
        
        ax2.set_xlabel('迭代次数', color='white')
        ax2.set_ylabel('|LLR| 均值', color='white')
        ax2.set_title('LLR 增长', color='white', fontweight='bold')
        ax2.grid(True, alpha=0.3, color='gray')
        
        self.conv_fig.tight_layout()
        self.conv_canvas.draw()
        
    def update_llr_plot(self):
        """更新软信息可视化"""
        self.llr_fig.clear()
        
        if not self.histories or not self.histories[0]:
            self.llr_canvas.draw()
            return
            
        # 取第一帧的历史
        history = self.histories[0]
        
        ax1 = self.llr_fig.add_subplot(121)
        ax2 = self.llr_fig.add_subplot(122)
        
        for ax in [ax1, ax2]:
            ax.set_facecolor('#2d2d2d')
            ax.tick_params(colors='white')
            for spine in ax.spines.values():
                spine.set_color('white')
                
        # 收集 LLR 数据
        llr_data = []
        for h in history:
            posterior = h.get('posterior_llr', None)
            if posterior is not None:
                llr_data.append(posterior[:100])  # 只取前 100 个比特
                
        if llr_data:
            llr_matrix = np.array(llr_data)
            
            # 热图
            im = ax1.imshow(llr_matrix.T, aspect='auto', cmap='RdBu_r',
                           vmin=-np.percentile(np.abs(llr_matrix), 95),
                           vmax=np.percentile(np.abs(llr_matrix), 95))
            ax1.set_xlabel('迭代次数', color='white')
            ax1.set_ylabel('比特索引', color='white')
            ax1.set_title('LLR 演变热图', color='white', fontweight='bold')
            cbar = self.llr_fig.colorbar(im, ax=ax1)
            cbar.set_label('LLR', color='white')
            cbar.ax.yaxis.set_tick_params(color='white')
            plt.setp(plt.getp(cbar.ax.axes, 'yticklabels'), color='white')
            
            # 最终 LLR 分布
            final_llr = llr_matrix[-1]
            ax2.hist(final_llr, bins=30, density=True, alpha=0.7, 
                    color='#4fc3f7', edgecolor='white', linewidth=0.5)
            ax2.axvline(0, color='red', linestyle='--', linewidth=2)
            ax2.set_xlabel('LLR', color='white')
            ax2.set_ylabel('密度', color='white')
            ax2.set_title('最终 LLR 分布', color='white', fontweight='bold')
            ax2.grid(True, alpha=0.3, color='gray')
            
        self.llr_fig.tight_layout()
        self.llr_canvas.draw()
        
    def update_iteration_plot(self):
        """更新迭代统计图"""
        self.iter_fig.clear()
        
        if not self.histories:
            self.iter_canvas.draw()
            return
            
        ax1 = self.iter_fig.add_subplot(121)
        ax2 = self.iter_fig.add_subplot(122)
        
        for ax in [ax1, ax2]:
            ax.set_facecolor('#2d2d2d')
            ax.tick_params(colors='white')
            for spine in ax.spines.values():
                spine.set_color('white')
                
        # 迭代次数分布
        iter_counts = [len(h) for h in self.histories if h]
        if iter_counts:
            ax1.hist(iter_counts, bins=20, alpha=0.7, color='#66bb6a', 
                    edgecolor='white', linewidth=0.5)
            ax1.axvline(np.mean(iter_counts), color='red', linestyle='--', 
                       linewidth=2, label=f'均值: {np.mean(iter_counts):.1f}')
            ax1.set_xlabel('迭代次数', color='white')
            ax1.set_ylabel('频数', color='white')
            ax1.set_title('迭代次数分布', color='white', fontweight='bold')
            ax1.legend(facecolor='#2d2d2d', edgecolor='white', labelcolor='white')
            ax1.grid(True, alpha=0.3, color='gray')
            
        # 收敛率
        convergence = [h[-1].get('syndrome_weight', 1) == 0 for h in self.histories if h]
        if convergence:
            conv_rate = sum(convergence) / len(convergence) * 100
            labels = ['收敛', '未收敛']
            sizes = [conv_rate, 100 - conv_rate]
            colors = ['#4caf50', '#f44336']
            ax2.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%',
                   startangle=90, textprops={'color': 'white'})
            ax2.set_title('收敛率统计', color='white', fontweight='bold')
            
        self.iter_fig.tight_layout()
        self.iter_canvas.draw()
        
    def export_results(self):
        """导出结果"""
        if not self.results.get('snr_db'):
            QMessageBox.warning(self, "提示", "没有可导出的结果")
            return
            
        filename, _ = QFileDialog.getSaveFileName(
            self, "导出结果", "", "CSV 文件 (*.csv);;所有文件 (*)"
        )
        
        if filename:
            try:
                import csv
                with open(filename, 'w', newline='', encoding='utf-8-sig') as f:
                    writer = csv.writer(f)
                    writer.writerow(['SNR (dB)', 'BER', 'BLER', 'Avg Iterations'])
                    for snr, ber, bler, avg_iter in zip(
                        self.results['snr_db'],
                        self.results['ber'],
                        self.results['bler'],
                        self.results.get('avg_iterations', [0] * len(self.results['snr_db']))
                    ):
                        writer.writerow([snr, ber, bler, avg_iter])
                QMessageBox.information(self, "成功", f"结果已导出到 {filename}")
            except Exception as e:
                QMessageBox.critical(self, "错误", f"导出失败: {str(e)}")
                
    def closeEvent(self, event):
        """关闭事件"""
        if self.sim_thread and self.sim_thread.isRunning():
            self.sim_thread.stop()
            self.sim_thread.wait()
        event.accept()


def run_app():
    """启动应用"""
    app = QApplication(sys.argv)
    app.setStyle('Fusion')
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())


if __name__ == '__main__':
    run_app()
