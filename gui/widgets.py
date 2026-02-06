"""
Custom Widgets
自定义控件

参数面板、绘图画布等
"""

import numpy as np
from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGroupBox, QLabel, 
    QComboBox, QSpinBox, QDoubleSpinBox, QPushButton, QProgressBar,
    QCheckBox, QGridLayout, QTabWidget, QTextEdit, QSplitter
)
from PyQt5.QtCore import Qt, QThread, pyqtSignal
# 使用 qt5agg
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure
import matplotlib.pyplot as plt


class PlotCanvas(FigureCanvas):
    """
    Matplotlib 绑定的 Qt 画布
    """
    
    def __init__(self, parent=None, width=8, height=6, dpi=100):
        self.fig = Figure(figsize=(width, height), dpi=dpi)
        self.axes = self.fig.add_subplot(111)
        super().__init__(self.fig)
        self.setParent(parent)
        
    def clear(self):
        """清除图形"""
        self.axes.clear()
        self.draw()
        
    def update_figure(self):
        """更新显示"""
        self.fig.tight_layout()
        self.draw()


class MultiPlotCanvas(FigureCanvas):
    """
    多子图画布
    """
    
    def __init__(self, parent=None, rows=2, cols=2, width=10, height=8, dpi=100):
        self.fig = Figure(figsize=(width, height), dpi=dpi)
        self.axes = self.fig.subplots(rows, cols)
        super().__init__(self.fig)
        self.setParent(parent)
        self.rows = rows
        self.cols = cols
        
    def clear_all(self):
        """清除所有子图"""
        for ax in self.axes.flat:
            ax.clear()
        self.draw()
        
    def update_figure(self):
        """更新显示"""
        self.fig.tight_layout()
        self.draw()


class ParameterPanel(QWidget):
    """
    参数配置面板
    """
    
    parameters_changed = pyqtSignal()
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.init_ui()
        
    def init_ui(self):
        """初始化界面"""
        layout = QVBoxLayout(self)
        layout.setSpacing(10)
        
        # 编码器设置
        encoder_group = QGroupBox("编码器设置")
        encoder_layout = QGridLayout(encoder_group)
        
        # 编码类型
        encoder_layout.addWidget(QLabel("编码类型:"), 0, 0)
        self.code_type = QComboBox()
        self.code_type.addItems(["卷积码", "LDPC"])
        self.code_type.currentIndexChanged.connect(self.on_code_type_changed)
        encoder_layout.addWidget(self.code_type, 0, 1)
        
        # 卷积码参数
        self.conv_params = QWidget()
        conv_layout = QGridLayout(self.conv_params)
        conv_layout.setContentsMargins(0, 0, 0, 0)
        
        conv_layout.addWidget(QLabel("约束长度 K:"), 0, 0)
        self.constraint_length = QSpinBox()
        self.constraint_length.setRange(3, 9)
        self.constraint_length.setValue(7)
        conv_layout.addWidget(self.constraint_length, 0, 1)
        
        conv_layout.addWidget(QLabel("生成多项式:"), 1, 0)
        self.generators = QComboBox()
        self.generators.addItems([
            "[171, 133] (标准)", 
            "[133, 171]",
            "[7, 5] (K=3)"
        ])
        conv_layout.addWidget(self.generators, 1, 1)
        
        encoder_layout.addWidget(self.conv_params, 1, 0, 1, 2)
        
        # LDPC 参数
        self.ldpc_params = QWidget()
        ldpc_layout = QGridLayout(self.ldpc_params)
        ldpc_layout.setContentsMargins(0, 0, 0, 0)
        
        ldpc_layout.addWidget(QLabel("码字长度 n:"), 0, 0)
        self.ldpc_n = QSpinBox()
        self.ldpc_n.setRange(32, 1024)
        self.ldpc_n.setValue(96)
        self.ldpc_n.setSingleStep(16)
        ldpc_layout.addWidget(self.ldpc_n, 0, 1)
        
        ldpc_layout.addWidget(QLabel("信息位长度 k:"), 1, 0)
        self.ldpc_k = QSpinBox()
        self.ldpc_k.setRange(16, 512)
        self.ldpc_k.setValue(48)
        self.ldpc_k.setSingleStep(8)
        ldpc_layout.addWidget(self.ldpc_k, 1, 1)
        
        ldpc_layout.addWidget(QLabel("最大迭代次数:"), 2, 0)
        self.bp_max_iter = QSpinBox()
        self.bp_max_iter.setRange(1, 100)
        self.bp_max_iter.setValue(20)
        ldpc_layout.addWidget(self.bp_max_iter, 2, 1)
        
        encoder_layout.addWidget(self.ldpc_params, 2, 0, 1, 2)
        self.ldpc_params.hide()
        
        layout.addWidget(encoder_group)
        
        # 信道设置
        channel_group = QGroupBox("信道设置")
        channel_layout = QGridLayout(channel_group)
        
        channel_layout.addWidget(QLabel("调制方式:"), 0, 0)
        self.modulation = QComboBox()
        self.modulation.addItems(["BPSK", "QPSK"])
        channel_layout.addWidget(self.modulation, 0, 1)
        
        channel_layout.addWidget(QLabel("SNR 起始 (dB):"), 1, 0)
        self.snr_start = QDoubleSpinBox()
        self.snr_start.setRange(-5, 20)
        self.snr_start.setValue(0)
        self.snr_start.setSingleStep(0.5)
        channel_layout.addWidget(self.snr_start, 1, 1)
        
        channel_layout.addWidget(QLabel("SNR 结束 (dB):"), 2, 0)
        self.snr_end = QDoubleSpinBox()
        self.snr_end.setRange(-5, 20)
        self.snr_end.setValue(8)
        self.snr_end.setSingleStep(0.5)
        channel_layout.addWidget(self.snr_end, 2, 1)
        
        channel_layout.addWidget(QLabel("SNR 步长 (dB):"), 3, 0)
        self.snr_step = QDoubleSpinBox()
        self.snr_step.setRange(0.1, 2)
        self.snr_step.setValue(0.5)
        self.snr_step.setSingleStep(0.1)
        channel_layout.addWidget(self.snr_step, 3, 1)
        
        layout.addWidget(channel_group)
        
        # 仿真设置
        sim_group = QGroupBox("仿真设置")
        sim_layout = QGridLayout(sim_group)
        
        sim_layout.addWidget(QLabel("每 SNR 帧数:"), 0, 0)
        self.n_frames = QSpinBox()
        self.n_frames.setRange(10, 100000)
        self.n_frames.setValue(500)
        self.n_frames.setSingleStep(100)
        sim_layout.addWidget(self.n_frames, 0, 1)
        
        sim_layout.addWidget(QLabel("随机种子:"), 1, 0)
        self.seed = QSpinBox()
        self.seed.setRange(0, 99999)
        self.seed.setValue(42)
        sim_layout.addWidget(self.seed, 1, 1)
        
        self.record_history = QCheckBox("记录迭代历史")
        self.record_history.setChecked(True)
        sim_layout.addWidget(self.record_history, 2, 0, 1, 2)
        
        layout.addWidget(sim_group)
        
        layout.addStretch()
        
    def on_code_type_changed(self, index):
        """编码类型改变"""
        if index == 0:  # 卷积码
            self.conv_params.show()
            self.ldpc_params.hide()
        else:  # LDPC
            self.conv_params.hide()
            self.ldpc_params.show()
        self.parameters_changed.emit()
        
    def get_parameters(self) -> dict:
        """获取所有参数"""
        gen_map = {
            0: [0o171, 0o133],
            1: [0o133, 0o171],
            2: [0o7, 0o5]
        }
        
        return {
            'code_type': 'conv' if self.code_type.currentIndex() == 0 else 'ldpc',
            # 卷积码参数
            'constraint_length': self.constraint_length.value(),
            'generators': gen_map.get(self.generators.currentIndex(), [0o171, 0o133]),
            # LDPC 参数
            'ldpc_n': self.ldpc_n.value(),
            'ldpc_k': self.ldpc_k.value(),
            'bp_max_iter': self.bp_max_iter.value(),
            # 信道参数
            'modulation': self.modulation.currentText(),
            'snr_start': self.snr_start.value(),
            'snr_end': self.snr_end.value(),
            'snr_step': self.snr_step.value(),
            # 仿真参数
            'n_frames': self.n_frames.value(),
            'seed': self.seed.value(),
            'record_history': self.record_history.isChecked()
        }


class SimulationThread(QThread):
    """
    仿真线程
    
    在后台运行仿真，避免阻塞 UI
    """
    
    progress = pyqtSignal(int, str)  # 进度百分比，状态信息
    snr_complete = pyqtSignal(float, float, float)  # snr, ber, bler
    iteration_history = pyqtSignal(list)  # 迭代历史
    finished_signal = pyqtSignal(dict)  # 完整结果
    error = pyqtSignal(str)  # 错误信息
    
    def __init__(self, params: dict, parent=None):
        super().__init__(parent)
        self.params = params
        self._stop = False
        
    def stop(self):
        """停止仿真"""
        self._stop = True
        
    def run(self):
        """运行仿真"""
        try:
            import sys
            import os
            # Ensure project root is in path
            root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            if root_dir not in sys.path:
                sys.path.insert(0, root_dir)
            
            from encoders import ConvolutionalEncoder, LDPCCode
            from decoders import ViterbiDecoder, BeliefPropagationDecoder
            from channel import AWGNChannel, BPSKModulator, QPSKModulator
            from analysis import BERCalculator, BLERCalculator
            
            params = self.params
            
            # 创建编码器和译码器
            if params['code_type'] == 'conv':
                K = params['constraint_length']
                if K == 3:
                    gens = [0o7, 0o5]
                else:
                    gens = params['generators']
                encoder = ConvolutionalEncoder(K, gens)
                decoder = ViterbiDecoder(K, gens)
                block_size = 100
            else:
                n, k = params['ldpc_n'], params['ldpc_k']
                ldpc = LDPCCode(n, k, seed=params['seed'])
                encoder = ldpc
                decoder = BeliefPropagationDecoder(ldpc.H, params['bp_max_iter'])
                block_size = k
                
            # 调制器
            if params['modulation'] == 'BPSK':
                modulator = BPSKModulator()
            else:
                modulator = QPSKModulator()
                
            # 信道
            channel = AWGNChannel()
            
            # SNR 范围
            snr_range = np.arange(params['snr_start'], 
                                 params['snr_end'] + params['snr_step'], 
                                 params['snr_step'])
            
            results = {
                'snr_db': [],
                'ber': [],
                'bler': [],
                'avg_iterations': []
            }
            
            all_histories = []
            
            total_snr = len(snr_range)
            
            for snr_idx, snr_db in enumerate(snr_range):
                if self._stop:
                    break
                    
                channel.set_snr(snr_db)
                ber_calc = BERCalculator()
                bler_calc = BLERCalculator()
                iteration_sum = 0
                iteration_count = 0
                
                n_frames = params['n_frames']
                
                for frame in range(n_frames):
                    if self._stop:
                        break
                        
                    # 生成随机比特
                    info_bits = np.random.randint(0, 2, block_size)
                    
                    # 编码
                    coded = encoder.encode(info_bits)
                    
                    # 调制
                    symbols = modulator.modulate(coded)
                    
                    # 信道
                    received = channel.transmit(symbols)
                    
                    # 软解调
                    if params['modulation'] == 'BPSK':
                        soft = modulator.soft_demodulate(received, channel.noise_variance)
                    else:
                        soft = modulator.soft_demodulate(received, channel.noise_variance)
                        
                    # 译码
                    if params['code_type'] == 'conv':
                        decoded, stats = decoder.decode(soft, mode='soft')
                    else:
                        decoded, history = decoder.decode(soft, record_history=params['record_history'])
                        if history:
                            iteration_sum += len(history)
                            iteration_count += 1
                            if params['record_history'] and frame < 10:  # 只保存前几帧
                                all_histories.append(history)
                                
                    # 更新统计
                    ber_calc.update(info_bits, decoded)
                    bler_calc.update(info_bits, decoded)
                    
                    # 进度更新
                    if frame % 50 == 0:
                        progress = int((snr_idx * n_frames + frame) / (total_snr * n_frames) * 100)
                        self.progress.emit(progress, f"SNR={snr_db:.1f}dB, Frame {frame}/{n_frames}")
                        
                # 保存结果
                ber = ber_calc.get_ber()
                bler = bler_calc.get_bler()
                avg_iter = iteration_sum / iteration_count if iteration_count > 0 else 0
                
                results['snr_db'].append(snr_db)
                results['ber'].append(ber)
                results['bler'].append(bler)
                results['avg_iterations'].append(avg_iter)
                
                self.snr_complete.emit(snr_db, ber, bler)
                
            if all_histories:
                self.iteration_history.emit(all_histories)
                
            results['histories'] = all_histories
            self.finished_signal.emit(results)
            
        except Exception as e:
            import traceback
            self.error.emit(f"仿真错误: {str(e)}\n{traceback.format_exc()}")


class ResultsPanel(QWidget):
    """
    结果显示面板
    """
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.init_ui()
        
    def init_ui(self):
        """初始化界面"""
        layout = QVBoxLayout(self)
        
        # 结果文本
        self.results_text = QTextEdit()
        self.results_text.setReadOnly(True)
        self.results_text.setMaximumHeight(200)
        layout.addWidget(QLabel("仿真结果:"))
        layout.addWidget(self.results_text)
        
    def update_results(self, results: dict):
        """更新结果显示"""
        text = "=" * 50 + "\n"
        text += "仿真结果摘要\n"
        text += "=" * 50 + "\n\n"
        
        if results.get('snr_db'):
            text += f"{'SNR (dB)':<12}{'BER':<15}{'BLER':<15}{'Avg Iter':<10}\n"
            text += "-" * 50 + "\n"
            
            for snr, ber, bler, avg_iter in zip(
                results['snr_db'], results['ber'], 
                results['bler'], results.get('avg_iterations', [0]*len(results['snr_db']))
            ):
                text += f"{snr:<12.1f}{ber:<15.2e}{bler:<15.2e}{avg_iter:<10.1f}\n"
                
        self.results_text.setText(text)
        
    def clear(self):
        """清除结果"""
        self.results_text.clear()
