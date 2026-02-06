#!/usr/bin/env python3
"""
信道编码性能分析与译码可视化软件
Channel Coding Performance Analysis and Decoding Visualization

支持卷积码/LDPC 编码，Viterbi/BP 译码
可视化译码迭代过程，分析 BER/BLER 性能

Usage:
    python main.py
"""

import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from gui.main_window import run_app


def main():
    """Main entry point."""
    print("=" * 60)
    print("信道编码性能分析与译码可视化软件")
    print("Channel Coding Performance Analyzer")
    print("=" * 60)
    print()
    print("功能特性:")
    print("  • 卷积码编码 + Viterbi 译码")
    print("  • LDPC 编码 + 置信传播 (BP) 译码")
    print("  • BER/BLER 性能曲线")
    print("  • 译码收敛可视化")
    print("  • 软信息 (LLR) 演变分析")
    print()
    print("启动图形界面...")
    print()
    
    run_app()


if __name__ == '__main__':
    main()
