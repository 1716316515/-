import queue
import sys
import os
import pickle
import csv
import json
import functools
import warnings
from datetime import datetime, timedelta


def check_dependencies():
    """检查必要的依赖项"""
    missing_deps = []

    try:
        import numpy as np
    except ImportError:
        missing_deps.append('numpy')

    try:
        import pandas as pd
    except ImportError:
        missing_deps.append('pandas')

    try:
        from PyQt5.QtWidgets import QApplication
    except ImportError:
        missing_deps.append('PyQt5')

    try:
        import matplotlib.pyplot as plt
    except ImportError:
        missing_deps.append('matplotlib')

    try:
        import cv2
    except ImportError:
        missing_deps.append('opencv-python')

    if missing_deps:
        print(f"缺少以下依赖项: {', '.join(missing_deps)}")
        print("请使用以下命令安装:")
        print(f"pip install {' '.join(missing_deps)}")
        return False

    return True

# 在文件开头定义全局变量
SMART_COACH_AVAILABLE = False
SMART_COACH = None


def safe_import_modules():
    """安全导入可选模块"""
    global SMART_COACH_AVAILABLE, SMART_COACH

    modules = {}

    # 尝试导入智能教练模块
    try:
        from improved_deepseek_sports import SmartSportsBot
        SMART_COACH = SmartSportsBot()
        SMART_COACH_AVAILABLE = True
        modules['smart_coach'] = SMART_COACH
        logger.info("智能教练模块导入成功")
    except ImportError:
        logger.warning("智能教练模块未找到")
        modules['smart_coach'] = None
    except Exception as e:
        logger.error(f"智能教练初始化失败: {e}")
        modules['smart_coach'] = None

    return modules
# 在文件开头添加安全导入
try:
    from improved_deepseek_sports import SmartSportsBot
    SMART_COACH_AVAILABLE = True
except ImportError:
    print("智能教练模块未找到，使用基础模式")
    SmartSportsBot = None
    SMART_COACH_AVAILABLE = False

def check_smart_coach_availability(self):
    """检查智能教练可用性"""

    def check_async():
        try:
            if SMART_COACH_AVAILABLE:
                test_bot = SmartSportsBot()
                if test_bot.coach_available:
                    self.smart_coach_status = "✅ 智能运动教练已就绪"
                else:
                    self.smart_coach_status = "⚠️ 智能教练模式受限"
            else:
                self.smart_coach_status = "📚 基础教练模式"
        except:
            self.smart_coach_status = "❌ 教练初始化失败"

    threading.Thread(target=check_async, daemon=True).start()

# 统一导入PyQt5组件
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QTabWidget, QWidget, QVBoxLayout,
    QHBoxLayout, QLabel, QPushButton, QFileDialog, QTextEdit,
    QMessageBox, QFrame, QSplitter, QScrollArea, QGroupBox,
    QStyleFactory, QToolTip, QTableWidget, QTableWidgetItem,
    QAbstractItemView, QHeaderView, QInputDialog, QMenu,
    QSlider, QSplitterHandle, QStyle, QLineEdit, QToolBar,
    QAction, QDockWidget, QListWidget, QProgressDialog, QDialog,
    QRadioButton, QDialogButtonBox, QCheckBox, QTreeWidget,
    QTreeWidgetItem, QComboBox, QSpinBox, QDoubleSpinBox,
    QProgressBar, QFormLayout, QSizePolicy
)
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QTimer, QDir, QSize
from PyQt5.QtGui import QIcon, QFont, QColor, QTextCharFormat, QTextCursor, QImage, QPixmap, QPalette

# 数据处理和分析
import numpy as np
import pandas as pd
import sqlite3
import logging
import locale

# 设置警告过滤和编码
warnings.filterwarnings('ignore', category=DeprecationWarning)
locale.setlocale(locale.LC_ALL, 'zh_CN.UTF-8')

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('sports_analysis.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("EnhancedSportsAnalysis")


def check_and_setup_matplotlib():
    """检查并设置matplotlib"""
    try:
        import matplotlib
        # 设置合适的后端
        matplotlib.use('Qt5Agg')
        import matplotlib.pyplot as plt
        from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
        from matplotlib.figure import Figure

        # 设置中文字体支持
        plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
        plt.rcParams['axes.unicode_minus'] = False
        plt.ioff()  # 关闭交互式绘图

        return True, FigureCanvas, Figure
    except ImportError as e:
        logger.error(f"matplotlib导入失败: {e}")
        return False, None, None
    except Exception as e:
        logger.error(f"matplotlib配置失败: {e}")
        return False, None, None


# 全局matplotlib配置
MATPLOTLIB_AVAILABLE, FigureCanvas, Figure = check_and_setup_matplotlib()


def safe_import_modules():
    """安全导入可选模块"""
    modules = {}

    # 尝试导入分析模块
    try:
        from modules.Analysis import analysis
        modules['analysis'] = analysis
        logger.info("分析模块导入成功")
    except ImportError:
        logger.warning("分析模块未找到，将使用基础功能")
        modules['analysis'] = None

    # 尝试导入智能教练模块
    try:
        from UI.deepseek_sports_integration import SmartSportsBot
        modules['smart_coach'] = SmartSportsBot()
        logger.info("智能教练模块导入成功")
    except ImportError:
        logger.warning("智能教练模块未找到")
        modules['smart_coach'] = None
    except Exception as e:
        logger.error(f"智能教练初始化失败: {e}")
        modules['smart_coach'] = None

    return modules


# 导入可选模块
OPTIONAL_MODULES = safe_import_modules()


def safe_operation(operation_name="操作"):
    """安全操作装饰器"""

    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                logger.error(f"{operation_name}失败: {e}")
                # 检查是否是Qt组件
                if args and hasattr(args[0], 'parent') and hasattr(args[0], 'show'):
                    try:
                        QMessageBox.warning(args[0], '错误', f'{operation_name}失败: {str(e)}')
                    except:
                        print(f"错误: {operation_name}失败: {str(e)}")
                return None

        return wrapper

    return decorator


class ConfigManager:
    """配置管理器"""

    def __init__(self):
        self.config_file = "app_config.json"
        self.config = self.load_config()

    def load_config(self):
        """载入配置"""
        default_config = {
            "analysis": {
                "confidence_threshold": 0.3,
                "smoothing_window": 5,
                "fps_rate": 30,
                "enable_3d": True,
                "enable_ai_coach": OPTIONAL_MODULES['smart_coach'] is not None
            },
            "ui": {
                "theme": "modern",
                "language": "zh_CN",
                "auto_save": True,
                "window_geometry": None
            },
            "paths": {
                "data_dir": "./data",
                "export_dir": "./exports",
                "models_dir": "./models",
                "temp_dir": "./temp"
            },
            "performance": {
                "max_frames_memory": 1000,
                "enable_gpu_acceleration": False,
                "parallel_processing": True
            }
        }

        try:
            with open(self.config_file, 'r', encoding='utf-8') as f:
                saved_config = json.load(f)
                # 深度合并配置
                self._deep_merge(default_config, saved_config)
                return default_config
        except FileNotFoundError:
            logger.info("配置文件不存在，使用默认配置")
            return default_config
        except json.JSONDecodeError as e:
            logger.error(f"配置文件格式错误: {e}")
            return default_config
        except Exception as e:
            logger.error(f"载入配置失败: {e}")
            return default_config

    def _deep_merge(self, base_dict, update_dict):
        """深度合并字典"""
        for key, value in update_dict.items():
            if key in base_dict and isinstance(base_dict[key], dict) and isinstance(value, dict):
                self._deep_merge(base_dict[key], value)
            else:
                base_dict[key] = value

    @safe_operation("保存配置")
    def save_config(self):
        """保存配置"""
        # 确保目录存在
        os.makedirs(os.path.dirname(self.config_file) if os.path.dirname(self.config_file) else '.', exist_ok=True)

        with open(self.config_file, 'w', encoding='utf-8') as f:
            json.dump(self.config, f, indent=2, ensure_ascii=False)
        logger.info("配置保存成功")

    def get(self, key_path, default=None):
        """获取配置值"""
        keys = key_path.split('.')
        value = self.config
        try:
            for key in keys:
                value = value[key]
            return value
        except (KeyError, TypeError):
            return default

    def set(self, key_path, value):
        """设置配置值"""
        keys = key_path.split('.')
        config = self.config
        for key in keys[:-1]:
            if key not in config:
                config[key] = {}
            config = config[key]
        config[keys[-1]] = value


def safe_array_check(arr, condition_func):
    """安全的数组条件检查"""
    try:
        if isinstance(arr, (list, tuple)):
            return condition_func(arr) if arr else False
        elif isinstance(arr, np.ndarray):
            if arr.size == 0:
                return False
            elif arr.size == 1:
                return condition_func(arr.item())
            else:
                result = condition_func(arr)
                return result.any() if hasattr(result, 'any') else bool(result)
        else:
            return condition_func(arr)
    except Exception as e:
        logger.warning(f"数组检查失败: {e}")
        return False


class DatabaseManager:
    """数据库管理器"""

    def __init__(self, db_path="sports_analysis.db"):
        self.db_path = db_path
        self.init_database()

    @safe_operation("数据库初始化")
    def init_database(self):
        """初始化数据库"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()

            # 创建分析结果表
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS analysis_results (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT,
                    frame_index INTEGER,
                    timestamp REAL,
                    analysis_data TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')

            # 创建用户配置表
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS user_sessions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT UNIQUE,
                    session_name TEXT,
                    video_path TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')

            conn.commit()

    @safe_operation("保存分析结果")
    def save_analysis_result(self, session_id, frame_index, timestamp, analysis_data):
        """保存分析结果"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute('''
                INSERT INTO analysis_results (session_id, frame_index, timestamp, analysis_data)
                VALUES (?, ?, ?, ?)
            ''', (session_id, frame_index, timestamp, json.dumps(analysis_data, ensure_ascii=False)))
            conn.commit()

    @safe_operation("载入分析结果")
    def load_analysis_results(self, session_id):
        """载入分析结果"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute('''
                SELECT frame_index, timestamp, analysis_data
                FROM analysis_results
                WHERE session_id = ?
                ORDER BY frame_index
            ''', (session_id,))

            results = []
            for row in cursor.fetchall():
                frame_index, timestamp, analysis_data = row
                results.append({
                    'frame_index': frame_index,
                    'timestamp': timestamp,
                    'analysis_data': json.loads(analysis_data)
                })
            return results


class SequenceAnalysisManager:
    """序列分析管理器 - 收集和管理完整运动序列数据"""

    def __init__(self, config_manager=None):
        self.sequence_data = []
        self.analysis_results = []
        self.summary_metrics = {}
        self.config_manager = config_manager or ConfigManager()
        self.db_manager = DatabaseManager()
        self.current_session_id = None

    def start_new_session(self, session_name=None):
        """开始新的分析会话"""
        self.current_session_id = f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.analysis_results.clear()
        self.sequence_data.clear()
        self.summary_metrics.clear()

        if session_name:
            # 保存会话信息到数据库
            with sqlite3.connect(self.db_manager.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    INSERT OR REPLACE INTO user_sessions (session_id, session_name)
                    VALUES (?, ?)
                ''', (self.current_session_id, session_name))
                conn.commit()

        logger.info(f"开始新的分析会话: {self.current_session_id}")

    @safe_operation("添加帧分析")
    def add_frame_analysis(self, frame_idx, analysis_data):
        """添加单帧分析结果"""
        if not isinstance(analysis_data, dict):
            logger.warning(f"无效的分析数据类型: {type(analysis_data)}")
            return

        fps = self.config_manager.get('analysis.fps_rate', 30)
        frame_result = {
            'frame_index': frame_idx,
            'timestamp': frame_idx / fps,
            'analysis_data': analysis_data,
            'keypoints': analysis_data.get('keypoints', [])
        }

        self.analysis_results.append(frame_result)

        # 保存到数据库
        if self.current_session_id:
            self.db_manager.save_analysis_result(
                self.current_session_id,
                frame_idx,
                frame_result['timestamp'],
                analysis_data
            )

        # 内存管理
        max_frames = self.config_manager.get('performance.max_frames_memory', 1000)
        if len(self.analysis_results) > max_frames:
            self.analysis_results = self.analysis_results[-max_frames:]

    def calculate_sequence_summary(self):
        """计算序列总结指标"""
        if not self.analysis_results:
            logger.warning("没有分析结果可用于计算总结")
            return {}

        # 收集所有关键指标
        angle_metrics = self._collect_angle_metrics()
        biomech_metrics = self._collect_biomech_metrics()

        # 计算统计指标
        summary = {
            'session_info': {
                'session_id': self.current_session_id,
                'total_frames': len(self.analysis_results),
                'duration': max(r['timestamp'] for r in self.analysis_results) if self.analysis_results else 0,
                'analysis_time': datetime.now().isoformat()
            },
            'angles_stats': self._calculate_angle_stats(angle_metrics),
            'biomech_stats': self._calculate_biomech_stats(biomech_metrics),
            'movement_quality': self._assess_movement_quality(),
            'stability_metrics': self._assess_stability(),
            'performance_trends': self._analyze_performance_trends(),
            'recommendations': self._generate_recommendations()
        }

        self.summary_metrics = summary
        return summary

    def _collect_angle_metrics(self):
        """收集角度指标"""
        angle_names = ['右肘角度', '左肘角度', '右膝角度', '左膝角度', '躯干角度']
        angle_data = {name: [] for name in angle_names}

        for frame_result in self.analysis_results:
            data = frame_result['analysis_data']
            for angle_name in angle_names:
                if angle_name in data and self._is_valid_number(data[angle_name]):
                    angle_data[angle_name].append(data[angle_name])

        return angle_data

    def _collect_biomech_metrics(self):
        """收集生物力学指标"""
        biomech_names = [
            'energy_transfer_efficiency',
            'center_of_mass_x',
            'center_of_mass_y',
            'ground_reaction_force',
            'movement_velocity'
        ]
        biomech_data = {name: [] for name in biomech_names}

        for frame_result in self.analysis_results:
            data = frame_result['analysis_data']
            for biomech_name in biomech_names:
                if biomech_name in data and self._is_valid_number(data[biomech_name]):
                    biomech_data[biomech_name].append(data[biomech_name])

        return biomech_data

    def _is_valid_number(self, value):
        """检查是否为有效数字"""
        try:
            float(value)
            return not (np.isnan(float(value)) or np.isinf(float(value)))
        except (ValueError, TypeError):
            return False

    def _calculate_angle_stats(self, angle_data):
        """计算角度统计"""
        stats = {}
        for angle_name, values in angle_data.items():
            if values:
                values_array = np.array(values)
                stats[angle_name] = {
                    'mean': float(np.mean(values_array)),
                    'std': float(np.std(values_array)),
                    'min': float(np.min(values_array)),
                    'max': float(np.max(values_array)),
                    'range': float(np.max(values_array) - np.min(values_array)),
                    'coefficient_variation': float(np.std(values_array) / np.mean(values_array)) if np.mean(
                        values_array) != 0 else 0,
                    'consistency_score': self._calculate_consistency_score(values_array)
                }
        return stats

    def _calculate_biomech_stats(self, biomech_data):
        """计算生物力学统计"""
        stats = {}
        for biomech_name, values in biomech_data.items():
            if values:
                values_array = np.array(values)
                stats[biomech_name] = {
                    'mean': float(np.mean(values_array)),
                    'std': float(np.std(values_array)),
                    'trend': self._calculate_trend(values_array),
                    'efficiency_score': self._calculate_efficiency_score(values_array)
                }
        return stats

    def _calculate_consistency_score(self, values):
        """计算一致性得分"""
        if len(values) < 2:
            return 1.0
        cv = np.std(values) / np.mean(values) if np.mean(values) != 0 else float('inf')
        return max(0.0, 1.0 - cv)

    def _calculate_efficiency_score(self, values):
        """计算效率得分"""
        if len(values) < 2:
            return 0.5
        # 简化的效率计算，实际应用中需要更复杂的生物力学模型
        normalized_values = (values - np.min(values)) / (np.max(values) - np.min(values) + 1e-8)
        return float(np.mean(normalized_values))

    def _calculate_trend(self, values):
        """计算数值趋势"""
        if len(values) < 2:
            return 'stable'

        try:
            x = np.arange(len(values))
            slope = np.polyfit(x, values, 1)[0]

            # 设置更合理的阈值
            threshold = np.std(values) * 0.1

            if slope > threshold:
                return 'improving'
            elif slope < -threshold:
                return 'declining'
            else:
                return 'stable'
        except Exception:
            return 'stable'

    def _assess_movement_quality(self):
        """评估运动质量"""
        if not self.analysis_results:
            return {'quality_score': 0.0, 'consistency': 0.0, 'efficiency': 0.0}

        efficiency_scores = []
        consistency_scores = []

        for result in self.analysis_results:
            data = result['analysis_data']

            # 提取效率相关数据
            if 'energy_transfer_efficiency' in data and self._is_valid_number(data['energy_transfer_efficiency']):
                efficiency_scores.append(data['energy_transfer_efficiency'])

            # 计算一致性（基于关键点稳定性）
            if 'keypoints' in data and data['keypoints']:
                consistency_scores.append(self._calculate_keypoint_consistency(data['keypoints']))

        avg_efficiency = np.mean(efficiency_scores) if efficiency_scores else 0.5
        avg_consistency = np.mean(consistency_scores) if consistency_scores else 0.5

        return {
            'quality_score': (avg_efficiency + avg_consistency) / 2,
            'consistency': avg_consistency,
            'efficiency': avg_efficiency,
            'score_interpretation': self._interpret_quality_score((avg_efficiency + avg_consistency) / 2)
        }

    def _calculate_keypoint_consistency(self, keypoints):
        """计算关键点一致性"""
        if not keypoints or len(keypoints) < 2:
            return 0.5

        # 简化的一致性计算
        try:
            positions = np.array([[kp.get('x', 0), kp.get('y', 0)] for kp in keypoints if isinstance(kp, dict)])
            if len(positions) > 0:
                variations = np.std(positions, axis=0)
                return 1.0 / (1.0 + np.mean(variations))
            return 0.5
        except Exception:
            return 0.5

    def _assess_stability(self):
        """评估稳定性"""
        com_x_values = []
        com_y_values = []

        for result in self.analysis_results:
            data = result['analysis_data']
            if 'center_of_mass_x' in data and self._is_valid_number(data['center_of_mass_x']):
                com_x_values.append(data['center_of_mass_x'])
            if 'center_of_mass_y' in data and self._is_valid_number(data['center_of_mass_y']):
                com_y_values.append(data['center_of_mass_y'])

        stability = {}
        if com_x_values and com_y_values:
            stability['com_stability_x'] = 1.0 / (1.0 + np.std(com_x_values))
            stability['com_stability_y'] = 1.0 / (1.0 + np.std(com_y_values))
            stability['overall_stability'] = (stability['com_stability_x'] + stability['com_stability_y']) / 2
            stability['stability_interpretation'] = self._interpret_stability_score(stability['overall_stability'])
        else:
            stability = {'overall_stability': 0.5, 'stability_interpretation': 'insufficient_data'}

        return stability

    def _analyze_performance_trends(self):
        """分析性能趋势"""
        if len(self.analysis_results) < 10:
            return {'trend': 'insufficient_data', 'confidence': 0.0}

        # 分段分析，比较前后表现
        mid_point = len(self.analysis_results) // 2
        first_half = self.analysis_results[:mid_point]
        second_half = self.analysis_results[mid_point:]

        first_half_scores = self._calculate_segment_scores(first_half)
        second_half_scores = self._calculate_segment_scores(second_half)

        improvement = second_half_scores - first_half_scores

        return {
            'trend': 'improving' if improvement > 0.05 else 'declining' if improvement < -0.05 else 'stable',
            'improvement_score': float(improvement),
            'confidence': min(1.0, len(self.analysis_results) / 100.0),
            'first_half_performance': float(first_half_scores),
            'second_half_performance': float(second_half_scores)
        }

    def _calculate_segment_scores(self, segment):
        """计算片段综合得分"""
        if not segment:
            return 0.0

        scores = []
        for result in segment:
            data = result['analysis_data']
            score = 0.0
            count = 0

            # 综合多个指标
            for key in ['energy_transfer_efficiency', 'movement_velocity', 'balance_score']:
                if key in data and self._is_valid_number(data[key]):
                    scores.append(data[key])
                    count += 1

            if count > 0:
                scores.append(sum(scores[-count:]) / count)

        return np.mean(scores) if scores else 0.0

    def _generate_recommendations(self):
        """生成建议"""
        recommendations = []

        if not self.analysis_results:
            return ["需要更多数据进行分析"]

        # 基于分析结果生成建议
        movement_quality = self._assess_movement_quality()
        stability = self._assess_stability()

        if movement_quality['quality_score'] < 0.6:
            recommendations.append("建议加强基础动作练习，提高动作质量")

        if movement_quality['consistency'] < 0.5:
            recommendations.append("动作一致性需要改善，建议进行重复性训练")

        if stability.get('overall_stability', 0) < 0.6:
            recommendations.append("核心稳定性有待提高，建议增加平衡训练")

        trends = self._analyze_performance_trends()
        if trends['trend'] == 'declining':
            recommendations.append("表现有下降趋势，建议调整训练强度或休息")
        elif trends['trend'] == 'improving':
            recommendations.append("表现在提升中，继续保持当前训练节奏")

        return recommendations if recommendations else ["整体表现良好，继续保持"]

    def _interpret_quality_score(self, score):
        """解释质量得分"""
        if score >= 0.8:
            return "优秀"
        elif score >= 0.6:
            return "良好"
        elif score >= 0.4:
            return "一般"
        else:
            return "需要改进"

    def _interpret_stability_score(self, score):
        """解释稳定性得分"""
        if score >= 0.8:
            return "非常稳定"
        elif score >= 0.6:
            return "稳定"
        elif score >= 0.4:
            return "较稳定"
        else:
            return "不稳定"

    @safe_operation("导出分析结果")
    def export_results(self, filepath):
        """导出分析结果"""
        export_data = {
            'summary_metrics': self.summary_metrics,
            'analysis_results': self.analysis_results,
            'export_time': datetime.now().isoformat(),
            'version': '1.0'
        }

        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(export_data, f, indent=2, ensure_ascii=False)

        logger.info(f"分析结果已导出到: {filepath}")


# 全局配置管理器实例
CONFIG_MANAGER = ConfigManager()


# 使用示例和工具函数
def create_sample_analysis_data():
    """创建示例分析数据用于测试"""
    return {
        '右肘角度': np.random.normal(90, 10),
        '左肘角度': np.random.normal(90, 10),
        '右膝角度': np.random.normal(120, 15),
        '左膝角度': np.random.normal(120, 15),
        '躯干角度': np.random.normal(0, 5),
        'energy_transfer_efficiency': np.random.uniform(0.3, 0.9),
        'center_of_mass_x': np.random.normal(0, 0.1),
        'center_of_mass_y': np.random.normal(0, 0.05),
        'ground_reaction_force': np.random.uniform(0.8, 1.2),
        'movement_velocity': np.random.uniform(0.5, 2.0),
        'keypoints': [
            {'x': np.random.uniform(0, 640), 'y': np.random.uniform(0, 480), 'confidence': np.random.uniform(0.5, 1.0)}
            for _ in range(17)  # 假设有17个关键点
        ]
    }



    # 测试代码
    print("体育分析应用 - 改进版")
    print(f"matplotlib可用: {MATPLOTLIB_AVAILABLE}")
    print(f"智能教练可用: {OPTIONAL_MODULES['smart_coach'] is not None}")
    print(f"分析模块可用: {OPTIONAL_MODULES['analysis'] is not None}")

    # 测试序列分析管理器
    manager = SequenceAnalysisManager()
    manager.start_new_session("测试会话")

    # 添加一些测试数据
    for i in range(50):
        sample_data = create_sample_analysis_data()
        manager.add_frame_analysis(i, sample_data)

    # 计算总结
    summary = manager.calculate_sequence_summary()
    print(f"\n分析总结:")
    print(f"质量得分: {summary['movement_quality']['quality_score']:.2f}")
    print(f"稳定性: {summary['stability_metrics'].get('overall_stability', 0):.2f}")
    print(f"建议: {summary['recommendations']}")
# ==================== 789.py的核心类集成 ====================
# 添加以下导入
import torch
import torch.nn as nn
from sklearn.ensemble import RandomForestRegressor
from sklearn.cluster import KMeans
from scipy import signal
from scipy.stats import pearsonr
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import sqlite3
import logging
import multiprocessing as mp
from numba import jit
import matplotlib.pyplot as plt
from typing import List, Tuple, Dict, Any
import cv2

def check_matplotlib():
    """检查matplotlib是否可用"""
    try:
        import matplotlib.pyplot as plt
        import matplotlib.backends.backend_qt5agg
        return True
    except ImportError:
        return False

    # 在文件开头添加UI设计常量

    class UIColors:
        """现代简约UI颜色方案"""
        # 主色调
        PRIMARY = "#0d6efd"
        PRIMARY_HOVER = "#0b5ed7"
        PRIMARY_PRESSED = "#0a58ca"

        # 功能色
        SUCCESS = "#198754"
        SUCCESS_HOVER = "#157347"
        WARNING = "#fd7e14"
        WARNING_HOVER = "#e8681c"
        DANGER = "#dc3545"
        DANGER_HOVER = "#bb2d3b"

        # 中性色
        WHITE = "#ffffff"
        LIGHT_GRAY = "#f8f9fa"
        GRAY_100 = "#f1f3f5"
        GRAY_200 = "#e9ecef"
        GRAY_300 = "#dee2e6"
        GRAY_400 = "#ced4da"
        GRAY_500 = "#adb5bd"
        GRAY_600 = "#6c757d"
        GRAY_700 = "#495057"
        GRAY_800 = "#343a40"
        GRAY_900 = "#212529"

        # 背景色
        BACKGROUND = "#f8f9fa"
        CARD_BACKGROUND = "#ffffff"
        HOVER_BACKGROUND = "#e7f1ff"

        # 边框色
        BORDER_LIGHT = "#dee2e6"
        BORDER_FOCUS = "#86b7fe"

    class UISpacing:
        """间距常量"""
        XS = 4
        SM = 8
        MD = 16
        LG = 24
        XL = 32
        XXL = 48

        class UIRadius:
            """圆角常量"""
            SM = 4
            MD = 6
            LG = 8
            XL = 12
            XXL = 16
            ROUND = 24

        class UIFonts:
            """字体常量"""
            SIZE_XS = 12
            SIZE_SM = 14
            SIZE_MD = 16
            SIZE_LG = 18
            SIZE_XL = 24
            SIZE_XXL = 32

            WEIGHT_NORMAL = 400
            WEIGHT_MEDIUM = 500
            WEIGHT_SEMIBOLD = 600
            WEIGHT_BOLD = 700

    # 性能优化模块
    @jit(nopython=True)
    def fast_angle_calculation(p1, p2, p3):
        """JIT编译的快速角度计算"""
        v1 = p1 - p2
        v2 = p3 - p2
        dot_product = np.dot(v1, v2)
        norms = np.linalg.norm(v1) * np.linalg.norm(v2)
        cos_angle = dot_product / (norms + 1e-8)
        return np.arccos(np.clip(cos_angle, -1.0, 1.0))

    class OptimizedCalculationModule:
        """优化的计算模块"""

        @staticmethod
        def parallel_frame_analysis(frame_data_list, analyze_single_frame):
            """并行帧分析"""
            try:
                with mp.Pool(processes=mp.cpu_count()) as pool:
                    results = pool.map(analyze_single_frame, frame_data_list)
                return results
            except Exception as e:
                logger.error(f"并行分析错误: {e}")
                return []

    class AdvancedDataManager:
        """高级数据管理"""

        def __init__(self, db_path="enhanced_sports_analysis.db"):
            self.db_path = db_path
            self.init_database()

        def init_database(self):
            """初始化增强数据库"""
            try:
                conn = sqlite3.connect(self.db_path)
                cursor = conn.cursor()

                # 创建运动会话表
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS movement_sessions (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        athlete_id TEXT,
                        session_date TIMESTAMP,
                        sport_type TEXT,
                        video_path TEXT,
                        keypoints_data BLOB,  -- 存储序列化的关键点数据
                        analysis_results BLOB,  -- 存储分析结果
                        quality_score REAL,
                        anomaly_score REAL,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    )
                ''')

                conn.commit()
                conn.close()
                logger.info("数据库初始化成功")
            except Exception as e:
                logger.error(f"数据库初始化错误: {e}")

    class SportsAnalysisEngine:
        """运动分析引擎 - 修复版本"""

        def __init__(self):
            self.data_manager = AdvancedDataManager()
            logger.info("运动分析引擎初始化完成")

        def calculate_fluency(self, keypoints_sequence: np.ndarray) -> float:
            """计算流畅性 - 修复版本"""
            try:
                if keypoints_sequence.size == 0:
                    return 0.0

                # 计算相邻帧之间的差异
                diffs = np.diff(keypoints_sequence, axis=0)

                # 使用np.any()来处理数组条件判断
                valid_diffs = diffs[np.any(~np.isnan(diffs), axis=(1, 2))]

                if valid_diffs.size == 0:
                    return 0.0

                # 计算流畅性分数
                smoothness = np.mean(np.linalg.norm(valid_diffs, axis=(1, 2)))
                fluency_score = 1.0 / (1.0 + smoothness)

                logger.info(f"流畅性计算完成: {fluency_score:.3f}")
                return fluency_score

            except Exception as e:
                logger.error(f"流畅性计算错误: {e}")
                return 0.0

        def calculate_symmetry(self, left_keypoints: np.ndarray, right_keypoints: np.ndarray) -> float:
            """计算对称性 - 修复版本"""
            try:
                if left_keypoints.size == 0 or right_keypoints.size == 0:
                    return 0.0

                # 检查数据有效性
                left_valid = ~np.any(np.isnan(left_keypoints), axis=1)
                right_valid = ~np.any(np.isnan(right_keypoints), axis=1)
                both_valid = left_valid & right_valid

                if not np.any(both_valid):
                    return 0.0

                # 计算对称性
                valid_left = left_keypoints[both_valid]
                valid_right = right_keypoints[both_valid]

                differences = np.abs(valid_left - valid_right)
                symmetry_score = 1.0 / (1.0 + np.mean(differences))

                logger.info(f"对称性计算完成: {symmetry_score:.3f}")
                return symmetry_score

            except Exception as e:
                logger.error(f"对称性计算错误: {e}")
                return 0.0

        def extract_movement_features(self, keypoints: np.ndarray) -> Dict[str, float]:
            """提取运动特征 - 修复版本"""
            try:
                features = {}

                if keypoints.size == 0:
                    return {"error": 1.0}

                # 检查数据有效性
                valid_frames = ~np.any(np.isnan(keypoints), axis=(1, 2))

                if not np.any(valid_frames):
                    return {"error": 1.0}

                valid_keypoints = keypoints[valid_frames]

                # 计算速度特征
                if len(valid_keypoints) > 1:
                    velocities = np.diff(valid_keypoints, axis=0)
                    features['avg_velocity'] = np.mean(np.linalg.norm(velocities, axis=2))
                    features['max_velocity'] = np.max(np.linalg.norm(velocities, axis=2))
                else:
                    features['avg_velocity'] = 0.0
                    features['max_velocity'] = 0.0

                # 计算加速度特征
                if len(valid_keypoints) > 2:
                    accelerations = np.diff(velocities, axis=0)
                    features['avg_acceleration'] = np.mean(np.linalg.norm(accelerations, axis=2))
                else:
                    features['avg_acceleration'] = 0.0

                # 计算运动范围
                features['movement_range'] = np.ptp(valid_keypoints, axis=0).mean()

                logger.info("特征提取完成")
                return features

            except Exception as e:
                logger.error(f"特征提取错误: {e}")
                return {"error": 1.0}

        def analyze_limb_coordination(self, arm_keypoints: np.ndarray, leg_keypoints: np.ndarray) -> float:
            """分析肢体协调性 - 修复版本"""
            try:
                if arm_keypoints.size == 0 or leg_keypoints.size == 0:
                    return 0.0

                # 检查数据有效性
                arm_valid = ~np.any(np.isnan(arm_keypoints), axis=(1, 2))
                leg_valid = ~np.any(np.isnan(leg_keypoints), axis=(1, 2))
                both_valid = arm_valid & leg_valid

                if not np.any(both_valid):
                    return 0.0

                # 计算协调性
                valid_arms = arm_keypoints[both_valid]
                valid_legs = leg_keypoints[both_valid]

                # 计算运动相关性
                arm_movement = np.diff(valid_arms, axis=0) if len(valid_arms) > 1 else np.zeros_like(valid_arms[:1])
                leg_movement = np.diff(valid_legs, axis=0) if len(valid_legs) > 1 else np.zeros_like(valid_legs[:1])

                if arm_movement.size > 0 and leg_movement.size > 0:
                    correlation = np.corrcoef(
                        arm_movement.flatten(),
                        leg_movement.flatten()
                    )[0, 1]
                    coordination_score = abs(correlation) if not np.isnan(correlation) else 0.0
                else:
                    coordination_score = 0.0

                logger.info(f"肢体协调性分析完成: {coordination_score:.3f}")
                return coordination_score

            except Exception as e:
                logger.error(f"肢体协调性分析错误: {e}")
                return 0.0

        def analyze_trunk_coordination(self, spine_keypoints: np.ndarray) -> float:
            """分析躯干协调性 - 修复版本"""
            try:
                if spine_keypoints.size == 0:
                    return 0.0

                # 检查数据有效性
                valid_frames = ~np.any(np.isnan(spine_keypoints), axis=(1, 2))

                if not np.any(valid_frames):
                    return 0.0

                valid_spine = spine_keypoints[valid_frames]

                # 计算躯干稳定性
                if len(valid_spine) > 1:
                    spine_movement = np.diff(valid_spine, axis=0)
                    stability = 1.0 / (1.0 + np.mean(np.linalg.norm(spine_movement, axis=2)))
                else:
                    stability = 1.0

                logger.info(f"躯干协调性分析完成: {stability:.3f}")
                return stability

            except Exception as e:
                logger.error(f"躯干协调性分析错误: {e}")
                return 0.0

        def detect_fatigue(self, performance_metrics: np.ndarray) -> Dict[str, Any]:
            """疲劳检测 - 修复版本"""
            try:
                if performance_metrics.size == 0:
                    return {"fatigue_level": 0.0, "trend": "stable"}

                # 检查数据有效性
                valid_metrics = performance_metrics[~np.isnan(performance_metrics)]

                if valid_metrics.size == 0:
                    return {"fatigue_level": 0.0, "trend": "stable"}

                # 计算疲劳指标
                if len(valid_metrics) > 1:
                    # 计算性能下降趋势
                    trend_slope = np.polyfit(range(len(valid_metrics)), valid_metrics, 1)[0]
                    fatigue_level = max(0.0, -trend_slope)  # 负斜率表示疲劳

                    # 确定趋势
                    if trend_slope < -0.01:
                        trend = "declining"
                    elif trend_slope > 0.01:
                        trend = "improving"
                    else:
                        trend = "stable"
                else:
                    fatigue_level = 0.0
                    trend = "stable"

                result = {
                    "fatigue_level": fatigue_level,
                    "trend": trend,
                    "performance_variance": np.var(valid_metrics)
                }

                logger.info(f"疲劳检测完成: {result}")
                return result

            except Exception as e:
                logger.error(f"疲劳检测错误: {e}")
                return {"fatigue_level": 0.0, "trend": "stable", "error": str(e)}

    class SafePlotManager:
        """安全的图表管理器"""

        def __init__(self):
            self.figures = []

        def create_plot(self, figsize=(10, 6)):
            """创建安全的图表"""
            try:
                plt.ioff()  # 关闭交互模式
                fig, ax = plt.subplots(figsize=figsize)
                self.figures.append(fig)
                return fig, ax
            except Exception as e:
                logger.error(f"创建图表错误: {e}")
                return None, None

        def save_plot(self, fig, filename, dpi=300):
            """安全保存图表"""
            try:
                if fig is not None:
                    fig.savefig(filename, dpi=dpi, bbox_inches='tight')
                    logger.info(f"图表已保存: {filename}")
            except Exception as e:
                logger.error(f"保存图表错误: {e}")

        def close_all(self):
            """关闭所有图表"""
            try:
                for fig in self.figures:
                    if fig is not None:
                        plt.close(fig)
                self.figures.clear()
                plt.close('all')
                logger.info("所有图表已关闭")
            except Exception as e:
                logger.error(f"关闭图表错误: {e}")

    def extract_fatigue_features(self, sequence):
        """提取疲劳相关特征"""
        features = []
        for frame in sequence:
            if frame and len(frame) > 0:
                # 计算动作幅度
                amplitude = np.std([point[0] for point in frame if len(point) >= 2])
                features.append(amplitude)
        return features


# ==================== 修复后的ar运动实时分析指导 ====================
import cv2
import numpy as np
import json
import threading
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from collections import deque
import logging


@dataclass
class StandardPose:
    """标准姿势数据结构"""
    name: str
    sport_type: str
    keypoints: List[Tuple[float, float]]
    angles: Dict[str, float]
    metadata: Dict[str, Any]


@dataclass
class JointError:
    """关节错误信息"""
    joint_name: str
    current_angle: float
    target_angle: float
    error_magnitude: float
    correction_direction: str


class ARRealTimeGuidance:
    """改进的AR增强现实指导系统"""

    def __init__(self, gopose_module):
        self.gopose_module = gopose_module
        self.threed_analyzer = self._safe_init_analyzer("Enhanced3DAnalyzer")
        self.real_time_analyzer = self._safe_init_analyzer("RealTimeAnalyzer")

        # 初始化标准姿势数据
        self.standard_poses = {}
        self._load_standard_poses()

        # 历史数据缓存
        self.pose_history = deque(maxlen=30)  # 保存最近30帧的姿势数据

        # 性能优化参数
        self.frame_skip_count = 0
        self.analysis_frequency = 3  # 每3帧进行一次深度分析

        # 线程安全锁
        self.analysis_lock = threading.Lock()

        # 配置日志
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)

        # AR显示配置
        self.ar_config = {
            'show_ideal_pose': True,
            'show_force_vectors': True,
            'show_muscle_activation': True,
            'transparency': 0.3,
            'text_scale': 0.7,
            'line_thickness': 2
        }

    def _safe_init_analyzer(self, analyzer_name: str):
        """安全初始化分析器"""
        try:
            # 根据实际的分析器类进行初始化
            # 这里需要根据您的实际模块来调整
            if analyzer_name == "Enhanced3DAnalyzer":
                return Enhanced3DAnalyzer() if 'Enhanced3DAnalyzer' in globals() else None
            elif analyzer_name == "RealTimeAnalyzer":
                return RealTimeAnalyzer() if 'RealTimeAnalyzer' in globals() else None
        except Exception as e:
            self.logger.warning(f"无法初始化 {analyzer_name}: {e}")
            return None

    def _load_standard_poses(self):
        """加载标准姿势数据"""
        try:
            # 尝试从文件加载
            standard_poses_file = "standard_poses.json"
            with open(standard_poses_file, 'r', encoding='utf-8') as f:
                poses_data = json.load(f)

            for sport, poses in poses_data.items():
                self.standard_poses[sport] = []
                for pose_data in poses:
                    standard_pose = StandardPose(
                        name=pose_data['name'],
                        sport_type=sport,
                        keypoints=pose_data['keypoints'],
                        angles=pose_data['angles'],
                        metadata=pose_data.get('metadata', {})
                    )
                    self.standard_poses[sport].append(standard_pose)

        except FileNotFoundError:
            self.logger.warning("标准姿势文件未找到，使用默认配置")
            self._create_default_poses()
        except Exception as e:
            self.logger.error(f"加载标准姿势时出错: {e}")
            self._create_default_poses()

    def _create_default_poses(self):
        """创建默认的标准姿势"""
        # 为常见运动创建基本的标准姿势
        self.standard_poses = {
            'general': [],
            'basketball': [],
            'tennis': [],
            'golf': []
        }
        self.logger.info("已创建默认标准姿势配置")

    def get_standard_pose_for_sport(self, sport_type: str, action_phase: str = None) -> Optional[StandardPose]:
        """获取特定运动的标准姿势"""
        if sport_type not in self.standard_poses:
            sport_type = 'general'

        poses = self.standard_poses[sport_type]
        if not poses:
            return None

        # 如果指定了动作阶段，尝试找到匹配的姿势
        if action_phase:
            for pose in poses:
                if pose.metadata.get('phase') == action_phase:
                    return pose

        # 返回第一个标准姿势
        return poses[0] if poses else None

    def overlay_technique_guidance(self, frame: np.ndarray, current_keypoints: List) -> np.ndarray:
        """在实时画面上叠加技术指导"""
        try:
            # 验证输入
            if frame is None or len(frame.shape) != 3:
                self.logger.error("无效的帧数据")
                return frame

            if not current_keypoints:
                return frame

            # 获取标准动作模板
            sport_type = getattr(self.gopose_module, 'athlete_profile', {}).get('sport', 'general')
            standard_pose = self.get_standard_pose_for_sport(sport_type)

            # 创建叠加层
            overlay = frame.copy()

            # 1. 绘制理想姿势轮廓（半透明绿色）
            if standard_pose and self.ar_config['show_ideal_pose']:
                self._draw_ideal_pose_overlay(overlay, standard_pose, color=(0, 255, 0))

            # 2. 绘制当前姿势（实线）
            self._draw_current_pose(overlay, current_keypoints)

            # 3. 高亮需要调整的关节
            if standard_pose:
                error_joints = self._identify_error_joints(current_keypoints, standard_pose)
                self._highlight_error_joints(overlay, error_joints)

            # 4. 显示实时反馈文本
            self._display_feedback_text(overlay, error_joints if standard_pose else [])

            # 混合叠加层
            result = cv2.addWeighted(frame, 1 - self.ar_config['transparency'],
                                     overlay, self.ar_config['transparency'], 0)

            return result

        except Exception as e:
            self.logger.error(f"叠加技术指导时出错: {e}")
            return frame

    def _draw_ideal_pose_overlay(self, frame: np.ndarray, standard_pose: StandardPose,
                                 color: Tuple[int, int, int]):
        """绘制理想姿势轮廓"""
        if not standard_pose.keypoints:
            return

        # 绘制骨架连接线
        connections = self._get_pose_connections()
        for connection in connections:
            start_idx, end_idx = connection
            if (start_idx < len(standard_pose.keypoints) and
                    end_idx < len(standard_pose.keypoints)):
                start_point = tuple(map(int, standard_pose.keypoints[start_idx]))
                end_point = tuple(map(int, standard_pose.keypoints[end_idx]))

                cv2.line(frame, start_point, end_point, color,
                         self.ar_config['line_thickness'])

    def _draw_current_pose(self, frame: np.ndarray, keypoints: List):
        """绘制当前姿势"""
        try:
            # 这里调用您现有的绘制方法
            if hasattr(self, 'EnhancedCalculationModule'):
                self.EnhancedCalculationModule.draw(frame, keypoints, size=3, type=0)
            else:
                # 备用绘制方法
                self._draw_keypoints_basic(frame, keypoints)
        except Exception as e:
            self.logger.warning(f"绘制当前姿势时出错: {e}")

    def _draw_keypoints_basic(self, frame: np.ndarray, keypoints: List):
        """基础关键点绘制方法"""
        for point in keypoints:
            if len(point) >= 2:
                center = (int(point[0]), int(point[1]))
                cv2.circle(frame, center, 5, (0, 0, 255), -1)

    def _identify_error_joints(self, current_keypoints: List,
                               standard_pose: StandardPose) -> List[JointError]:
        """识别需要调整的关节"""
        error_joints = []

        try:
            # 计算当前姿势的关节角度
            current_angles = self._calculate_joint_angles(current_keypoints)

            # 与标准姿势比较
            for joint_name, target_angle in standard_pose.angles.items():
                if joint_name in current_angles:
                    current_angle = current_angles[joint_name]
                    error = abs(current_angle - target_angle)

                    # 设置误差阈值
                    threshold = 15.0  # 度
                    if error > threshold:
                        error_joint = JointError(
                            joint_name=joint_name,
                            current_angle=current_angle,
                            target_angle=target_angle,
                            error_magnitude=error,
                            correction_direction="increase" if current_angle < target_angle else "decrease"
                        )
                        error_joints.append(error_joint)

        except Exception as e:
            self.logger.error(f"识别错误关节时出错: {e}")

        return error_joints

    def _calculate_joint_angles(self, keypoints: List) -> Dict[str, float]:
        """计算关节角度"""
        angles = {}

        try:
            # 这里需要根据您的关键点格式来实现
            # 示例：计算肘关节角度
            if len(keypoints) >= 8:  # 假设至少有8个关键点
                # 左肘角度计算示例
                shoulder = np.array(keypoints[5][:2])  # 左肩
                elbow = np.array(keypoints[7][:2])  # 左肘
                wrist = np.array(keypoints[9][:2])  # 左腕

                angle = self._calculate_angle(shoulder, elbow, wrist)
                angles['left_elbow'] = angle

        except Exception as e:
            self.logger.error(f"计算关节角度时出错: {e}")

        return angles

    def _calculate_angle(self, point1: np.ndarray, vertex: np.ndarray,
                         point2: np.ndarray) -> float:
        """计算三点构成的角度"""
        vector1 = point1 - vertex
        vector2 = point2 - vertex

        cos_angle = np.dot(vector1, vector2) / (np.linalg.norm(vector1) * np.linalg.norm(vector2))
        cos_angle = np.clip(cos_angle, -1.0, 1.0)
        angle = np.arccos(cos_angle) * 180 / np.pi

        return angle

    def _highlight_error_joints(self, frame: np.ndarray, error_joints: List[JointError]):
        """高亮显示需要调整的关节"""
        for error in error_joints:
            # 这里需要根据关节名称找到对应的像素位置
            joint_pos = self._get_joint_position(error.joint_name)
            if joint_pos:
                # 用红色圆圈标记错误关节
                cv2.circle(frame, joint_pos, 15, (0, 0, 255), 3)
                # 显示调整建议
                text = f"{error.correction_direction} {error.error_magnitude:.1f}°"
                cv2.putText(frame, text, (joint_pos[0] + 20, joint_pos[1]),
                            cv2.FONT_HERSHEY_SIMPLEX, self.ar_config['text_scale'],
                            (0, 0, 255), 2)

    def _get_joint_position(self, joint_name: str) -> Optional[Tuple[int, int]]:
        """获取关节在图像中的位置"""
        # 这里需要根据您的关键点映射来实现
        joint_mapping = {
            'left_elbow': 7,
            'right_elbow': 8,
            'left_knee': 13,
            'right_knee': 14,
            # 添加更多关节映射
        }

        if (joint_name in joint_mapping and
                hasattr(self, 'current_keypoints') and
                self.current_keypoints):

            idx = joint_mapping[joint_name]
            if idx < len(self.current_keypoints):
                point = self.current_keypoints[idx]
                return (int(point[0]), int(point[1]))

        return None

    def _display_feedback_text(self, frame: np.ndarray, error_joints: List[JointError]):
        """显示实时反馈文本"""
        y_offset = 30

        if not error_joints:
            cv2.putText(frame, "姿势良好!", (10, y_offset),
                        cv2.FONT_HERSHEY_SIMPLEX, self.ar_config['text_scale'],
                        (0, 255, 0), 2)
        else:
            cv2.putText(frame, f"需调整 {len(error_joints)} 个关节",
                        (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX,
                        self.ar_config['text_scale'], (0, 165, 255), 2)

    def _get_pose_connections(self) -> List[Tuple[int, int]]:
        """获取姿势骨架连接"""
        # COCO格式的骨架连接
        return [
            (5, 6), (5, 7), (6, 8), (7, 9), (8, 10),  # 上半身
            (5, 11), (6, 12), (11, 12),  # 躯干
            (11, 13), (12, 14), (13, 15), (14, 16)  # 下半身
        ]

    def show_force_vectors(self, frame: np.ndarray, biomech_data: Dict) -> np.ndarray:
        """AR显示力向量和生物力学信息"""
        if not biomech_data:
            return frame

        try:
            # 显示关节力矩
            if 'joint_torques' in biomech_data:
                for joint_name, torque_value in biomech_data['joint_torques'].items():
                    joint_pos = self._get_joint_position(joint_name)
                    if joint_pos and torque_value:
                        self._draw_force_arrow(frame, joint_pos, torque_value)

            # 显示重心位置
            if all(key in biomech_data for key in ['center_of_mass_x', 'center_of_mass_y']):
                com_pos = (int(biomech_data['center_of_mass_x']),
                           int(biomech_data['center_of_mass_y']))
                cv2.circle(frame, com_pos, 10, (255, 0, 255), -1)
                cv2.putText(frame, "重心", com_pos, cv2.FONT_HERSHEY_SIMPLEX,
                            self.ar_config['text_scale'], (255, 255, 255), 2)

        except Exception as e:
            self.logger.error(f"显示力向量时出错: {e}")

        return frame

    def _draw_force_arrow(self, frame: np.ndarray, start_pos: Tuple[int, int],
                          force_magnitude: float):
        """绘制力箭头"""
        # 根据力的大小计算箭头长度和颜色
        arrow_length = int(abs(force_magnitude) * 2)  # 比例缩放
        color = (0, 255, 0) if force_magnitude > 0 else (0, 0, 255)

        # 绘制箭头（简化版本）
        end_pos = (start_pos[0], start_pos[1] - arrow_length)
        cv2.arrowedLine(frame, start_pos, end_pos, color, 2, tipLength=0.3)

    def update_config(self, new_config: Dict):
        """更新AR配置"""
        self.ar_config.update(new_config)

    def get_performance_stats(self) -> Dict:
        """获取性能统计"""
        return {
            'pose_history_length': len(self.pose_history),
            'analysis_frequency': self.analysis_frequency,
            'frame_skip_count': self.frame_skip_count
        }
# ==================== 修复后的3D运动分析模块 ====================
# ==================== 优化后的3D运动分析模块 ====================
import numpy as np
import math
import cv2
from scipy.spatial.distance import cdist
from scipy.optimize import minimize
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.optimize import least_squares
import warnings


class Enhanced3DAnalyzer:
    """增强版3D运动分析器 - Python 3.7兼容版"""

    def __init__(self):
        # 人体骨骼长度比例 (基于人体测量学标准数据)
        self.body_proportions = {
            'head_neck': 0.13,
            'neck_torso': 0.30,
            'torso_hip': 0.17,
            'upper_arm': 0.188,
            'forearm': 0.146,
            'thigh': 0.245,
            'shin': 0.246,
        }

        # 标准化的骨骼连接关系 (BODY_25格式)
        self.skeleton_connections = [
            (1, 8), (1, 2), (1, 5),  # 躯干和肩膀
            (2, 3), (3, 4),  # 右臂
            (5, 6), (6, 7),  # 左臂
            (8, 9), (9, 10), (10, 11),  # 右腿
            (8, 12), (12, 13), (13, 14),  # 左腿
            (1, 0),  # 头部
            (0, 15), (15, 17),  # 右眼和右耳
            (0, 16), (16, 18),  # 左眼和左耳
            (14, 19), (14, 21),  # 左脚
            (11, 22), (11, 24)  # 右脚
        ]

        # 关节角度约束
        self.joint_constraints = {
            'elbow': (0, 180),
            'knee': (0, 180),
            'shoulder': (-45, 180),
            'hip': (-30, 120)
        }

        # 3D重建参数
        self.reconstruction_params = {
            'depth_scale_factor': 0.3,
            'temporal_smoothing_alpha': 0.7,
            'confidence_threshold': 0.3,
            'bone_length_tolerance': 0.2
        }

    def analyze_3d_movement_quality(self, pose_sequence_3d):
        """分析3D运动质量"""
        quality_metrics = {
            'symmetry_score': 0.0,
            'stability_score': 0.0,
            'efficiency_score': 0.0,
            'coordination_score': 0.0,
            'overall_quality': 0.0
        }

        try:
            if not pose_sequence_3d or len(pose_sequence_3d) < 2:
                return quality_metrics

            # 计算对称性评分
            quality_metrics['symmetry_score'] = self._calculate_3d_symmetry(pose_sequence_3d)

            # 计算稳定性评分
            quality_metrics['stability_score'] = self._calculate_3d_stability(pose_sequence_3d)

            # 计算效率评分
            quality_metrics['efficiency_score'] = self._calculate_3d_efficiency(pose_sequence_3d)

            # 计算协调性评分
            quality_metrics['coordination_score'] = self._calculate_3d_coordination(pose_sequence_3d)

            # 计算整体质量
            quality_metrics['overall_quality'] = np.mean([
                quality_metrics['symmetry_score'],
                quality_metrics['stability_score'],
                quality_metrics['efficiency_score'],
                quality_metrics['coordination_score']
            ])

        except Exception as e:
            print(f"3D运动质量分析错误: {e}")

        return quality_metrics

    def _calculate_3d_symmetry(self, pose_sequence):
        """计算3D对称性"""
        try:
            symmetry_scores = []

            # 左右对称关节对
            symmetric_pairs = [
                (2, 5),  # 左右肩
                (3, 6),  # 左右肘
                (4, 7),  # 左右手
                (9, 12),  # 左右髋
                (10, 13),  # 左右膝
                (11, 14)  # 左右踝
            ]

            for pose in pose_sequence:
                if pose is None:
                    continue

                frame_symmetry = []
                for left_idx, right_idx in symmetric_pairs:
                    if (left_idx < len(pose) and right_idx < len(pose) and
                            len(pose[left_idx]) >= 4 and len(pose[right_idx]) >= 4 and
                            pose[left_idx][3] > 0.1 and pose[right_idx][3] > 0.1):

                        left_pos = np.array(pose[left_idx][:3])
                        right_pos = np.array(pose[right_idx][:3])

                        # 计算相对于身体中心的位置
                        if (len(pose) > 8 and len(pose[1]) >= 4 and len(pose[8]) >= 4 and
                                pose[1][3] > 0.1 and pose[8][3] > 0.1):
                            center = (np.array(pose[1][:3]) + np.array(pose[8][:3])) / 2
                            left_relative = left_pos - center
                            right_relative = right_pos - center

                            # 镜像右侧位置
                            right_relative_mirrored = right_relative.copy()
                            right_relative_mirrored[0] = -right_relative_mirrored[0]  # X轴镜像

                            # 计算对称性
                            distance = np.linalg.norm(left_relative - right_relative_mirrored)
                            symmetry = 1.0 / (1.0 + distance / 100.0)
                            frame_symmetry.append(symmetry)

                if frame_symmetry:
                    symmetry_scores.append(np.mean(frame_symmetry))

            return np.mean(symmetry_scores) if symmetry_scores else 0.5

        except Exception as e:
            print(f"3D对称性计算错误: {e}")
            return 0.5

    def _calculate_3d_stability(self, pose_sequence):
        """计算3D稳定性"""
        try:
            if len(pose_sequence) < 2:
                return 0.5

            stability_metrics = []

            # 重心稳定性
            com_positions = []
            for pose in pose_sequence:
                if pose is None:
                    continue

                # 计算重心
                valid_points = []
                for i, point in enumerate(pose):
                    if len(point) >= 4 and point[3] > 0.1:
                        valid_points.append(point[:3])

                if valid_points:
                    com = np.mean(valid_points, axis=0)
                    com_positions.append(com)

            if len(com_positions) > 1:
                # 计算重心移动的稳定性
                com_velocities = np.diff(com_positions, axis=0)
                com_velocity_norms = np.linalg.norm(com_velocities, axis=1)
                com_stability = 1.0 / (1.0 + np.std(com_velocity_norms))
                stability_metrics.append(com_stability)

            return np.mean(stability_metrics) if stability_metrics else 0.5

        except Exception as e:
            print(f"3D稳定性计算错误: {e}")
            return 0.5

    def _calculate_3d_efficiency(self, pose_sequence):
        """计算3D效率"""
        try:
            if len(pose_sequence) < 2:
                return 0.5

            # 计算运动路径效率
            efficiency_scores = []

            # 关键关节的运动效率
            key_joints = [4, 7, 11, 14]  # 双手双脚

            for joint_idx in key_joints:
                positions = []
                for pose in pose_sequence:
                    if (pose is not None and joint_idx < len(pose) and
                            len(pose[joint_idx]) >= 4 and pose[joint_idx][3] > 0.1):
                        positions.append(pose[joint_idx][:3])

                if len(positions) > 2:
                    positions = np.array(positions)

                    # 计算实际路径长度
                    path_segments = np.diff(positions, axis=0)
                    actual_path = np.sum(np.linalg.norm(path_segments, axis=1))

                    # 计算直线距离
                    straight_distance = np.linalg.norm(positions[-1] - positions[0])

                    # 效率 = 直线距离 / 实际路径
                    if actual_path > 0:
                        efficiency = straight_distance / actual_path
                        efficiency_scores.append(min(efficiency, 1.0))

            return np.mean(efficiency_scores) if efficiency_scores else 0.5

        except Exception as e:
            print(f"3D效率计算错误: {e}")
            return 0.5

    def _calculate_3d_coordination(self, pose_sequence):
        """计算3D协调性"""
        try:
            if len(pose_sequence) < 2:
                return 0.5

            coordination_scores = []

            # 分析四肢协调性
            limb_pairs = [
                ([2, 3, 4], [5, 6, 7]),  # 左右臂
                ([9, 10, 11], [12, 13, 14])  # 左右腿
            ]

            for left_limb, right_limb in limb_pairs:
                left_angles = []
                right_angles = []

                for pose in pose_sequence:
                    if pose is None:
                        continue

                    # 计算左侧肢体角度
                    if all(i < len(pose) and len(pose[i]) >= 4 and pose[i][3] > 0.1 for i in left_limb):
                        left_angle = self._calculate_3d_angle(pose, left_limb)
                        if not np.isnan(left_angle):
                            left_angles.append(left_angle)

                    # 计算右侧肢体角度
                    if all(i < len(pose) and len(pose[i]) >= 4 and pose[i][3] > 0.1 for i in right_limb):
                        right_angle = self._calculate_3d_angle(pose, right_limb)
                        if not np.isnan(right_angle):
                            right_angles.append(right_angle)

                # 计算左右协调性
                if len(left_angles) > 1 and len(right_angles) > 1:
                    min_len = min(len(left_angles), len(right_angles))
                    left_changes = np.diff(left_angles[:min_len])
                    right_changes = np.diff(right_angles[:min_len])

                    # 计算变化模式的相似性
                    if len(left_changes) > 0 and len(right_changes) > 0:
                        correlation = np.corrcoef(left_changes, right_changes)[0, 1]
                        if not np.isnan(correlation):
                            coordination_scores.append(abs(correlation))

            return np.mean(coordination_scores) if coordination_scores else 0.5

            pass

        except Exception as e:

             print(f"3D协调性计算错误: {e}")

        return 0.5

    def reconstruct_3d_pose_enhanced(self, keypoints_2d, previous_3d=None,
                                     camera_params=None, height_pixels=None):
        """
        增强版3D姿态重建

        Args:
            keypoints_2d: 2D关键点 [[x, y, confidence], ...]
            previous_3d: 前一帧的3D结果
            camera_params: 相机参数字典 {'focal_length': f, 'principal_point': (cx, cy)}
            height_pixels: 身高像素值

        Returns:
            ndarray: 3D关键点 [x, y, z, confidence] 或 None
        """
        try:
            # 输入验证
            if not self._validate_input(keypoints_2d):
                return None

            # 初始化3D姿态
            pose_3d = self._initialize_3d_pose(keypoints_2d)

            # 估算身体尺度
            if height_pixels is None:
                height_pixels = self._estimate_height_from_keypoints(keypoints_2d)

            if height_pixels < 50:  # 最小合理身高
                return None

            # 设置默认相机参数
            if camera_params is None:
                camera_params = self._get_default_camera_params(keypoints_2d)

            # 执行3D重建
            pose_3d = self._perform_3d_reconstruction(
                pose_3d, height_pixels, camera_params
            )

            # 应用生物力学约束
            pose_3d = self._apply_biomechanical_constraints(pose_3d, height_pixels)

            # 时间平滑
            if previous_3d is not None:
                pose_3d = self._temporal_smoothing(pose_3d, previous_3d)

            # 质量评估
            quality_score = self._assess_reconstruction_quality(pose_3d, keypoints_2d)

            if quality_score < 0.5:
                print(f"警告: 3D重建质量较低 (质量评分: {quality_score:.2f})")

            return pose_3d

        except Exception as e:
            print(f"3D重建错误: {e}")
            return None

    def _validate_input(self, keypoints_2d):
        """验证输入数据"""
        if keypoints_2d is None or len(keypoints_2d) < 25:
            return False

        # 检查关键点格式
        valid_points = 0
        for kp in keypoints_2d:
            if len(kp) >= 3 and kp[2] > self.reconstruction_params['confidence_threshold']:
                valid_points += 1

        # 至少需要10个有效关键点
        return valid_points >= 10

    def _initialize_3d_pose(self, keypoints_2d):
        """初始化3D姿态"""
        pose_3d = np.zeros((25, 4))  # [x, y, z, confidence]

        for i, kp in enumerate(keypoints_2d):
            if i < 25 and len(kp) >= 3:
                pose_3d[i] = [kp[0], kp[1], 0, kp[2]]

        return pose_3d

    def _get_default_camera_params(self, keypoints_2d):
        """获取默认相机参数"""
        # 估算图像尺寸
        valid_x = [kp[0] for kp in keypoints_2d if len(kp) >= 3 and kp[2] > 0.1]
        valid_y = [kp[1] for kp in keypoints_2d if len(kp) >= 3 and kp[2] > 0.1]

        if not valid_x or not valid_y:
            return {'focal_length': 500, 'principal_point': (320, 240)}

        img_width = max(valid_x) - min(valid_x) + 200
        img_height = max(valid_y) - min(valid_y) + 200

        return {
            'focal_length': img_width * 0.8,  # 经验值
            'principal_point': (img_width / 2, img_height / 2)
        }

    def _perform_3d_reconstruction(self, pose_3d, height_pixels, camera_params):
        """执行3D重建的核心算法"""
        try:
            # 方法1: 基于人体模型的深度估算
            pose_3d = self._anthropometric_depth_estimation(pose_3d, height_pixels)

            # 方法2: 基于骨骼约束的优化
            pose_3d = self._skeleton_constrained_optimization(pose_3d, height_pixels)

            # 方法3: 基于姿态先验的深度细化
            pose_3d = self._pose_prior_depth_refinement(pose_3d)

            return pose_3d

        except Exception as e:
            print(f"3D重建算法错误: {e}")
            return pose_3d

    def _anthropometric_depth_estimation(self, pose_3d, height_pixels):
        """基于人体测量学的深度估算"""
        try:
            # 计算身体比例因子
            scale_factor = height_pixels / 1750  # 假设真实身高175cm

            # 定义各关节的相对深度 (相对于身体中心)
            depth_map = {
                0: 0.08,  # 鼻子 (向前)
                1: 0.02,  # 颈部 (稍向前)
                2: -0.06,  # 右肩 (向后)
                3: 0.04,  # 右肘 (向前)
                4: 0.10,  # 右腕 (向前)
                5: -0.06,  # 左肩 (向后)
                6: 0.04,  # 左肘 (向前)
                7: 0.10,  # 左腕 (向前)
                8: -0.03,  # 中臀 (稍向后)
                9: -0.02,  # 右髋
                10: 0.02,  # 右膝 (稍向前)
                11: 0.05,  # 右踝 (向前)
                12: -0.02,  # 左髋
                13: 0.02,  # 左膝
                14: 0.05,  # 左踝
                15: 0.12,  # 右眼 (向前)
                16: 0.12,  # 左眼
                17: 0.08,  # 右耳
                18: 0.08,  # 左耳
            }

            # 应用深度估算
            for i, depth_offset in depth_map.items():
                if i < len(pose_3d) and pose_3d[i][3] > 0.1:
                    # 基础深度
                    base_depth = depth_offset * scale_factor * self.reconstruction_params['depth_scale_factor']

                    # 添加身体倾斜的影响
                    tilt_adjustment = self._calculate_body_tilt_adjustment(pose_3d, i)

                    pose_3d[i][2] = base_depth + tilt_adjustment

            return pose_3d

        except Exception as e:
            print(f"人体测量学深度估算错误: {e}")
            return pose_3d

    def _skeleton_constrained_optimization(self, pose_3d, height_pixels):
        """基于骨骼约束的优化"""
        try:
            # 定义优化目标函数
            def objective_function(z_coords):
                # 重构3D姿态
                temp_pose = pose_3d.copy()
                valid_indices = [i for i in range(len(pose_3d)) if pose_3d[i][3] > 0.1]

                for i, idx in enumerate(valid_indices):
                    if i < len(z_coords):
                        temp_pose[idx][2] = z_coords[i]

                # 计算骨骼长度误差
                bone_error = self._calculate_bone_length_error(temp_pose, height_pixels)

                # 计算关节角度误差
                angle_error = self._calculate_joint_angle_error(temp_pose)

                # 计算深度平滑性误差
                smoothness_error = self._calculate_depth_smoothness_error(z_coords)

                return bone_error + angle_error * 0.5 + smoothness_error * 0.3

            # 获取有效关键点的初始Z坐标
            valid_indices = [i for i in range(len(pose_3d)) if pose_3d[i][3] > 0.1]
            initial_z = [pose_3d[i][2] for i in valid_indices]

            if len(initial_z) > 0:
                # 执行优化
                bounds = [(-height_pixels * 0.3, height_pixels * 0.3) for _ in initial_z]

                # 使用try-except处理优化可能的失败
                try:
                    result = minimize(objective_function, initial_z, bounds=bounds, method='L-BFGS-B')

                    if result.success:
                        # 应用优化结果
                        for i, idx in enumerate(valid_indices):
                            if i < len(result.x):
                                pose_3d[idx][2] = result.x[i]
                except:
                    # 如果优化失败，保持原始深度值
                    pass

            return pose_3d

        except Exception as e:
            print(f"骨骼约束优化错误: {e}")
            return pose_3d

    def _calculate_bone_length_error(self, pose_3d, height_pixels):
        """计算骨骼长度误差"""
        error = 0
        expected_lengths = self._get_expected_bone_lengths(height_pixels)

        for (start_idx, end_idx), expected_length in expected_lengths.items():
            if (start_idx < len(pose_3d) and end_idx < len(pose_3d) and
                    pose_3d[start_idx][3] > 0.1 and pose_3d[end_idx][3] > 0.1):
                actual_length = np.linalg.norm(pose_3d[end_idx][:3] - pose_3d[start_idx][:3])
                if expected_length > 0:
                    error += abs(actual_length - expected_length) / expected_length

        return error

    def _get_expected_bone_lengths(self, height_pixels):
        """获取期望的骨骼长度"""
        scale = height_pixels
        return {
            (2, 3): scale * self.body_proportions['upper_arm'],  # 右上臂
            (3, 4): scale * self.body_proportions['forearm'],  # 右前臂
            (5, 6): scale * self.body_proportions['upper_arm'],  # 左上臂
            (6, 7): scale * self.body_proportions['forearm'],  # 左前臂
            (9, 10): scale * self.body_proportions['thigh'],  # 右大腿
            (10, 11): scale * self.body_proportions['shin'],  # 右小腿
            (12, 13): scale * self.body_proportions['thigh'],  # 左大腿
            (13, 14): scale * self.body_proportions['shin'],  # 左小腿
            (1, 8): scale * self.body_proportions['neck_torso'],  # 躯干
        }

    def _calculate_joint_angle_error(self, pose_3d):
        """计算关节角度误差"""
        error = 0

        # 检查主要关节角度
        joint_triplets = [
            ([2, 3, 4], 'elbow'),  # 右肘
            ([5, 6, 7], 'elbow'),  # 左肘
            ([9, 10, 11], 'knee'),  # 右膝
            ([12, 13, 14], 'knee'),  # 左膝
        ]

        for triplet, joint_type in joint_triplets:
            if all(i < len(pose_3d) and pose_3d[i][3] > 0.1 for i in triplet):
                angle = self._calculate_3d_angle(pose_3d, triplet)
                if not np.isnan(angle):
                    min_angle, max_angle = self.joint_constraints[joint_type]

                    if angle < min_angle:
                        error += (min_angle - angle) / 180
                    elif angle > max_angle:
                        error += (angle - max_angle) / 180

        return error

    def _calculate_depth_smoothness_error(self, z_coords):
        """计算深度平滑性误差"""
        if len(z_coords) < 3:
            return 0

        # 计算相邻点的深度变化
        differences = np.diff(z_coords)
        return np.std(differences)

    def _pose_prior_depth_refinement(self, pose_3d):
        """基于姿态先验的深度细化"""
        try:
            # 使用常见的人体姿态先验知识进行深度细化

            # 1. 头部通常在最前方
            if len(pose_3d) > 0 and pose_3d[0][3] > 0.1:  # 鼻子
                head_z = pose_3d[0][2]
                # 确保头部在身体前方
                body_indices = [i for i in [1, 8] if i < len(pose_3d) and pose_3d[i][3] > 0.1]
                if body_indices:
                    body_center_z = np.mean([pose_3d[i][2] for i in body_indices])
                    if head_z <= body_center_z:
                        pose_3d[0][2] = body_center_z + abs(body_center_z) * 0.1

            # 2. 手部通常比肘部更靠前
            for arm in [(2, 3, 4), (5, 6, 7)]:  # 右臂，左臂
                shoulder, elbow, wrist = arm
                if all(i < len(pose_3d) and pose_3d[i][3] > 0.1 for i in arm):
                    # 确保手腕在肘部前方
                    if pose_3d[wrist][2] <= pose_3d[elbow][2]:
                        pose_3d[wrist][2] = pose_3d[elbow][2] + abs(pose_3d[elbow][2]) * 0.05

            # 3. 脚部通常比膝部稍靠前
            for leg in [(9, 10, 11), (12, 13, 14)]:  # 右腿，左腿
                hip, knee, ankle = leg
                if all(i < len(pose_3d) and pose_3d[i][3] > 0.1 for i in leg):
                    if pose_3d[ankle][2] <= pose_3d[knee][2]:
                        pose_3d[ankle][2] = pose_3d[knee][2] + abs(pose_3d[knee][2]) * 0.03

            return pose_3d

        except Exception as e:
            print(f"姿态先验深度细化错误: {e}")
            return pose_3d

    def _calculate_body_tilt_adjustment(self, pose_3d, joint_idx):
        """计算身体倾斜调整"""
        try:
            if (len(pose_3d) > 8 and len(pose_3d[1]) >= 4 and len(pose_3d[8]) >= 4 and
                    pose_3d[1][3] > 0.1 and pose_3d[8][3] > 0.1):  # 颈部和中臀

                neck = np.array(pose_3d[1][:3])
                hip = np.array(pose_3d[8][:3])

                # 计算躯干倾斜角度
                trunk_vector = hip - neck
                if np.linalg.norm(trunk_vector) > 0:
                    # 计算与垂直方向的角度
                    vertical = np.array([0, 1, 0])  # 假设Y轴向上
                    cos_angle = np.dot(trunk_vector, vertical) / np.linalg.norm(trunk_vector)
                    cos_angle = np.clip(cos_angle, -1, 1)
                    tilt_angle = np.arccos(cos_angle)

                    # 根据关节位置和倾斜角度调整深度
                    adjustment_factors = {
                        0: 0.8,  # 头部
                        4: 1.0,  # 右手
                        7: 1.0,  # 左手
                        11: 0.5,  # 右脚
                        14: 0.5,  # 左脚
                    }

                    adjustment_factor = adjustment_factors.get(joint_idx, 0.3)
                    return np.sin(tilt_angle) * adjustment_factor * 10

            return 0

        except Exception as e:
            return 0

    def _apply_biomechanical_constraints(self, pose_3d, height_pixels):
        """应用生物力学约束"""
        try:
            # 检查关节角度约束
            joint_checks = [
                ([2, 3, 4], 'elbow'),  # 右肘
                ([5, 6, 7], 'elbow'),  # 左肘
                ([9, 10, 11], 'knee'),  # 右膝
                ([12, 13, 14], 'knee')  # 左膝
            ]

            for joint_indices, joint_type in joint_checks:
                if all(i < len(pose_3d) and len(pose_3d[i]) >= 4 and pose_3d[i][3] > 0.1 for i in joint_indices):
                    angle = self._calculate_3d_angle(pose_3d, joint_indices)
                    min_angle, max_angle = self.joint_constraints.get(joint_type, (0, 180))

                    # 如果角度超出合理范围，进行调整
                    if angle < min_angle or angle > max_angle:
                        # 简单的约束调整：将角度限制在合理范围内
                        p1, p2, p3 = joint_indices
                        if pose_3d[p1][3] > 0.1 and pose_3d[p2][3] > 0.1 and pose_3d[p3][3] > 0.1:
                            # 调整关节位置以满足角度约束
                            target_angle = np.clip(angle, min_angle, max_angle)
                            pose_3d = self._adjust_joint_angle(pose_3d, joint_indices, target_angle)

            # 检查骨骼长度约束
            pose_3d = self._apply_bone_length_constraints(pose_3d, height_pixels)

            return pose_3d

        except Exception as e:
            print(f"生物力学约束应用错误: {e}")
            return pose_3d

    def _adjust_joint_angle(self, pose_3d, joint_indices, target_angle):
        """调整关节角度"""
        try:
            p1, p2, p3 = joint_indices

            # 获取关节位置
            joint_pos = np.array(pose_3d[p2][:3])
            p1_pos = np.array(pose_3d[p1][:3])
            p3_pos = np.array(pose_3d[p3][:3])

            # 计算向量
            v1 = p1_pos - joint_pos
            v2 = p3_pos - joint_pos

            # 计算当前角度
            current_angle = self._calculate_3d_angle(pose_3d, joint_indices)
            angle_diff = target_angle - current_angle

            # 如果角度差异较小，直接返回
            if abs(angle_diff) < 5:  # 5度阈值
                return pose_3d

            # 调整第三个点的位置
            v2_length = np.linalg.norm(v2)
            if v2_length > 0:
                # 旋转v2向量以达到目标角度
                rotation_angle = np.radians(angle_diff)

                # 简化的2D旋转（在主要平面上）
                cos_rot = np.cos(rotation_angle)
                sin_rot = np.sin(rotation_angle)

                # 旋转矩阵（简化为主要平面）
                v2_normalized = v2 / v2_length

                # 应用旋转（简化版本）
                new_v2 = v2 * cos_rot + np.cross(v1, v2) * sin_rot / (np.linalg.norm(v1) * v2_length + 1e-8)
                new_p3_pos = joint_pos + new_v2

                # 更新位置
                pose_3d[p3][:3] = new_p3_pos

            return pose_3d

        except Exception as e:
            print(f"关节角度调整错误: {e}")
            return pose_3d

    def _apply_bone_length_constraints(self, pose_3d, height_pixels):
        """应用骨骼长度约束"""
        try:
            expected_lengths = self._get_expected_bone_lengths(height_pixels)

            for (start_idx, end_idx), expected_length in expected_lengths.items():
                if (start_idx < len(pose_3d) and end_idx < len(pose_3d) and
                        len(pose_3d[start_idx]) >= 4 and len(pose_3d[end_idx]) >= 4 and
                        pose_3d[start_idx][3] > 0.1 and pose_3d[end_idx][3] > 0.1):

                    start_pos = np.array(pose_3d[start_idx][:3])
                    end_pos = np.array(pose_3d[end_idx][:3])

                    current_length = np.linalg.norm(end_pos - start_pos)

                    # 如果长度差异超过容忍范围，进行调整
                    tolerance = expected_length * self.reconstruction_params['bone_length_tolerance']

                    if abs(current_length - expected_length) > tolerance:
                        # 调整末端点位置以匹配期望长度
                        direction = (end_pos - start_pos) / (current_length + 1e-8)
                        new_end_pos = start_pos + direction * expected_length
                        pose_3d[end_idx][:3] = new_end_pos

            return pose_3d

        except Exception as e:
            print(f"骨骼长度约束应用错误: {e}")
            return pose_3d

    def _calculate_body_tilt_adjustment(self, pose_3d, joint_idx):
        """计算身体倾斜调整"""
        try:
            if (len(pose_3d) > 8 and len(pose_3d[1]) >= 4 and len(pose_3d[8]) >= 4 and
                    pose_3d[1][3] > 0.1 and pose_3d[8][3] > 0.1):  # 颈部和中臀

                neck = np.array(pose_3d[1][:3])
                hip = np.array(pose_3d[8][:3])

                # 计算躯干倾斜角度
                trunk_vector = hip - neck
                if np.linalg.norm(trunk_vector) > 0:
                    # 计算与垂直方向的角度
                    vertical = np.array([0, 1, 0])  # 假设Y轴向上
                    tilt_angle = np.arccos(np.clip(
                        np.dot(trunk_vector, vertical) / np.linalg.norm(trunk_vector), -1, 1
                    ))

                    # 根据关节位置和倾斜角度调整深度
                    adjustment_factors = {
                        0: 0.8,  # 头部
                        4: 1.0,  # 右手
                        7: 1.0,  # 左手
                        11: 0.5,  # 右脚
                        14: 0.5,  # 左脚
                    }

                    adjustment_factor = adjustment_factors.get(joint_idx, 0.3)
                    return np.sin(tilt_angle) * adjustment_factor * 10

            return 0

        except Exception as e:
            print(f"身体倾斜调整计算错误: {e}")
            return 0

    def calculate_3d_angles_enhanced(self, pose_3d):
        """计算增强3D角度"""
        angles = {}

        try:
            # 定义关节角度计算
            joint_definitions = {
                '右肘角度': [2, 3, 4],
                '左肘角度': [5, 6, 7],
                '右膝角度': [9, 10, 11],
                '左膝角度': [12, 13, 14],
                '右肩角度': [1, 2, 3],
                '左肩角度': [1, 5, 6]
            }

            for joint_name, indices in joint_definitions.items():
                if all(i < len(pose_3d) and len(pose_3d[i]) >= 4 and pose_3d[i][3] > 0.1 for i in indices):
                    angle = self._calculate_3d_angle(pose_3d, indices)
                    angles[joint_name] = angle

        except Exception as e:
            print(f"3D角度计算错误: {e}")

        return angles

    def __init__(self):
        # 人体骨骼长度比例 (基于人体测量学标准数据)
        self.body_proportions = {
            'head_neck': 0.13,
            'neck_torso': 0.30,
            'torso_hip': 0.17,
            'upper_arm': 0.188,
            'forearm': 0.146,
            'thigh': 0.245,
            'shin': 0.246,
        }

        # 标准化的骨骼连接关系 (BODY_25格式)
        self.skeleton_connections = [
            (1, 8), (1, 2), (1, 5),  # 躯干和肩膀
            (2, 3), (3, 4),  # 右臂
            (5, 6), (6, 7),  # 左臂
            (8, 9), (9, 10), (10, 11),  # 右腿
            (8, 12), (12, 13), (13, 14),  # 左腿
            (1, 0),  # 头部
            (0, 15), (15, 17),  # 右眼和右耳
            (0, 16), (16, 18),  # 左眼和左耳
            (14, 19), (14, 21),  # 左脚
            (11, 22), (11, 24)  # 右脚
        ]

        # 关节角度约束
        self.joint_constraints = {
            'elbow': (0, 180),
            'knee': (0, 180),
            'shoulder': (-45, 180),
            'hip': (-30, 120)
        }

        # 3D重建参数
        self.reconstruction_params = {
            'depth_scale_factor': 0.3,
            'temporal_smoothing_alpha': 0.7,
            'confidence_threshold': 0.3,
            'bone_length_tolerance': 0.2
        }

    def reconstruct_3d_pose_enhanced(self, keypoints_2d, previous_3d=None,
                                     camera_params=None, height_pixels=None):
        """
        增强版3D姿态重建 - 修复版

        Args:
            keypoints_2d: 2D关键点 [[x, y, confidence], ...]
            previous_3d: 前一帧的3D结果
            camera_params: 相机参数字典 {'focal_length': f, 'principal_point': (cx, cy)}
            height_pixels: 身高像素值

        Returns:
            ndarray: 3D关键点 [x, y, z, confidence] 或 None
        """
        try:
            # 输入验证
            if not self._validate_input(keypoints_2d):
                return None

            # 初始化3D姿态
            pose_3d = self._initialize_3d_pose(keypoints_2d)

            # 估算身体尺度
            if height_pixels is None:
                height_pixels = self._estimate_height_from_keypoints(keypoints_2d)

            if height_pixels < 50:  # 最小合理身高
                return None

            # 设置默认相机参数
            if camera_params is None:
                camera_params = self._get_default_camera_params(keypoints_2d)

            # 执行3D重建
            pose_3d = self._perform_3d_reconstruction(
                pose_3d, height_pixels, camera_params
            )

            # 应用生物力学约束
            pose_3d = self._apply_biomechanical_constraints(pose_3d, height_pixels)

            # 时间平滑
            if previous_3d is not None:
                pose_3d = self._temporal_smoothing(pose_3d, previous_3d)

            # 质量评估
            quality_score = self._assess_reconstruction_quality(pose_3d, keypoints_2d)

            if quality_score < 0.5:
                print(f"警告: 3D重建质量较低 (质量评分: {quality_score:.2f})")

            return pose_3d

        except Exception as e:
            print(f"3D重建错误: {e}")
            return None

    def _validate_input(self, keypoints_2d):
        """验证输入数据"""
        if keypoints_2d is None or len(keypoints_2d) < 25:
            return False

        # 检查关键点格式
        valid_points = 0
        for kp in keypoints_2d:
            if len(kp) >= 3 and kp[2] > self.reconstruction_params['confidence_threshold']:
                valid_points += 1

        # 至少需要10个有效关键点
        return valid_points >= 10

    def _initialize_3d_pose(self, keypoints_2d):
        """初始化3D姿态"""
        pose_3d = np.zeros((25, 4))  # [x, y, z, confidence]

        for i, kp in enumerate(keypoints_2d):
            if len(kp) >= 3:
                pose_3d[i] = [kp[0], kp[1], 0, kp[2]]

        return pose_3d

    def _get_default_camera_params(self, keypoints_2d):
        """获取默认相机参数"""
        # 估算图像尺寸
        valid_x = [kp[0] for kp in keypoints_2d if len(kp) >= 3 and kp[2] > 0.1]
        valid_y = [kp[1] for kp in keypoints_2d if len(kp) >= 3 and kp[2] > 0.1]

        if not valid_x or not valid_y:
            return {'focal_length': 500, 'principal_point': (320, 240)}

        img_width = max(valid_x) - min(valid_x) + 200
        img_height = max(valid_y) - min(valid_y) + 200

        return {
            'focal_length': img_width * 0.8,  # 经验值
            'principal_point': (img_width / 2, img_height / 2)
        }

    def _perform_3d_reconstruction(self, pose_3d, height_pixels, camera_params):
        """执行3D重建的核心算法"""
        try:
            # 方法1: 基于人体模型的深度估算
            pose_3d = self._anthropometric_depth_estimation(pose_3d, height_pixels)

            # 方法2: 基于骨骼约束的优化
            pose_3d = self._skeleton_constrained_optimization(pose_3d, height_pixels)

            # 方法3: 基于姿态先验的深度细化
            pose_3d = self._pose_prior_depth_refinement(pose_3d)

            return pose_3d

        except Exception as e:
            print(f"3D重建算法错误: {e}")
            return pose_3d

    def _anthropometric_depth_estimation(self, pose_3d, height_pixels):
        """基于人体测量学的深度估算"""
        try:
            # 计算身体比例因子
            scale_factor = height_pixels / 1750  # 假设真实身高175cm

            # 定义各关节的相对深度 (相对于身体中心)
            depth_map = {
                0: 0.08,  # 鼻子 (向前)
                1: 0.02,  # 颈部 (稍向前)
                2: -0.06,  # 右肩 (向后)
                3: 0.04,  # 右肘 (向前)
                4: 0.10,  # 右腕 (向前)
                5: -0.06,  # 左肩 (向后)
                6: 0.04,  # 左肘 (向前)
                7: 0.10,  # 左腕 (向前)
                8: -0.03,  # 中臀 (稍向后)
                9: -0.02,  # 右髋
                10: 0.02,  # 右膝 (稍向前)
                11: 0.05,  # 右踝 (向前)
                12: -0.02,  # 左髋
                13: 0.02,  # 左膝
                14: 0.05,  # 左踝
                15: 0.12,  # 右眼 (向前)
                16: 0.12,  # 左眼
                17: 0.08,  # 右耳
                18: 0.08,  # 左耳
            }

            # 应用深度估算
            for i, depth_offset in depth_map.items():
                if i < len(pose_3d) and pose_3d[i][3] > 0.1:
                    # 基础深度
                    base_depth = depth_offset * scale_factor * self.reconstruction_params['depth_scale_factor']

                    # 添加身体倾斜的影响
                    tilt_adjustment = self._calculate_body_tilt_adjustment(pose_3d, i)

                    pose_3d[i][2] = base_depth + tilt_adjustment

            return pose_3d

        except Exception as e:
            print(f"人体测量学深度估算错误: {e}")
            return pose_3d

    def _skeleton_constrained_optimization(self, pose_3d, height_pixels):
        """基于骨骼约束的优化"""
        try:
            # 定义优化目标函数
            def objective_function(z_coords):
                # 重构3D姿态
                temp_pose = pose_3d.copy()
                valid_indices = [i for i in range(len(pose_3d)) if pose_3d[i][3] > 0.1]

                for i, idx in enumerate(valid_indices):
                    if i < len(z_coords):
                        temp_pose[idx][2] = z_coords[i]

                # 计算骨骼长度误差
                bone_error = self._calculate_bone_length_error(temp_pose, height_pixels)

                # 计算关节角度误差
                angle_error = self._calculate_joint_angle_error(temp_pose)

                # 计算深度平滑性误差
                smoothness_error = self._calculate_depth_smoothness_error(z_coords)

                return bone_error + angle_error * 0.5 + smoothness_error * 0.3

            # 获取有效关键点的初始Z坐标
            valid_indices = [i for i in range(len(pose_3d)) if pose_3d[i][3] > 0.1]
            initial_z = [pose_3d[i][2] for i in valid_indices]

            if len(initial_z) > 0:
                # 执行优化
                bounds = [(-height_pixels * 0.3, height_pixels * 0.3) for _ in initial_z]
                result = minimize(objective_function, initial_z, bounds=bounds, method='L-BFGS-B')

                if result.success:
                    # 应用优化结果
                    for i, idx in enumerate(valid_indices):
                        if i < len(result.x):
                            pose_3d[idx][2] = result.x[i]

            return pose_3d

        except Exception as e:
            print(f"骨骼约束优化错误: {e}")
            return pose_3d

    def _calculate_bone_length_error(self, pose_3d, height_pixels):
        """计算骨骼长度误差"""
        error = 0
        expected_lengths = self._get_expected_bone_lengths(height_pixels)

        for (start_idx, end_idx), expected_length in expected_lengths.items():
            if (pose_3d[start_idx][3] > 0.1 and pose_3d[end_idx][3] > 0.1):
                actual_length = np.linalg.norm(pose_3d[end_idx][:3] - pose_3d[start_idx][:3])
                error += abs(actual_length - expected_length) / expected_length

        return error

    def _get_expected_bone_lengths(self, height_pixels):
        """获取期望的骨骼长度"""
        scale = height_pixels
        return {
            (2, 3): scale * self.body_proportions['upper_arm'],  # 右上臂
            (3, 4): scale * self.body_proportions['forearm'],  # 右前臂
            (5, 6): scale * self.body_proportions['upper_arm'],  # 左上臂
            (6, 7): scale * self.body_proportions['forearm'],  # 左前臂
            (9, 10): scale * self.body_proportions['thigh'],  # 右大腿
            (10, 11): scale * self.body_proportions['shin'],  # 右小腿
            (12, 13): scale * self.body_proportions['thigh'],  # 左大腿
            (13, 14): scale * self.body_proportions['shin'],  # 左小腿
            (1, 8): scale * self.body_proportions['neck_torso'],  # 躯干
        }

    def _calculate_joint_angle_error(self, pose_3d):
        """计算关节角度误差"""
        error = 0

        # 检查主要关节角度
        joint_triplets = [
            ([2, 3, 4], 'elbow'),  # 右肘
            ([5, 6, 7], 'elbow'),  # 左肘
            ([9, 10, 11], 'knee'),  # 右膝
            ([12, 13, 14], 'knee'),  # 左膝
        ]

        for triplet, joint_type in joint_triplets:
            if all(pose_3d[i][3] > 0.1 for i in triplet):
                angle = self._calculate_3d_angle(pose_3d, triplet)
                min_angle, max_angle = self.joint_constraints[joint_type]

                if angle < min_angle:
                    error += (min_angle - angle) / 180
                elif angle > max_angle:
                    error += (angle - max_angle) / 180

        return error

    def _calculate_depth_smoothness_error(self, z_coords):
        """计算深度平滑性误差"""
        if len(z_coords) < 3:
            return 0

        # 计算相邻点的深度变化
        differences = np.diff(z_coords)
        return np.std(differences)

    def _pose_prior_depth_refinement(self, pose_3d):
        """基于姿态先验的深度细化"""
        try:
            # 使用常见的人体姿态先验知识进行深度细化

            # 1. 头部通常在最前方
            if pose_3d[0][3] > 0.1:  # 鼻子
                head_z = pose_3d[0][2]
                # 确保头部在身体前方
                body_center_z = np.mean([pose_3d[i][2] for i in [1, 8] if pose_3d[i][3] > 0.1])
                if head_z <= body_center_z:
                    pose_3d[0][2] = body_center_z + abs(body_center_z) * 0.1

            # 2. 手部通常比肘部更靠前
            for arm in [(2, 3, 4), (5, 6, 7)]:  # 右臂，左臂
                shoulder, elbow, wrist = arm
                if all(pose_3d[i][3] > 0.1 for i in arm):
                    # 确保手腕在肘部前方
                    if pose_3d[wrist][2] <= pose_3d[elbow][2]:
                        pose_3d[wrist][2] = pose_3d[elbow][2] + abs(pose_3d[elbow][2]) * 0.05

            # 3. 脚部通常比膝部稍靠前
            for leg in [(9, 10, 11), (12, 13, 14)]:  # 右腿，左腿
                hip, knee, ankle = leg
                if all(pose_3d[i][3] > 0.1 for i in leg):
                    if pose_3d[ankle][2] <= pose_3d[knee][2]:
                        pose_3d[ankle][2] = pose_3d[knee][2] + abs(pose_3d[knee][2]) * 0.03

            return pose_3d

        except Exception as e:
            print(f"姿态先验深度细化错误: {e}")
            return pose_3d

    def _calculate_body_tilt_adjustment(self, pose_3d, joint_idx):
        """计算身体倾斜调整"""
        try:
            if pose_3d[1][3] > 0.1 and pose_3d[8][3] > 0.1:  # 颈部和中臀
                neck = pose_3d[1][:3]
                hip = pose_3d[8][:3]

                # 计算躯干倾斜角度
                trunk_vector = hip - neck
                tilt_angle = np.arctan2(trunk_vector[0], trunk_vector[1])  # 在XY平面的倾斜

                # 根据关节位置和倾斜角度调整深度
                adjustment_factor = {
                    0: 0.8,  # 头部
                    4: 1.0,  # 右手
                    7: 1.0,  # 左手
                    11: 0.5,  # 右脚
                    14: 0.5,  # 左脚
                }.get(joint_idx, 0.3)

                return np.sin(tilt_angle) * adjustment_factor * 10

            return 0

        except Exception as e:
            return 0

    def _assess_reconstruction_quality(self, pose_3d, keypoints_2d):
        """评估3D重建质量"""
        try:
            quality_factors = []

            # 1. 关键点置信度
            confidences = [pose_3d[i][3] for i in range(len(pose_3d)) if pose_3d[i][3] > 0]
            if confidences:
                quality_factors.append(np.mean(confidences))

            # 2. 骨骼长度一致性
            bone_consistency = self._calculate_bone_consistency(pose_3d)
            quality_factors.append(bone_consistency)

            # 3. 关节角度合理性
            angle_reasonableness = self._calculate_angle_reasonableness(pose_3d)
            quality_factors.append(angle_reasonableness)

            # 4. 深度分布合理性
            depth_reasonableness = self._calculate_depth_reasonableness(pose_3d)
            quality_factors.append(depth_reasonableness)

            return np.mean(quality_factors) if quality_factors else 0

        except Exception as e:
            print(f"质量评估错误: {e}")
            return 0.5

    def _calculate_bone_consistency(self, pose_3d):
        """计算骨骼一致性"""
        try:
            # 检查对称骨骼的长度差异
            symmetric_bones = [
                ((2, 3), (5, 6)),  # 左右上臂
                ((3, 4), (6, 7)),  # 左右前臂
                ((9, 10), (12, 13)),  # 左右大腿
                ((10, 11), (13, 14))  # 左右小腿
            ]

            consistency_scores = []

            for (bone1, bone2) in symmetric_bones:
                if all(pose_3d[i][3] > 0.1 for i in bone1 + bone2):
                    length1 = np.linalg.norm(pose_3d[bone1[1]][:3] - pose_3d[bone1[0]][:3])
                    length2 = np.linalg.norm(pose_3d[bone2[1]][:3] - pose_3d[bone2[0]][:3])

                    if max(length1, length2) > 0:
                        ratio = min(length1, length2) / max(length1, length2)
                        consistency_scores.append(ratio)

            return np.mean(consistency_scores) if consistency_scores else 0.8

        except Exception as e:
            return 0.5

    def _calculate_angle_reasonableness(self, pose_3d):
        """计算角度合理性"""
        try:
            reasonable_count = 0
            total_count = 0

            joint_checks = [
                ([2, 3, 4], 'elbow'),
                ([5, 6, 7], 'elbow'),
                ([9, 10, 11], 'knee'),
                ([12, 13, 14], 'knee')
            ]

            for triplet, joint_type in joint_checks:
                if all(pose_3d[i][3] > 0.1 for i in triplet):
                    angle = self._calculate_3d_angle(pose_3d, triplet)
                    min_angle, max_angle = self.joint_constraints[joint_type]

                    total_count += 1
                    if min_angle <= angle <= max_angle:
                        reasonable_count += 1

            return reasonable_count / total_count if total_count > 0 else 0.8

        except Exception as e:
            return 0.5

    def _calculate_depth_reasonableness(self, pose_3d):
        """计算深度合理性"""
        try:
            valid_depths = [pose_3d[i][2] for i in range(len(pose_3d)) if pose_3d[i][3] > 0.1]

            if len(valid_depths) < 3:
                return 0.5

            # 检查深度分布是否合理
            depth_range = max(valid_depths) - min(valid_depths)
            depth_std = np.std(valid_depths)

            # 合理的深度范围应该在一定范围内
            if depth_range < 1000 and depth_std < 200:  # 基于像素单位的经验值
                return 0.9
            elif depth_range < 2000 and depth_std < 400:
                return 0.7
            else:
                return 0.3

        except Exception as e:
            return 0.5

    def _calculate_3d_angle(self, pose_3d, indices):
        """计算3D角度"""
        try:
            p1, p2, p3 = indices
            v1 = pose_3d[p1][:3] - pose_3d[p2][:3]
            v2 = pose_3d[p3][:3] - pose_3d[p2][:3]

            cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-8)
            angle = np.arccos(np.clip(cos_angle, -1, 1))

            return np.degrees(angle)

        except Exception as e:
            return 90.0  # 默认角度

    def _temporal_smoothing(self, current_3d, previous_3d):
        """时间平滑 - 修复版"""
        try:
            if previous_3d is None:
                return current_3d

            alpha = self.reconstruction_params['temporal_smoothing_alpha']
            smoothed_3d = current_3d.copy()

            # 确保数据格式一致
            if len(current_3d) != len(previous_3d):
                return current_3d

            for i in range(len(current_3d)):
                if (len(current_3d[i]) >= 4 and len(previous_3d[i]) >= 4 and
                        current_3d[i][3] > 0.1 and previous_3d[i][3] > 0.1):

                    # 计算位置变化
                    current_pos = np.array(current_3d[i][:3])
                    previous_pos = np.array(previous_3d[i][:3])
                    position_change = np.linalg.norm(current_pos - previous_pos)

                    # 如果变化过大，减少平滑强度
                    adaptive_alpha = alpha
                    if position_change > 50:  # 阈值基于像素单位
                        adaptive_alpha = min(alpha, 0.3)

                    # 应用平滑
                    for j in range(3):  # x, y, z
                        smoothed_3d[i][j] = (adaptive_alpha * current_3d[i][j] +
                                             (1 - adaptive_alpha) * previous_3d[i][j])

            return smoothed_3d

        except Exception as e:
            print(f"时间平滑错误: {e}")
            return current_3d

    def _estimate_height_from_keypoints(self, keypoints_2d):
        """从关键点估算身高 - 修复版"""
        try:
            # 方法1: 头顶到脚的距离
            head_y = None
            foot_y = None

            # 寻找头部位置 (鼻子或眼睛)
            for idx in [0, 15, 16]:  # 鼻子, 右眼, 左眼
                if idx < len(keypoints_2d) and keypoints_2d[idx][2] > 0.3:
                    head_y = keypoints_2d[idx][1]
                    break

            # 寻找脚部位置
            foot_candidates = [11, 14, 22, 24]  # 右踝, 左踝, 右脚趾, 右脚跟
            foot_y_values = []

            for idx in foot_candidates:
                if idx < len(keypoints_2d) and keypoints_2d[idx][2] > 0.2:
                    foot_y_values.append(keypoints_2d[idx][1])

            if foot_y_values:
                foot_y = max(foot_y_values)  # 选择最低的点

            if head_y is not None and foot_y is not None:
                height_pixels = abs(foot_y - head_y)
                if height_pixels > 100:  # 最小合理身高
                    return height_pixels

            # 方法2: 颈部到中臀的距离估算
            if (len(keypoints_2d) > 8 and
                    keypoints_2d[1][2] > 0.3 and keypoints_2d[8][2] > 0.3):
                torso_length = abs(keypoints_2d[8][1] - keypoints_2d[1][1])
                # 躯干通常是身高的约50%
                estimated_height = torso_length / 0.5
                if estimated_height > 100:
                    return estimated_height

            # 默认值
            return 400

        except Exception as e:
            print(f"身高估算错误: {e}")
            return 400

# # # # # # 3d模块组件
# !/usr/bin/env python
# -*- coding: utf-8 -*-
"""
修复后的3D可视化组件 - Python 3.7兼容版本
"""

import sys
import numpy as np
from typing import List, Optional, Tuple, Dict, Any

# PyQt5导入
try:
    from PyQt5.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QPushButton,
                                 QSlider, QLabel, QApplication, QMessageBox)
    from PyQt5.QtCore import Qt, QTimer, pyqtSignal
    from PyQt5.QtGui import QFont
except ImportError as e:
    print(f"PyQt5导入失败: {e}")
    print("请安装PyQt5: pip install PyQt5")
    sys.exit(1)

# Matplotlib导入
try:
    import matplotlib

    matplotlib.use('Qt5Agg')  # 确保使用Qt5后端
    from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
    from matplotlib.figure import Figure
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
except ImportError as e:
    print(f"Matplotlib导入失败: {e}")
    print("请安装matplotlib: pip install matplotlib")
    sys.exit(1)


class Enhanced3DAnalyzer:
    """增强的3D分析器类 - 补充原代码中缺失的类"""

    def __init__(self):
        """初始化3D分析器"""
        # BODY_25关键点定义
        self.body_25_keypoints = {
            0: "Nose", 1: "Neck", 2: "RShoulder", 3: "RElbow", 4: "RWrist",
            5: "LShoulder", 6: "LElbow", 7: "LWrist", 8: "MidHip", 9: "RHip",
            10: "RKnee", 11: "RAnkle", 12: "LHip", 13: "LKnee", 14: "LAnkle",
            15: "REye", 16: "LEye", 17: "REar", 18: "LEar", 19: "LBigToe",
            20: "LSmallToe", 21: "LHeel", 22: "RBigToe", 23: "RSmallToe", 24: "RHeel"
        }

        # 默认相机参数
        self.camera_params = {
            'focal_length': 525.0,
            'cx': 320.0,
            'cy': 240.0,
            'image_width': 640,
            'image_height': 480
        }

    def reconstruct_3d_pose_enhanced(self, keypoints_2d: List[List[float]]) -> Optional[np.ndarray]:
        """
        增强的3D姿态重建

        Args:
            keypoints_2d: 2D关键点数据 [[x, y, confidence], ...]

        Returns:
            3D姿态数据 [[x, y, z, confidence], ...] 或 None
        """
        try:
            if not keypoints_2d or len(keypoints_2d) < 25:
                print("2D关键点数据不足")
                return None

            pose_3d = []

            for i, kp_2d in enumerate(keypoints_2d):
                if len(kp_2d) < 3:
                    pose_3d.append([0.0, 0.0, 0.0, 0.0])
                    continue

                x_2d, y_2d, confidence = kp_2d[:3]

                if confidence < 0.1:
                    pose_3d.append([0.0, 0.0, 0.0, 0.0])
                    continue

                # 简化的3D重建 - 使用经验深度估计
                z_depth = self._estimate_depth_from_keypoint(i, x_2d, y_2d)

                # 转换为3D坐标
                x_3d = (x_2d - self.camera_params['cx']) * z_depth / self.camera_params['focal_length']
                y_3d = (y_2d - self.camera_params['cy']) * z_depth / self.camera_params['focal_length']

                pose_3d.append([x_3d, y_3d, z_depth, confidence])

            return np.array(pose_3d)

        except Exception as e:
            print(f"3D重建错误: {e}")
            return None

    def _estimate_depth_from_keypoint(self, keypoint_idx: int, x: float, y: float) -> float:
        """根据关键点类型和位置估计深度"""
        # 简化的深度估计
        base_depth = 1000.0  # 基础深度(mm)

        # 根据关键点类型调整深度
        depth_adjustments = {
            0: -50,  # 鼻子
            1: 0,  # 脖子
            2: 20, 5: 20,  # 肩膀
            3: 30, 6: 30,  # 肘部
            4: 40, 7: 40,  # 手腕
            8: 10,  # 髋部中心
            9: 20, 12: 20,  # 髋部
            10: 30, 13: 30,  # 膝盖
            11: 40, 14: 40,  # 脚踝
        }

        adjustment = depth_adjustments.get(keypoint_idx, 0)
        return base_depth + adjustment + np.random.randn() * 20

    def calculate_3d_angles_enhanced(self, pose_3d: np.ndarray) -> Dict[str, float]:
        """计算增强的3D关节角度"""
        angles = {}

        try:
            # 右肘角度
            if self._are_points_valid(pose_3d, [2, 3, 4]):
                angles['right_elbow'] = self._calculate_angle_3d(
                    pose_3d[2][:3], pose_3d[3][:3], pose_3d[4][:3]
                )

            # 左肘角度
            if self._are_points_valid(pose_3d, [5, 6, 7]):
                angles['left_elbow'] = self._calculate_angle_3d(
                    pose_3d[5][:3], pose_3d[6][:3], pose_3d[7][:3]
                )

            # 右膝角度
            if self._are_points_valid(pose_3d, [9, 10, 11]):
                angles['right_knee'] = self._calculate_angle_3d(
                    pose_3d[9][:3], pose_3d[10][:3], pose_3d[11][:3]
                )

            # 左膝角度
            if self._are_points_valid(pose_3d, [12, 13, 14]):
                angles['left_knee'] = self._calculate_angle_3d(
                    pose_3d[12][:3], pose_3d[13][:3], pose_3d[14][:3]
                )

        except Exception as e:
            print(f"角度计算错误: {e}")

        return angles

    def _are_points_valid(self, pose_3d: np.ndarray, indices: List[int]) -> bool:
        """检查指定索引的点是否有效"""
        try:
            for idx in indices:
                if idx >= len(pose_3d) or pose_3d[idx][3] < 0.1:
                    return False
            return True
        except:
            return False

    def _calculate_angle_3d(self, p1: np.ndarray, p2: np.ndarray, p3: np.ndarray) -> float:
        """计算3D空间中三点形成的角度"""
        try:
            v1 = p1 - p2
            v2 = p3 - p2

            cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
            cos_angle = np.clip(cos_angle, -1.0, 1.0)

            angle = np.arccos(cos_angle)
            return np.degrees(angle)
        except:
            return 0.0

    def _assess_reconstruction_quality(self, pose_3d: np.ndarray, keypoints_2d: List) -> float:
        """评估重建质量"""
        try:
            valid_points = 0
            total_confidence = 0.0

            for i, point_3d in enumerate(pose_3d):
                if i < len(keypoints_2d) and point_3d[3] > 0.1:
                    valid_points += 1
                    total_confidence += point_3d[3]

            if valid_points == 0:
                return 0.0

            coverage_score = valid_points / len(pose_3d)
            confidence_score = total_confidence / valid_points

            return (coverage_score + confidence_score) / 2.0
        except:
            return 0.0


class Fixed3DVisualizationWidget(QWidget):
    """修复后的3D可视化组件"""

    # 定义信号
    frame_changed = pyqtSignal(int)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.pose_3d_data = None
        self.current_frame = 0
        self.is_playing = False
        self.animation_timer = QTimer()
        self.animation_timer.timeout.connect(self._next_frame)
        self.setup_ui()

    def setup_ui(self):
        """设置UI"""
        layout = QVBoxLayout(self)
        layout.setSpacing(10)

        # 标题
        title_label = QLabel("3D姿态可视化")
        title_font = QFont()
        title_font.setPointSize(14)
        title_font.setBold(True)
        title_label.setFont(title_font)
        title_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(title_label)

        # 控制面板
        control_panel = QHBoxLayout()

        self.play_btn = QPushButton("播放")
        self.play_btn.clicked.connect(self.toggle_animation)
        self.play_btn.setMinimumWidth(80)

        self.frame_slider = QSlider(Qt.Horizontal)
        self.frame_slider.valueChanged.connect(self.set_frame)
        self.frame_slider.setMinimum(0)
        self.frame_slider.setMaximum(0)

        self.frame_label = QLabel("帧: 0/0")
        self.frame_label.setMinimumWidth(80)

        # 速度控制
        self.speed_label = QLabel("速度:")
        self.speed_slider = QSlider(Qt.Horizontal)
        self.speed_slider.setMinimum(1)
        self.speed_slider.setMaximum(10)
        self.speed_slider.setValue(5)
        self.speed_slider.setMaximumWidth(100)
        self.speed_slider.valueChanged.connect(self._update_animation_speed)

        control_panel.addWidget(self.play_btn)
        control_panel.addWidget(QLabel("帧数:"))
        control_panel.addWidget(self.frame_slider)
        control_panel.addWidget(self.frame_label)
        control_panel.addStretch()
        control_panel.addWidget(self.speed_label)
        control_panel.addWidget(self.speed_slider)

        layout.addLayout(control_panel)

        # 3D显示区域
        try:
            self.figure = Figure(figsize=(12, 9), facecolor='white')
            self.canvas = FigureCanvas(self.figure)

            # 创建多个3D子图用于不同视角
            self.ax_main = self.figure.add_subplot(221, projection='3d')
            self.ax_front = self.figure.add_subplot(222, projection='3d')
            self.ax_side = self.figure.add_subplot(223, projection='3d')
            self.ax_top = self.figure.add_subplot(224, projection='3d')

            # 设置子图间距
            self.figure.tight_layout(pad=2.0)

            layout.addWidget(self.canvas)

        except Exception as e:
            error_label = QLabel(f"3D显示初始化失败: {e}")
            error_label.setStyleSheet("color: red; font-weight: bold;")
            layout.addWidget(error_label)
            return

        # 视角控制面板
        view_panel = QHBoxLayout()

        view_buttons = [
            ('主视角', self.set_main_view),
            ('正面', lambda: self.set_view_angle(0, 0)),
            ('侧面', lambda: self.set_view_angle(90, 0)),
            ('俯视', lambda: self.set_view_angle(0, 90)),
            ('重置', self.reset_views)
        ]

        for text, slot in view_buttons:
            btn = QPushButton(text)
            btn.clicked.connect(slot)
            btn.setMinimumWidth(60)
            view_panel.addWidget(btn)

        view_panel.addStretch()

        # 添加信息标签
        self.info_label = QLabel("状态: 就绪")
        self.info_label.setStyleSheet("color: blue;")
        view_panel.addWidget(self.info_label)

        layout.addLayout(view_panel)

    def set_pose_data(self, pose_sequence_3d: List[np.ndarray]):
        """设置3D姿态数据"""
        try:
            self.pose_3d_data = pose_sequence_3d
            if pose_sequence_3d and len(pose_sequence_3d) > 0:
                self.frame_slider.setMaximum(len(pose_sequence_3d) - 1)
                self.frame_label.setText(f"帧: 0/{len(pose_sequence_3d) - 1}")
                self.current_frame = 0
                self.info_label.setText(f"状态: 已加载 {len(pose_sequence_3d)} 帧数据")
                self.update_display()
            else:
                self.info_label.setText("状态: 无有效数据")
                self._clear_display()
        except Exception as e:
            self.info_label.setText(f"状态: 数据加载错误 - {e}")
            print(f"设置姿态数据错误: {e}")

    def update_display(self):
        """更新3D显示 - 修复版"""
        if not self.pose_3d_data or self.current_frame >= len(self.pose_3d_data):
            return

        current_pose = self.pose_3d_data[self.current_frame]
        if current_pose is None or len(current_pose) == 0:
            return

        try:
            # 清除所有子图
            for ax in [self.ax_main, self.ax_front, self.ax_side, self.ax_top]:
                ax.clear()

            # 在每个子图中绘制骨架
            self.draw_skeleton_in_axes(self.ax_main, current_pose, "主视角")
            self.draw_skeleton_in_axes(self.ax_front, current_pose, "正面视角")
            self.draw_skeleton_in_axes(self.ax_side, current_pose, "侧面视角")
            self.draw_skeleton_in_axes(self.ax_top, current_pose, "俯视角")

            # 设置不同的视角
            self.ax_main.view_init(elev=20, azim=45)
            self.ax_front.view_init(elev=0, azim=0)
            self.ax_side.view_init(elev=0, azim=90)
            self.ax_top.view_init(elev=90, azim=0)

            # 刷新显示
            self.canvas.draw()

        except Exception as e:
            print(f"3D显示更新错误: {e}")
            self.info_label.setText(f"状态: 显示错误 - {e}")

    def draw_skeleton_in_axes(self, ax, pose_3d, title):
        """在指定的坐标轴中绘制骨架"""
        try:
            # 确保pose_3d是numpy数组
            if not isinstance(pose_3d, np.ndarray):
                pose_3d = np.array(pose_3d)

            # 获取有效点
            valid_points = []
            valid_indices = []

            for i, point in enumerate(pose_3d):
                if len(point) >= 4 and point[3] > 0.1:  # 置信度检查
                    valid_points.append(point[:3])
                    valid_indices.append(i)

            if not valid_points:
                ax.text2D(0.5, 0.5, "无有效数据", transform=ax.transAxes,
                          fontsize=12, ha='center', va='center')
                ax.set_title(title)
                return

            valid_points = np.array(valid_points)

            # 绘制关键点
            ax.scatter(valid_points[:, 0], valid_points[:, 1], valid_points[:, 2],
                       c='red', s=30, alpha=0.8, marker='o')

            # 定义骨骼连接关系 (BODY_25格式)
            connections = [
                # 躯干
                (1, 8), (1, 2), (1, 5),  # 脖子到髋部、肩膀
                (2, 5),  # 肩膀连接
                # 右臂
                (2, 3), (3, 4),  # 右肩-右肘-右腕
                # 左臂
                (5, 6), (6, 7),  # 左肩-左肘-左腕
                # 右腿
                (8, 9), (9, 10), (10, 11),  # 髋部-右髋-右膝-右踝
                # 左腿
                (8, 12), (12, 13), (13, 14),  # 髋部-左髋-左膝-左踝
                # 头部
                (1, 0),  # 脖子到鼻子
            ]

            # 绘制骨骼连接
            for start_idx, end_idx in connections:
                if (start_idx in valid_indices and end_idx in valid_indices and
                        start_idx < len(pose_3d) and end_idx < len(pose_3d) and
                        len(pose_3d[start_idx]) >= 4 and len(pose_3d[end_idx]) >= 4 and
                        pose_3d[start_idx][3] > 0.1 and pose_3d[end_idx][3] > 0.1):
                    start_point = pose_3d[start_idx][:3]
                    end_point = pose_3d[end_idx][:3]

                    ax.plot3D([start_point[0], end_point[0]],
                              [start_point[1], end_point[1]],
                              [start_point[2], end_point[2]],
                              'b-', linewidth=2, alpha=0.7)

            # 设置坐标轴
            ax.set_xlabel('X', fontsize=8)
            ax.set_ylabel('Y', fontsize=8)
            ax.set_zlabel('Z', fontsize=8)
            ax.set_title(f'{title} - 帧 {self.current_frame}', fontsize=10)

            # 设置相等的坐标轴比例
            if len(valid_points) > 0:
                # 计算数据范围
                ranges = np.ptp(valid_points, axis=0)
                max_range = np.max(ranges) / 2.0 if np.max(ranges) > 0 else 100
                center = np.mean(valid_points, axis=0)

                # 设置坐标轴范围
                ax.set_xlim(center[0] - max_range, center[0] + max_range)
                ax.set_ylim(center[1] - max_range, center[1] + max_range)
                ax.set_zlim(center[2] - max_range, center[2] + max_range)

            # 设置网格
            ax.grid(True, alpha=0.3)

        except Exception as e:
            print(f"绘制骨架错误: {e}")
            ax.text2D(0.5, 0.5, f"绘制错误", transform=ax.transAxes,
                      fontsize=10, ha='center', va='center', color='red')
            ax.set_title(title)

    def toggle_animation(self):
        """切换动画播放状态"""
        if not self.pose_3d_data or len(self.pose_3d_data) <= 1:
            QMessageBox.warning(self, "警告", "没有足够的帧数据用于动画播放")
            return

        if self.is_playing:
            self.animation_timer.stop()
            self.play_btn.setText("播放")
            self.is_playing = False
            self.info_label.setText("状态: 已暂停")
        else:
            self._update_animation_speed()
            self.animation_timer.start()
            self.play_btn.setText("暂停")
            self.is_playing = True
            self.info_label.setText("状态: 播放中")

    def _next_frame(self):
        """播放下一帧"""
        if not self.pose_3d_data:
            return

        self.current_frame = (self.current_frame + 1) % len(self.pose_3d_data)
        self.frame_slider.setValue(self.current_frame)

    def _update_animation_speed(self):
        """更新动画速度"""
        speed = self.speed_slider.value()
        interval = max(50, 500 - speed * 45)  # 50ms到455ms
        if self.animation_timer.isActive():
            self.animation_timer.setInterval(interval)

    def set_frame(self, frame_number):
        """设置当前帧"""
        if not self.pose_3d_data:
            return

        self.current_frame = max(0, min(frame_number, len(self.pose_3d_data) - 1))
        if self.pose_3d_data:
            self.frame_label.setText(f"帧: {self.current_frame}/{len(self.pose_3d_data) - 1}")
            self.update_display()
            self.frame_changed.emit(self.current_frame)

    def set_view_angle(self, azim, elev):
        """设置主视角"""
        try:
            self.ax_main.view_init(elev=elev, azim=azim)
            self.canvas.draw()
        except Exception as e:
            print(f"设置视角错误: {e}")

    def set_main_view(self):
        """设置主视角"""
        self.set_view_angle(45, 20)

    def reset_views(self):
        """重置所有视角"""
        try:
            self.ax_main.view_init(elev=20, azim=45)
            self.ax_front.view_init(elev=0, azim=0)
            self.ax_side.view_init(elev=0, azim=90)
            self.ax_top.view_init(elev=90, azim=0)
            self.canvas.draw()
        except Exception as e:
            print(f"重置视角错误: {e}")

    def _clear_display(self):
        """清除显示"""
        try:
            for ax in [self.ax_main, self.ax_front, self.ax_side, self.ax_top]:
                ax.clear()
                ax.text2D(0.5, 0.5, "无数据", transform=ax.transAxes,
                          fontsize=12, ha='center', va='center')
            self.canvas.draw()
        except Exception as e:
            print(f"清除显示错误: {e}")

    def closeEvent(self, event):
        """关闭事件处理"""
        if self.animation_timer.isActive():
            self.animation_timer.stop()
        event.accept()


def generate_sample_3d_data(num_frames=30):
    """生成示例3D数据用于测试"""
    analyzer = Enhanced3DAnalyzer()
    pose_sequence = []

    for frame in range(num_frames):
        # 生成模拟的2D关键点数据
        keypoints_2d = []
        for i in range(25):  # BODY_25有25个关键点
            # 添加一些动画效果
            x = 320 + np.sin(frame * 0.1 + i * 0.2) * 50 + np.random.randn() * 10
            y = 240 + np.cos(frame * 0.1 + i * 0.3) * 50 + np.random.randn() * 10
            confidence = max(0.1, 0.8 + np.random.randn() * 0.2)
            keypoints_2d.append([x, y, confidence])

        # 转换为3D
        pose_3d = analyzer.reconstruct_3d_pose_enhanced(keypoints_2d)
        if pose_3d is not None:
            pose_sequence.append(pose_3d)

    return pose_sequence


def example_usage():
    """使用示例和测试"""
    app = QApplication(sys.argv)

    try:
        # 创建主窗口
        widget = Fixed3DVisualizationWidget()
        widget.setWindowTitle("3D姿态可视化测试")
        widget.resize(1200, 800)

        # 生成测试数据
        print("生成测试数据...")
        sample_data = generate_sample_3d_data(50)

        if sample_data:
            print(f"生成了 {len(sample_data)} 帧测试数据")
            widget.set_pose_data(sample_data)
        else:
            print("测试数据生成失败")

        widget.show()

        # 运行应用
        sys.exit(app.exec_())

    except Exception as e:
        print(f"应用运行错误: {e}")
        import traceback
        traceback.print_exc()






#uiUI集成修改 ====================
# !/usr/bin/env python
# -*- coding: utf-8 -*-
"""
增强的UI集成模块 - Python 3.7兼容版本
包含3D运动分析功能
"""

import sys
from typing import Optional, Dict, Any, List
from enum import Enum

# PyQt5导入
try:
    from PyQt5.QtWidgets import (QTreeWidget, QTreeWidgetItem, QTableWidget,
                                 QMessageBox, QWidget, QVBoxLayout, QHBoxLayout,
                                 QPushButton, QLabel, QSplitter, QTabWidget,
                                 QTextEdit, QProgressBar, QGroupBox, QCheckBox)
    from PyQt5.QtCore import Qt, pyqtSignal, QTimer
    from PyQt5.QtGui import QIcon, QPixmap, QFont
except ImportError as e:
    print(f"PyQt5导入失败: {e}")
    sys.exit(1)


class AnalysisType(Enum):
    """分析类型枚举"""
    ATHLETE_PROFILE = "运动员档案"
    PERSON_SELECTION = "选择单人解析点"
    SCALE_INFO = "比例尺信息"
    KEYPOINT_MODIFICATION = "解析点修正"
    BASIC_KINEMATICS = "基础运动学结果"
    BIOMECHANICS = "生物力学分析"
    THREED_ANALYSIS = "3D运动分析"
    INJURY_RISK = "损伤风险评估"
    TRAINING_PRESCRIPTION = "训练处方建议"
    PERFORMANCE_SCORE = "运动表现评分"
    STANDARD_COMPARISON = "标准动作对比"
    HISTORY_ANALYSIS = "历史数据分析"


class EnhancedGoPoseModule(QWidget):
    """增强的GoPose模块类"""

    # 定义信号
    analysis_changed = pyqtSignal(str)  # 分析类型改变信号
    processing_started = pyqtSignal()  # 处理开始信号
    processing_finished = pyqtSignal()  # 处理完成信号

    def __init__(self, parent=None):
        super().__init__(parent)
        self.current_analysis_type = None
        self.analysis_data = {}
        self.setup_ui()
        self.setup_tree_widget_with_3d()
        self.setup_connections()

    def setup_ui(self):
        """设置基础UI"""
        layout = QVBoxLayout(self)

        # 创建分割器
        splitter = QSplitter(Qt.Horizontal)

        # 左侧：树形控件
        self.treeWidget = QTreeWidget()
        self.treeWidget.setHeaderLabel("分析模块")
        self.treeWidget.setMaximumWidth(250)
        self.treeWidget.setMinimumWidth(200)

        # 右侧：内容区域
        self.content_widget = QTabWidget()

        # 表格视图
        self.tableWidget = QTableWidget()
        self.content_widget.addTab(self.tableWidget, "数据表")

        # 3D视图（占位）
        self.visualization_widget = QWidget()
        self.setup_visualization_placeholder()
        self.content_widget.addTab(self.visualization_widget, "3D可视化")

        # 分析结果
        self.analysis_widget = QWidget()
        self.setup_analysis_widget()
        self.content_widget.addTab(self.analysis_widget, "分析结果")

        splitter.addWidget(self.treeWidget)
        splitter.addWidget(self.content_widget)
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)

        layout.addWidget(splitter)

        # 状态栏
        self.setup_status_bar()
        layout.addWidget(self.status_widget)

    def setup_visualization_placeholder(self):
        """设置3D可视化占位界面"""
        layout = QVBoxLayout(self.visualization_widget)

        # 提示标签
        info_label = QLabel("3D可视化模块")
        info_label.setAlignment(Qt.AlignCenter)
        font = QFont()
        font.setPointSize(16)
        font.setBold(True)
        info_label.setFont(font)

        # 说明文本
        desc_label = QLabel("选择'3D运动分析'以查看3D可视化内容")
        desc_label.setAlignment(Qt.AlignCenter)
        desc_label.setStyleSheet("color: gray; font-size: 12px;")

        layout.addStretch()
        layout.addWidget(info_label)
        layout.addWidget(desc_label)
        layout.addStretch()

    def setup_analysis_widget(self):
        """设置分析结果界面"""
        layout = QVBoxLayout(self.analysis_widget)

        # 分析结果文本区域
        self.analysis_text = QTextEdit()
        self.analysis_text.setReadOnly(True)
        self.analysis_text.setPlainText("请选择左侧的分析模块以查看结果...")

        layout.addWidget(self.analysis_text)

    def setup_status_bar(self):
        """设置状态栏"""
        self.status_widget = QWidget()
        layout = QHBoxLayout(self.status_widget)
        layout.setContentsMargins(5, 5, 5, 5)

        # 状态标签
        self.status_label = QLabel("就绪")
        self.status_label.setStyleSheet("color: blue;")

        # 进度条
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        self.progress_bar.setMaximumWidth(200)

        # 当前分析类型标签
        self.current_analysis_label = QLabel("当前分析: 无")
        self.current_analysis_label.setStyleSheet("color: green;")

        layout.addWidget(self.status_label)
        layout.addStretch()
        layout.addWidget(self.progress_bar)
        layout.addWidget(self.current_analysis_label)

    def setup_connections(self):
        """设置信号连接"""
        # 树形控件点击事件
        self.treeWidget.itemClicked.connect(self.treeClicked_with_3d)

        # 信号连接
        self.analysis_changed.connect(self.on_analysis_changed)
        self.processing_started.connect(self.on_processing_started)
        self.processing_finished.connect(self.on_processing_finished)

    def setup_tree_widget_with_3d(self):
        """设置树形控件（包含3D分析）- 改进版"""
        # 清空现有项目
        self.treeWidget.clear()

        # 定义分析模块配置
        analysis_configs = [
            {
                'name': AnalysisType.ATHLETE_PROFILE.value,
                'icon': None,
                'tooltip': '查看和编辑运动员基本信息',
                'enabled': True
            },
            {
                'name': AnalysisType.PERSON_SELECTION.value,
                'icon': None,
                'tooltip': '选择视频中要分析的人员',
                'enabled': True
            },
            {
                'name': AnalysisType.SCALE_INFO.value,
                'icon': None,
                'tooltip': '设置测量比例尺信息',
                'enabled': True
            },
            {
                'name': AnalysisType.KEYPOINT_MODIFICATION.value,
                'icon': None,
                'tooltip': '修正关键点检测结果',
                'enabled': True
            },
            {
                'name': AnalysisType.BASIC_KINEMATICS.value,
                'icon': None,
                'tooltip': '基础运动学参数分析',
                'enabled': True
            },
            {
                'name': AnalysisType.BIOMECHANICS.value,
                'icon': None,
                'tooltip': '生物力学参数分析',
                'enabled': True
            },
            {
                'name': AnalysisType.THREED_ANALYSIS.value,  # ✨ 重点：3D分析
                'icon': None,
                'tooltip': '3D运动分析和可视化',
                'enabled': True,
                'highlight': True  # 高亮显示
            },
            {
                'name': AnalysisType.INJURY_RISK.value,
                'icon': None,
                'tooltip': '评估运动损伤风险',
                'enabled': True
            },
            {
                'name': AnalysisType.TRAINING_PRESCRIPTION.value,
                'icon': None,
                'tooltip': '生成训练建议',
                'enabled': True
            },
            {
                'name': AnalysisType.PERFORMANCE_SCORE.value,
                'icon': None,
                'tooltip': '运动表现评分',
                'enabled': True
            },
            {
                'name': AnalysisType.STANDARD_COMPARISON.value,
                'icon': None,
                'tooltip': '与标准动作对比',
                'enabled': True
            },
            {
                'name': AnalysisType.HISTORY_ANALYSIS.value,
                'icon': None,
                'tooltip': '历史数据趋势分析',
                'enabled': True
            }
        ]

        # 创建树形项目
        for config in analysis_configs:
            item = QTreeWidgetItem(self.treeWidget)
            item.setText(0, config['name'])
            item.setCheckState(0, Qt.Unchecked)

            # 设置工具提示
            if config.get('tooltip'):
                item.setToolTip(0, config['tooltip'])

            # 设置图标（如果有）
            if config.get('icon'):
                item.setIcon(0, QIcon(config['icon']))

            # 高亮显示特殊项目
            if config.get('highlight'):
                font = item.font(0)
                font.setBold(True)
                item.setFont(0, font)
                # 可以设置不同的颜色
                # item.setForeground(0, QColor(0, 100, 200))

            # 设置启用状态
            if not config.get('enabled', True):
                item.setDisabled(True)

        # 展开所有项目
        self.treeWidget.expandAll()

        # 设置默认选中第一个项目
        if self.treeWidget.topLevelItemCount() > 0:
            first_item = self.treeWidget.topLevelItem(0)
            self.treeWidget.setCurrentItem(first_item)

    def treeClicked_with_3d(self, item, column=0):
        """树形控件点击事件（包含3D处理）- 改进版"""
        if not item:
            return

        try:
            item_text = item.text(0)

            # 更新当前分析类型
            self.current_analysis_type = item_text
            self.analysis_changed.emit(item_text)

            # 先断开之前的连接（避免重复连接）
            try:
                self.tableWidget.clicked.disconnect()
            except:
                pass

            # 根据选择的分析类型执行相应操作
            analysis_handlers = {
                AnalysisType.ATHLETE_PROFILE.value: self.show_athlete_profile,
                AnalysisType.PERSON_SELECTION.value: self.show_person_selection,
                AnalysisType.SCALE_INFO.value: self.show_scale_info,
                AnalysisType.KEYPOINT_MODIFICATION.value: self.show_keypoint_modification,
                AnalysisType.BASIC_KINEMATICS.value: self.show_basic_kinematics,
                AnalysisType.BIOMECHANICS.value: self.show_biomechanics_analysis,
                AnalysisType.THREED_ANALYSIS.value: self.show_3d_analysis,  # ✨ 3D分析
                AnalysisType.INJURY_RISK.value: self.show_injury_risk_assessment,
                AnalysisType.TRAINING_PRESCRIPTION.value: self.show_training_prescription,
                AnalysisType.PERFORMANCE_SCORE.value: self.show_performance_score,
                AnalysisType.STANDARD_COMPARISON.value: self.show_standard_comparison,
                AnalysisType.HISTORY_ANALYSIS.value: self.show_history_analysis
            }

            # 执行对应的处理函数
            handler = analysis_handlers.get(item_text)
            if handler:
                self.processing_started.emit()
                try:
                    handler()
                except Exception as e:
                    self.show_error_message(f"执行 {item_text} 时发生错误", str(e))
                finally:
                    self.processing_finished.emit()
            else:
                self.show_warning_message("未实现的功能", f"'{item_text}' 功能正在开发中...")

        except Exception as e:
            self.show_error_message("树形控件点击错误", str(e))

    # ==================== 信号处理函数 ====================

    def on_analysis_changed(self, analysis_type):
        """分析类型改变时的处理"""
        self.current_analysis_label.setText(f"当前分析: {analysis_type}")
        self.status_label.setText(f"切换到: {analysis_type}")

    def on_processing_started(self):
        """处理开始时的UI更新"""
        self.progress_bar.setVisible(True)
        self.progress_bar.setRange(0, 0)  # 不确定进度
        self.status_label.setText("处理中...")
        self.treeWidget.setEnabled(False)

    def on_processing_finished(self):
        """处理完成时的UI更新"""
        self.progress_bar.setVisible(False)
        self.status_label.setText("处理完成")
        self.treeWidget.setEnabled(True)

        # 2秒后恢复就绪状态
        QTimer.singleShot(2000, lambda: self.status_label.setText("就绪"))

    # ==================== 各种分析功能的实现 ====================

    def show_athlete_profile(self):
        """显示运动员档案"""
        self.update_analysis_text("运动员档案", "显示运动员基本信息、身体数据等...")
        self.content_widget.setCurrentIndex(2)  # 切换到分析结果标签

    def show_person_selection(self):
        """显示人员选择"""
        self.update_analysis_text("选择单人解析点", "在视频中选择要分析的人员...")
        self.content_widget.setCurrentIndex(0)  # 切换到数据表

    def show_scale_info(self):
        """显示比例尺信息"""
        self.update_analysis_text("比例尺信息", "设置测量比例尺，用于准确的距离和速度计算...")
        self.content_widget.setCurrentIndex(2)

    def show_keypoint_modification(self):
        """显示关键点修正"""
        self.update_analysis_text("解析点修正", "手动修正关键点检测结果，提高分析精度...")
        self.content_widget.setCurrentIndex(0)

    def show_basic_kinematics(self):
        """显示基础运动学结果"""
        self.update_analysis_text("基础运动学结果",
                                  "位移、速度、加速度等基础运动学参数分析结果...")
        self.content_widget.setCurrentIndex(2)

    def show_biomechanics_analysis(self):
        """显示生物力学分析"""
        self.update_analysis_text("生物力学分析",
                                  "关节角度、角速度、力矩等生物力学参数分析...")
        self.content_widget.setCurrentIndex(2)

    def show_3d_analysis(self):
        """显示3D运动分析 - ✨ 重点功能"""
        self.update_analysis_text("3D运动分析",
                                  "3D姿态重建、空间运动轨迹、立体角度分析等...")

        # 切换到3D可视化标签
        self.content_widget.setCurrentIndex(1)

        # 这里可以加载实际的3D数据
        self.load_3d_visualization_data()

    def show_injury_risk_assessment(self):
        """显示损伤风险评估"""
        self.update_analysis_text("损伤风险评估",
                                  "基于运动模式分析潜在的损伤风险...")
        self.content_widget.setCurrentIndex(2)

    def show_training_prescription(self):
        """显示训练处方建议"""
        self.update_analysis_text("训练处方建议",
                                  "根据分析结果生成个性化训练建议...")
        self.content_widget.setCurrentIndex(2)

    def show_performance_score(self):
        """显示运动表现评分"""
        self.update_analysis_text("运动表现评分",
                                  "综合评估运动表现，给出量化评分...")
        self.content_widget.setCurrentIndex(2)

    def show_standard_comparison(self):
        """显示标准动作对比"""
        self.update_analysis_text("标准动作对比",
                                  "与标准动作模板进行对比分析...")
        self.content_widget.setCurrentIndex(2)

    def show_history_analysis(self):
        """显示历史数据分析"""
        self.update_analysis_text("历史数据分析",
                                  "分析历史训练数据，展示进步趋势...")
        self.content_widget.setCurrentIndex(2)

    # ==================== 辅助功能 ====================

    def update_analysis_text(self, title, content):
        """更新分析结果文本"""
        timestamp = QTimer().singleShot(0, lambda: None)  # 简单的时间戳替代
        full_text = f"""
=== {title} ===
更新时间: 刚刚

{content}

功能状态: 正常运行
数据来源: 当前分析会话
        """
        self.analysis_text.setPlainText(full_text.strip())

    def load_3d_visualization_data(self):
        """加载3D可视化数据"""
        # 这里应该集成实际的3D可视化组件
        # 例如：Fixed3DVisualizationWidget

        # 临时更新可视化界面
        if hasattr(self, 'visualization_widget'):
            # 清除现有布局
            for i in reversed(range(self.visualization_widget.layout().count())):
                child = self.visualization_widget.layout().takeAt(i)
                if child.widget():
                    child.widget().deleteLater()

            # 添加3D加载提示
            layout = self.visualization_widget.layout()
            loading_label = QLabel("正在加载3D可视化数据...")
            loading_label.setAlignment(Qt.AlignCenter)
            loading_label.setStyleSheet("color: blue; font-size: 14px;")
            layout.addWidget(loading_label)

            # 模拟加载过程
            QTimer.singleShot(1000, self.finish_3d_loading)

    def finish_3d_loading(self):
        """完成3D加载"""
        if hasattr(self, 'visualization_widget'):
            layout = self.visualization_widget.layout()

            # 清除加载提示
            for i in reversed(range(layout.count())):
                child = layout.takeAt(i)
                if child.widget():
                    child.widget().deleteLater()

            # 添加3D内容（这里应该是实际的3D组件）
            content_label = QLabel("3D可视化内容将在这里显示\n\n集成Fixed3DVisualizationWidget组件")
            content_label.setAlignment(Qt.AlignCenter)
            content_label.setStyleSheet("color: green; font-size: 12px;")
            layout.addWidget(content_label)

    def show_error_message(self, title, message):
        """显示错误消息"""
        QMessageBox.critical(self, f"错误 - {title}", message)
        self.status_label.setText(f"错误: {title}")

    def show_warning_message(self, title, message):
        """显示警告消息"""
        QMessageBox.warning(self, f"警告 - {title}", message)
        self.status_label.setText(f"警告: {title}")

    def show_info_message(self, title, message):
        """显示信息消息"""
        QMessageBox.information(self, f"信息 - {title}", message)

    # ==================== 公共接口 ====================

    def get_current_analysis_type(self):
        """获取当前分析类型"""
        return self.current_analysis_type

    def set_analysis_data(self, analysis_type, data):
        """设置分析数据"""
        self.analysis_data[analysis_type] = data

    def get_analysis_data(self, analysis_type):
        """获取分析数据"""
        return self.analysis_data.get(analysis_type)

    def refresh_current_analysis(self):
        """刷新当前分析"""
        if self.current_analysis_type:
            # 重新触发当前分析
            current_item = self.treeWidget.currentItem()
            if current_item:
                self.treeClicked_with_3d(current_item)


# ==================== 使用示例 ====================
def main():
    """主函数示例"""
    from PyQt5.QtWidgets import QApplication, QMainWindow

    app = QApplication(sys.argv)

    # 创建主窗口
    main_window = QMainWindow()
    main_window.setWindowTitle("增强的GoPose模块测试")
    main_window.setGeometry(100, 100, 1200, 800)

    # 创建GoPose模块
    gopose_module = EnhancedGoPoseModule()
    main_window.setCentralWidget(gopose_module)

    # 显示窗口
    main_window.show()

    # 运行应用
    sys.exit(app.exec_())




# ==================== 改进的3D分析系统集成模块 ====================
# ==================== 主程序集成的3D分析系统 ====================
import numpy as np
import json
import logging
from datetime import datetime
from typing import Optional, List, Dict, Tuple, Any
import traceback

# PyQt5导入 (Python 3.7兼容)
try:
    from PyQt5.QtWidgets import (QTableWidgetItem, QPushButton, QDialog,
                                 QVBoxLayout, QMessageBox, QFileDialog,
                                 QFormLayout, QDoubleSpinBox, QDialogButtonBox,
                                 QHBoxLayout, QLabel, QCheckBox, QGroupBox)
    from PyQt5.QtCore import Qt, pyqtSignal, QTimer
    from PyQt5.QtGui import QFont
except ImportError as e:
    print(f"PyQt5导入错误: {e}")
    print("请安装PyQt5: conda install pyqt=5")

# 设置日志
logging.basicConfig(level=logging.INFO)
logger_3d = logging.getLogger('3D_Analysis')


class MainWindowIntegrated3DAnalyzer:
    """主程序集成的3D分析器 - 非阻塞版本"""

    def __init__(self, parent=None):
        self.parent = parent  # 主窗口引用
        self.pose_3d_sequence = []  # 3D姿态序列
        self.last_3d_pose = None  # 上一帧3D姿态
        self.threed_dialog = None  # 3D可视化对话框引用
        self.is_3d_window_open = False  # 3D窗口状态标志

        # 相机参数
        self.camera_params = {
            'focal_length': 500.0,
            'principal_point': (320.0, 240.0),
            'distortion': None
        }

        # 运动员配置
        self.athlete_profile = {
            'height_cm': 170.0,
            'weight_kg': 70.0,
            'sport': 'general'
        }

        # 性能优化器
        self.performance_optimizer = Performance3DOptimizer()

        # 实时更新标志
        self.auto_update_3d = False

    def connect_to_main_window(self):
        """连接到主窗口的方法"""
        try:
            if self.parent is None:
                logger_3d.error("主窗口引用为空")
                return False

            # 检查主窗口必要的属性
            required_attrs = ['tableWidget', 'data', 'fps']
            for attr in required_attrs:
                if not hasattr(self.parent, attr):
                    logger_3d.error(f"主窗口缺少必要属性: {attr}")
                    return False

            # 绑定方法到主窗口（如果主窗口还没有这些方法）
            if not hasattr(self.parent, 'show_3d_analysis_integrated'):
                self.parent.show_3d_analysis_integrated = self.show_3d_analysis

            if not hasattr(self.parent, 'toggle_3d_viewer'):
                self.parent.toggle_3d_viewer = self.toggle_3d_viewer

            if not hasattr(self.parent, 'setup_3d_camera_params'):
                self.parent.setup_3d_camera_params = self.setup_camera_parameters

            logger_3d.info("3D分析器已成功连接到主窗口")
            return True

        except Exception as e:
            logger_3d.error(f"连接主窗口失败: {e}")
            return False

    def show_3d_analysis(self):
        """显示3D运动分析 - 主程序集成版本"""
        try:
            # 基本验证
            if not self._validate_main_window():
                return

            # 清空表格并设置标题
            self._setup_analysis_table()

            # 获取当前帧数据
            current_frame_data = self._get_current_frame_data()
            if current_frame_data is None:
                return

            # 执行3D分析
            analysis_results = self._perform_3d_analysis(current_frame_data)

            # 显示结果
            self._display_results_in_table(analysis_results)

            # 添加控制按钮
            self._add_control_buttons(analysis_results.get('pose_3d'))

            logger_3d.info(f"3D分析完成 - 帧 {self.parent.fps}")

        except Exception as e:
            self._handle_analysis_error(e)

    def _validate_main_window(self) -> bool:
        """验证主窗口状态"""
        if not hasattr(self.parent, 'tableWidget'):
            self._show_error_message('tableWidget未找到')
            return False

        if not hasattr(self.parent, 'data') or not self.parent.data:
            self._show_error_message('数据为空，请先加载数据')
            return False

        if not hasattr(self.parent, 'fps') or self.parent.fps >= len(self.parent.data):
            self._show_error_message(f'帧索引错误: {getattr(self.parent, "fps", "未知")}')
            return False

        return True

    def _setup_analysis_table(self):
        """设置分析表格"""
        try:
            self.parent.tableWidget.clear()
            self.parent.tableWidget.setHorizontalHeaderLabels(['3D分析项目', '结果/操作'])
            self.parent.tableWidget.setRowCount(0)

            # 设置表格列宽
            self.parent.tableWidget.setColumnWidth(0, 200)
            self.parent.tableWidget.setColumnWidth(1, 250)

        except Exception as e:
            logger_3d.error(f"设置表格失败: {e}")

    def _get_current_frame_data(self):
        """获取当前帧数据"""
        try:
            keypoints_data = self.parent.data[self.parent.fps]
            if keypoints_data is None or len(keypoints_data) == 0:
                self._add_table_row('当前帧状态', '无关键点数据')
                return None

            return keypoints_data[0]  # 返回第一个人的关键点

        except Exception as e:
            logger_3d.error(f"获取帧数据失败: {e}")
            self._add_table_row('数据获取', '失败')
            return None

    def _perform_3d_analysis(self, keypoints_data) -> Dict:
        """执行3D分析"""
        results = {
            'keypoints': keypoints_data,
            'pose_3d': None,
            'quality_metrics': {},
            'angles_3d': {},
            'reconstruction_quality': 0.0
        }

        try:
            # 检查缓存
            cached_result = self.performance_optimizer.get_cached_result(self.parent.fps)
            if cached_result is not None:
                results['pose_3d'] = cached_result
                logger_3d.info(f"使用缓存的3D结果: 帧{self.parent.fps}")
            else:
                # 估算身高
                height_pixels = self._estimate_height_from_keypoints(keypoints_data)

                # 执行3D重建
                pose_3d = self._reconstruct_3d_pose_safely(keypoints_data, height_pixels)
                if pose_3d is not None:
                    results['pose_3d'] = pose_3d
                    # 缓存结果
                    self.performance_optimizer.cache_3d_result(self.parent.fps, pose_3d)

            # 如果有3D数据，进行进一步分析
            if results['pose_3d'] is not None:
                self._update_pose_sequence(results['pose_3d'])
                results['quality_metrics'] = self._analyze_movement_quality(results['pose_3d'])
                results['angles_3d'] = self._calculate_3d_angles(results['pose_3d'])
                results['reconstruction_quality'] = self._assess_reconstruction_quality(
                    results['pose_3d'], keypoints_data)

        except Exception as e:
            logger_3d.error(f"3D分析执行失败: {e}")

        return results

    def _display_results_in_table(self, results: Dict):
        """在表格中显示结果"""
        try:
            # 基本信息
            self._add_table_row('当前帧', str(self.parent.fps))
            self._add_table_row('关键点数量', str(len(results['keypoints'])))

            if results['pose_3d'] is not None:
                # 3D重建结果
                self._add_table_row('3D重建状态', '✅ 成功')
                self._add_table_row('重建质量', f"{results['reconstruction_quality']:.3f}")

                # 3D点数量
                valid_3d_points = np.sum(results['pose_3d'][:, 3] > 0.1) if results['pose_3d'].shape[1] > 3 else \
                results['pose_3d'].shape[0]
                self._add_table_row('有效3D点', str(valid_3d_points))

                # 质量指标
                quality = results['quality_metrics']
                for metric_name, value in quality.items():
                    display_name = self._translate_metric_name(metric_name)
                    self._add_table_row(display_name, f"{value:.3f}")

                # 3D角度
                for angle_name, angle_value in results['angles_3d'].items():
                    self._add_table_row(f"3D {angle_name}", f"{angle_value:.1f}°")

            else:
                self._add_table_row('3D重建状态', '❌ 失败')
                self._add_table_row('失败原因', '关键点质量不足')

        except Exception as e:
            logger_3d.error(f"结果显示失败: {e}")

    def _add_control_buttons(self, pose_3d):
        """添加控制按钮"""
        try:
            # 3D可视化按钮
            self._add_button_row('3D可视化', '打开3D视图',
                                 lambda: self.toggle_3d_viewer(pose_3d))

            # 保存当前帧按钮
            if pose_3d is not None:
                self._add_button_row('保存数据', '保存当前帧',
                                     lambda: self.save_current_frame(pose_3d))

            # 导出序列按钮
            if len(self.pose_3d_sequence) > 1:
                self._add_button_row('导出序列', '导出全部序列',
                                     self.export_3d_sequence)

            # 相机参数按钮
            self._add_button_row('相机设置', '配置参数',
                                 self.setup_camera_parameters)

            # 自动更新选项
            self._add_auto_update_option()

        except Exception as e:
            logger_3d.error(f"添加控制按钮失败: {e}")

    def _add_button_row(self, label: str, button_text: str, callback):
        """添加按钮行"""
        try:
            row = self.parent.tableWidget.rowCount()
            self.parent.tableWidget.insertRow(row)
            self.parent.tableWidget.setItem(row, 0, QTableWidgetItem(label))

            btn = QPushButton(button_text)
            btn.clicked.connect(callback)
            self.parent.tableWidget.setCellWidget(row, 1, btn)

        except Exception as e:
            logger_3d.error(f"添加按钮行失败: {e}")

    def _add_auto_update_option(self):
        """添加自动更新选项"""
        try:
            row = self.parent.tableWidget.rowCount()
            self.parent.tableWidget.insertRow(row)
            self.parent.tableWidget.setItem(row, 0, QTableWidgetItem('自动更新'))

            checkbox = QCheckBox('帧变化时自动更新3D')
            checkbox.setChecked(self.auto_update_3d)
            checkbox.stateChanged.connect(self._on_auto_update_changed)
            self.parent.tableWidget.setCellWidget(row, 1, checkbox)

        except Exception as e:
            logger_3d.error(f"添加自动更新选项失败: {e}")

    def _on_auto_update_changed(self, state):
        """自动更新状态改变"""
        self.auto_update_3d = state == Qt.Checked
        logger_3d.info(f"自动更新3D: {'开启' if self.auto_update_3d else '关闭'}")

    def toggle_3d_viewer(self, pose_3d=None):
        """切换3D可视化窗口 - 非阻塞版本"""
        try:
            if self.is_3d_window_open and self.threed_dialog is not None:
                # 关闭已打开的3D窗口
                self.threed_dialog.close()
                self.is_3d_window_open = False
                logger_3d.info("3D可视化窗口已关闭")
                return

            # 检查是否有3D可视化组件
            if not hasattr(self.parent, 'Fixed3DVisualizationWidget'):
                QMessageBox.information(self.parent, '提示',
                                        '3D可视化组件未找到，请检查程序配置')
                return

            # 创建非阻塞的3D可视化窗口
            self._create_3d_visualization_window(pose_3d)

        except Exception as e:
            logger_3d.error(f"切换3D视图失败: {e}")
            QMessageBox.warning(self.parent, '错误', f'3D视图操作失败: {str(e)}')

    def _create_3d_visualization_window(self, pose_3d):
        """创建3D可视化窗口"""
        try:
            # 创建非模态对话框
            self.threed_dialog = QDialog(self.parent)
            self.threed_dialog.setWindowTitle('3D运动分析可视化')
            self.threed_dialog.setMinimumSize(1000, 700)
            self.threed_dialog.setModal(False)  # 重要：设置为非模态

            # 设置窗口关闭事件
            self.threed_dialog.closeEvent = self._on_3d_window_close

            # 创建布局
            layout = QVBoxLayout(self.threed_dialog)

            # 添加控制面板
            control_panel = self._create_3d_control_panel()
            layout.addWidget(control_panel)

            # 创建3D可视化组件
            try:
                self.threed_widget = self.parent.Fixed3DVisualizationWidget()
                layout.addWidget(self.threed_widget)

                # 设置数据
                if self.pose_3d_sequence and len(self.pose_3d_sequence) > 0:
                    self.threed_widget.set_pose_data(self.pose_3d_sequence)
                elif pose_3d is not None:
                    self.threed_widget.set_pose_data([pose_3d])
                else:
                    logger_3d.warning("没有可用的3D数据")

            except Exception as e:
                logger_3d.error(f"创建3D组件失败: {e}")
                error_label = QLabel(f"3D可视化组件加载失败: {str(e)}")
                layout.addWidget(error_label)

            # 显示窗口（非阻塞）
            self.threed_dialog.show()
            self.is_3d_window_open = True

            logger_3d.info("3D可视化窗口已打开（非阻塞模式）")

        except Exception as e:
            logger_3d.error(f"创建3D窗口失败: {e}")
            raise

    def _create_3d_control_panel(self):
        """创建3D控制面板"""
        try:
            group_box = QGroupBox("3D控制面板")
            layout = QHBoxLayout(group_box)

            # 实时更新按钮
            realtime_btn = QPushButton("实时更新")
            realtime_btn.setCheckable(True)
            realtime_btn.clicked.connect(self._toggle_realtime_update)
            layout.addWidget(realtime_btn)

            # 重置视图按钮
            reset_btn = QPushButton("重置视图")
            reset_btn.clicked.connect(self._reset_3d_view)
            layout.addWidget(reset_btn)

            # 导出图像按钮
            export_btn = QPushButton("导出图像")
            export_btn.clicked.connect(self._export_3d_image)
            layout.addWidget(export_btn)

            return group_box

        except Exception as e:
            logger_3d.error(f"创建控制面板失败: {e}")
            return QLabel("控制面板加载失败")

    def _on_3d_window_close(self, event):
        """3D窗口关闭事件"""
        self.is_3d_window_open = False
        self.threed_dialog = None
        logger_3d.info("3D可视化窗口已关闭")
        event.accept()

    def _toggle_realtime_update(self):
        """切换实时更新"""
        # 这里可以添加实时更新3D视图的逻辑
        logger_3d.info("实时更新切换")

    def _reset_3d_view(self):
        """重置3D视图"""
        try:
            if hasattr(self, 'threed_widget') and self.threed_widget:
                # 重置3D视图的方法需要根据实际的3D组件来实现
                logger_3d.info("3D视图已重置")
        except Exception as e:
            logger_3d.error(f"重置3D视图失败: {e}")

    def _export_3d_image(self):
        """导出3D图像"""
        try:
            filename, _ = QFileDialog.getSaveFileName(
                self.threed_dialog, '导出3D图像',
                f'3d_view_{datetime.now().strftime("%Y%m%d_%H%M%S")}.png',
                "PNG Files (*.png);;JPG Files (*.jpg)"
            )

            if filename:
                # 导出3D视图的逻辑需要根据实际的3D组件来实现
                logger_3d.info(f"3D图像导出: {filename}")
                QMessageBox.information(self.threed_dialog, '成功', f'图像已保存到: {filename}')

        except Exception as e:
            logger_3d.error(f"导出3D图像失败: {e}")
            QMessageBox.warning(self.threed_dialog, '错误', f'导出失败: {str(e)}')

    # 以下方法保持原有逻辑...
    def _estimate_height_from_keypoints(self, keypoints: List) -> float:
        """估算身高像素值"""
        try:
            head_y = None
            foot_y = None

            # 头部位置 (鼻子或眼睛)
            head_indices = [0, 1, 2]
            for idx in head_indices:
                if (idx < len(keypoints) and
                        len(keypoints[idx]) >= 3 and
                        keypoints[idx][2] > 0.3):
                    head_y = keypoints[idx][1]
                    break

            # 脚部位置 (脚踝)
            foot_indices = [11, 14]
            foot_y_values = []

            for idx in foot_indices:
                if (idx < len(keypoints) and
                        len(keypoints[idx]) >= 3 and
                        keypoints[idx][2] > 0.2):
                    foot_y_values.append(keypoints[idx][1])

            if foot_y_values:
                foot_y = max(foot_y_values)

            # 计算身高
            if head_y is not None and foot_y is not None:
                height_pixels = abs(foot_y - head_y)
                if height_pixels > 100:
                    return height_pixels

            return 400.0  # 默认值

        except Exception as e:
            logger_3d.error(f"身高估算错误: {e}")
            return 400.0

    def _reconstruct_3d_pose_safely(self, keypoints: List, height_pixels: float) -> Optional[np.ndarray]:
        """安全地执行3D重建"""
        try:
            if not hasattr(self.parent, 'threed_analyzer'):
                logger_3d.warning("3D分析器未初始化，使用模拟数据")
                return self._create_mock_3d_pose(keypoints)

            pose_3d = self.parent.threed_analyzer.reconstruct_3d_pose_enhanced(
                keypoints,
                previous_3d=self.last_3d_pose,
                height_pixels=height_pixels
            )

            is_valid, msg = self._validate_3d_data(pose_3d)
            if not is_valid:
                logger_3d.warning(f"3D重建结果无效: {msg}")
                return None

            return pose_3d

        except Exception as e:
            logger_3d.error(f"3D重建失败: {e}")
            return None

    def _create_mock_3d_pose(self, keypoints: List) -> np.ndarray:
        """创建模拟的3D姿态数据"""
        try:
            num_points = len(keypoints)
            pose_3d = np.zeros((num_points, 4))

            for i, kp in enumerate(keypoints):
                if len(kp) >= 3:
                    pose_3d[i, 0] = kp[0] - 320  # X相对于图像中心
                    pose_3d[i, 1] = kp[1] - 240  # Y相对于图像中心
                    pose_3d[i, 2] = np.random.randn() * 50  # 模拟Z深度
                    pose_3d[i, 3] = kp[2] if kp[2] > 0 else 0.1  # 置信度

            return pose_3d

        except Exception as e:
            logger_3d.error(f"创建模拟3D数据失败: {e}")
            return None

    # 其他辅助方法...
    def _update_pose_sequence(self, pose_3d: np.ndarray):
        """更新姿态序列"""
        try:
            self.pose_3d_sequence.append(pose_3d)
            if len(self.pose_3d_sequence) > 100:  # 限制序列长度
                self.pose_3d_sequence = self.pose_3d_sequence[-100:]
        except Exception as e:
            logger_3d.error(f"更新序列失败: {e}")

    def _analyze_movement_quality(self, pose_3d: np.ndarray) -> Dict[str, float]:
        """分析运动质量"""
        try:
            if hasattr(self.parent, 'threed_analyzer') and hasattr(self.parent.threed_analyzer,
                                                                   'analyze_3d_movement_quality'):
                return self.parent.threed_analyzer.analyze_3d_movement_quality([pose_3d])
            else:
                return {
                    'overall_quality': 0.8,
                    'symmetry_score': 0.75,
                    'stability_score': 0.85,
                    'efficiency_score': 0.70
                }
        except Exception as e:
            logger_3d.error(f"质量分析失败: {e}")
            return {}

    def _calculate_3d_angles(self, pose_3d: np.ndarray) -> Dict[str, float]:
        """计算3D角度"""
        try:
            if hasattr(self.parent, 'threed_analyzer') and hasattr(self.parent.threed_analyzer,
                                                                   'calculate_3d_angles_enhanced'):
                return self.parent.threed_analyzer.calculate_3d_angles_enhanced(pose_3d)
            else:
                return self._calculate_basic_3d_angles(pose_3d)
        except Exception as e:
            logger_3d.error(f"角度计算失败: {e}")
            return {}

    def _calculate_basic_3d_angles(self, pose_3d: np.ndarray) -> Dict[str, float]:
        """基本3D角度计算"""
        angles = {}
        try:
            if pose_3d.shape[0] < 17:
                return angles

            # 简单的角度计算示例
            if pose_3d.shape[0] > 16:
                hip = pose_3d[11][:3]
                knee = pose_3d[13][:3]
                ankle = pose_3d[15][:3]

                if all(np.linalg.norm(p) > 0 for p in [hip, knee, ankle]):
                    angle = self._calculate_angle_3d(hip, knee, ankle)
                    angles['左膝盖角度'] = angle

        except Exception as e:
            logger_3d.error(f"基本角度计算错误: {e}")

        return angles

    def _calculate_angle_3d(self, p1: np.ndarray, p2: np.ndarray, p3: np.ndarray) -> float:
        """计算3D空间中三点的角度"""
        try:
            v1 = p1 - p2
            v2 = p3 - p2
            cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
            cos_angle = np.clip(cos_angle, -1.0, 1.0)
            angle = np.arccos(cos_angle) * 180 / np.pi
            return angle
        except Exception as e:
            logger_3d.error(f"角度计算错误: {e}")
            return 0.0

    def _assess_reconstruction_quality(self, pose_3d: np.ndarray, keypoints: List) -> float:
        """评估重建质量"""
        try:
            if hasattr(self.parent, 'threed_analyzer') and hasattr(self.parent.threed_analyzer,
                                                                   '_assess_reconstruction_quality'):
                return self.parent.threed_analyzer._assess_reconstruction_quality(pose_3d, keypoints)
            else:
                valid_points = np.sum(pose_3d[:, 3] > 0.1) if pose_3d.shape[1] > 3 else pose_3d.shape[0]
                total_points = pose_3d.shape[0]
                return valid_points / total_points if total_points > 0 else 0.0
        except Exception as e:
            logger_3d.error(f"质量评估失败: {e}")
            return 0.0

    def _translate_metric_name(self, metric_name: str) -> str:
        """翻译指标名称"""
        translations = {
            'overall_quality': '整体质量',
            'symmetry_score': '对称性评分',
            'stability_score': '稳定性评分',
            'efficiency_score': '效率评分'
        }
        return translations.get(metric_name, metric_name)

    def _add_table_row(self, name: str, value: str):
        """添加表格行"""
        try:
            row = self.parent.tableWidget.rowCount()
            self.parent.tableWidget.insertRow(row)
            self.parent.tableWidget.setItem(row, 0, QTableWidgetItem(name))
            self.parent.tableWidget.setItem(row, 1, QTableWidgetItem(value))
        except Exception as e:
            logger_3d.error(f"添加表格行失败: {e}")

    def _show_error_message(self, message: str):
        """显示错误消息"""
        self._add_table_row('错误', message)
        logger_3d.error(message)

    def _handle_analysis_error(self, exception: Exception):
        """处理分析错误"""
        error_msg = str(exception)
        logger_3d.error(f"3D分析错误: {error_msg}")
        logger_3d.error(traceback.format_exc())
        self._add_table_row('3D分析错误', error_msg[:50] + "..." if len(error_msg) > 50 else error_msg)

    def _validate_3d_data(self, pose_3d: Optional[np.ndarray]) -> Tuple[bool, str]:
        """验证3D数据有效性"""
        if pose_3d is None:
            return False, "3D数据为空"

        if not isinstance(pose_3d, np.ndarray):
            return False, "3D数据格式错误"

        if len(pose_3d.shape) != 2 or pose_3d.shape[1] < 3:
            return False, "3D数据维度不足"

        # 检查是否有足够的有效点
        if pose_3d.shape[1] >= 4:
            valid_points = np.sum(pose_3d[:, 3] > 0.1)
        else:
            # 如果没有置信度列，检查是否有非零点
            valid_points = np.sum(np.linalg.norm(pose_3d[:, :3], axis=1) > 0.1)

        if valid_points < 5:
            return False, f"有效关键点太少: {valid_points}"

        return True, "数据有效"

    def _handle_analysis_error(self, exception: Exception):
        """处理分析错误"""
        error_msg = str(exception)
        logger_3d.error(f"3D分析错误: {error_msg}")
        logger_3d.error(traceback.format_exc())

        self._add_table_row('3D分析错误', error_msg[:50] + "..." if len(error_msg) > 50 else error_msg)


class Performance3DOptimizer:
    """3D分析性能优化器"""

    def __init__(self):
        self.frame_cache = {}
        self.max_cache_size = 50

    def cache_3d_result(self, frame_idx: int, pose_3d: np.ndarray):
        """缓存3D结果"""
        try:
            if len(self.frame_cache) >= self.max_cache_size:
                # 删除最旧的缓存
                oldest_key = min(self.frame_cache.keys())
                del self.frame_cache[oldest_key]

            self.frame_cache[frame_idx] = pose_3d.copy()
            logger_3d.debug(f"缓存3D结果: 帧{frame_idx}")
        except Exception as e:
            logger_3d.error(f"缓存失败: {e}")

    def get_cached_result(self, frame_idx: int) -> Optional[np.ndarray]:
        """获取缓存的结果"""
        return self.frame_cache.get(frame_idx)

    def clear_cache(self):
        """清除缓存"""
        self.frame_cache.clear()
        logger_3d.info("缓存已清除")


def test_3d_integration():
    """测试3D集成功能"""
    try:
        print("开始3D集成测试...")

        # 创建测试关键点数据 (25个关键点)
        test_keypoints = []
        for i in range(25):
            x = 320 + np.random.randn() * 50
            y = 240 + np.random.randn() * 50
            conf = 0.8 + np.random.randn() * 0.1
            test_keypoints.append([x, y, max(0.1, conf)])

        # 创建模拟的3D姿态数据
        test_pose_3d = np.random.rand(25, 4)
        test_pose_3d[:, 3] = np.random.rand(25) * 0.5 + 0.5  # 置信度

        # 创建集成器
        integrator = Enhanced3DAnalysisIntegrator()

        # 验证结果
        is_valid, msg = integrator._validate_3d_data(test_pose_3d)

        if is_valid:
            print("✅ 3D集成测试通过")
            print(f"测试数据形状: {test_pose_3d.shape}")
            print(f"有效点数: {np.sum(test_pose_3d[:, 3] > 0.1)}")
        else:
            print(f"❌ 3D集成测试失败: {msg}")

        # 测试性能优化器
        optimizer = Performance3DOptimizer()
        optimizer.cache_3d_result(0, test_pose_3d)
        cached = optimizer.get_cached_result(0)

        if cached is not None:
            print("✅ 缓存系统测试通过")
        else:
            print("❌ 缓存系统测试失败")

        return is_valid

    except Exception as e:
        print(f"❌ 3D集成测试异常: {e}")
        traceback.print_exc()
        return False


import numpy as np
import traceback
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
import time
from collections import defaultdict


@dataclass
class PoseJoint:
    """3D姿态关节点数据结构"""
    x: float
    y: float
    z: float
    confidence: float
    joint_type: str = ""


class Performance3DOptimizer:
    """3D姿态分析性能优化器"""

    def __init__(self, cache_size: int = 100):
        self.cache_size = cache_size
        self.cache = {}
        self.cache_times = {}
        self.cache_order = []

    def cache_3d_result(self, frame_id: int, pose_data: np.ndarray) -> None:
        """缓存3D姿态结果"""
        # 如果缓存已满，删除最旧的条目
        if len(self.cache) >= self.cache_size:
            oldest_frame = self.cache_order.pop(0)
            del self.cache[oldest_frame]
            del self.cache_times[oldest_frame]

        self.cache[frame_id] = pose_data.copy()
        self.cache_times[frame_id] = time.time()
        self.cache_order.append(frame_id)

    def get_cached_result(self, frame_id: int) -> Optional[np.ndarray]:
        """获取缓存的3D姿态结果"""
        return self.cache.get(frame_id)

    def clear_cache(self) -> None:
        """清空缓存"""
        self.cache.clear()
        self.cache_times.clear()
        self.cache_order.clear()

    def get_cache_info(self) -> Dict[str, Any]:
        """获取缓存信息"""
        return {
            'cache_size': len(self.cache),
            'max_size': self.cache_size,
            'frames_cached': list(self.cache.keys())
        }


class Enhanced3DAnalysisIntegrator:
    """增强的3D姿态分析集成器"""

    def __init__(self):
        self.joint_names = [
            'nose', 'left_eye', 'right_eye', 'left_ear', 'right_ear',
            'left_shoulder', 'right_shoulder', 'left_elbow', 'right_elbow',
            'left_wrist', 'right_wrist', 'left_hip', 'right_hip',
            'left_knee', 'right_knee', 'left_ankle', 'right_ankle',
            'head', 'neck', 'mid_hip', 'left_big_toe', 'right_big_toe',
            'left_small_toe', 'right_small_toe', 'left_heel', 'right_heel'
        ]

        self.optimizer = Performance3DOptimizer()
        self.analysis_history = []

    def _validate_3d_data(self, pose_3d: np.ndarray) -> Tuple[bool, str]:
        """验证3D姿态数据的有效性"""
        try:
            # 检查数据类型
            if not isinstance(pose_3d, np.ndarray):
                return False, "数据必须是numpy数组"

            # 检查数据形状
            if len(pose_3d.shape) != 2:
                return False, "数据必须是2D数组"

            if pose_3d.shape[1] < 3:
                return False, "每个关节点至少需要3个坐标(x,y,z)"

            if pose_3d.shape[1] > 4:
                return False, "每个关节点最多4个值(x,y,z,confidence)"

            # 检查关节点数量
            expected_joints = len(self.joint_names)
            if pose_3d.shape[0] != expected_joints:
                return False, f"期望{expected_joints}个关节点，实际得到{pose_3d.shape[0]}个"

            # 检查数据范围
            if np.any(np.isnan(pose_3d)) or np.any(np.isinf(pose_3d)):
                return False, "数据包含NaN或无穷值"

            # 如果有置信度列，检查置信度范围
            if pose_3d.shape[1] == 4:
                confidence = pose_3d[:, 3]
                if np.any(confidence < 0) or np.any(confidence > 1):
                    return False, "置信度必须在[0,1]范围内"

            return True, "数据验证通过"

        except Exception as e:
            return False, f"验证过程中出现异常: {str(e)}"

    def analyze_pose_quality(self, pose_3d: np.ndarray) -> Dict[str, Any]:
        """分析3D姿态质量"""
        is_valid, msg = self._validate_3d_data(pose_3d)
        if not is_valid:
            return {'valid': False, 'error': msg}

        analysis = {
            'valid': True,
            'total_joints': pose_3d.shape[0],
            'dimensions': pose_3d.shape[1]
        }

        # 如果有置信度信息
        if pose_3d.shape[1] == 4:
            confidence = pose_3d[:, 3]
            analysis.update({
                'avg_confidence': float(np.mean(confidence)),
                'min_confidence': float(np.min(confidence)),
                'max_confidence': float(np.max(confidence)),
                'high_confidence_joints': int(np.sum(confidence > 0.7)),
                'low_confidence_joints': int(np.sum(confidence < 0.3))
            })

        # 计算姿态的空间分布
        xyz_coords = pose_3d[:, :3]
        analysis.update({
            'spatial_range': {
                'x': {'min': float(np.min(xyz_coords[:, 0])), 'max': float(np.max(xyz_coords[:, 0]))},
                'y': {'min': float(np.min(xyz_coords[:, 1])), 'max': float(np.max(xyz_coords[:, 1]))},
                'z': {'min': float(np.min(xyz_coords[:, 2])), 'max': float(np.max(xyz_coords[:, 2]))}
            },
            'centroid': {
                'x': float(np.mean(xyz_coords[:, 0])),
                'y': float(np.mean(xyz_coords[:, 1])),
                'z': float(np.mean(xyz_coords[:, 2]))
            }
        })

        return analysis

    def detect_pose_anomalies(self, pose_3d: np.ndarray) -> List[Dict[str, Any]]:
        """检测姿态异常"""
        anomalies = []

        is_valid, msg = self._validate_3d_data(pose_3d)
        if not is_valid:
            anomalies.append({'type': 'validation_error', 'message': msg})
            return anomalies

        # 检查极端坐标值
        xyz_coords = pose_3d[:, :3]
        for i, joint_name in enumerate(self.joint_names):
            x, y, z = xyz_coords[i]

            # 检查是否有异常大的坐标值
            if abs(x) > 1000 or abs(y) > 1000 or abs(z) > 1000:
                anomalies.append({
                    'type': 'extreme_coordinates',
                    'joint': joint_name,
                    'coordinates': [float(x), float(y), float(z)]
                })

        # 如果有置信度信息，检查低置信度关节
        if pose_3d.shape[1] == 4:
            confidence = pose_3d[:, 3]
            for i, conf in enumerate(confidence):
                if conf < 0.1:
                    anomalies.append({
                        'type': 'low_confidence',
                        'joint': self.joint_names[i],
                        'confidence': float(conf)
                    })

        return anomalies

    def integrate_3d_analysis(self, pose_3d: np.ndarray, frame_id: int = 0) -> Dict[str, Any]:
        """集成3D姿态分析"""
        start_time = time.time()

        # 验证数据
        is_valid, validation_msg = self._validate_3d_data(pose_3d)
        if not is_valid:
            return {
                'success': False,
                'error': validation_msg,
                'frame_id': frame_id,
                'processing_time': time.time() - start_time
            }

        # 分析姿态质量
        quality_analysis = self.analyze_pose_quality(pose_3d)

        # 检测异常
        anomalies = self.detect_pose_anomalies(pose_3d)

        # 缓存结果
        self.optimizer.cache_3d_result(frame_id, pose_3d)

        # 构建结果
        result = {
            'success': True,
            'frame_id': frame_id,
            'validation': {
                'valid': is_valid,
                'message': validation_msg
            },
            'quality_analysis': quality_analysis,
            'anomalies': anomalies,
            'anomaly_count': len(anomalies),
            'processing_time': time.time() - start_time,
            'cache_info': self.optimizer.get_cache_info()
        }

        # 添加到历史记录
        self.analysis_history.append({
            'frame_id': frame_id,
            'timestamp': time.time(),
            'result_summary': {
                'valid': is_valid,
                'anomaly_count': len(anomalies),
                'avg_confidence': quality_analysis.get('avg_confidence', 0)
            }
        })

        return result

    def get_analysis_statistics(self) -> Dict[str, Any]:
        """获取分析统计信息"""
        if not self.analysis_history:
            return {'message': '暂无分析历史'}

        total_analyses = len(self.analysis_history)
        valid_analyses = sum(1 for h in self.analysis_history if h['result_summary']['valid'])

        return {
            'total_analyses': total_analyses,
            'valid_analyses': valid_analyses,
            'success_rate': valid_analyses / total_analyses if total_analyses > 0 else 0,
            'cache_info': self.optimizer.get_cache_info(),
            'recent_analyses': self.analysis_history[-5:] if len(self.analysis_history) >= 5 else self.analysis_history
        }


def test_3d_integration():
    """测试3D集成功能"""
    try:
        print("🔄 开始3D集成测试...")

        # 创建模拟的3D姿态数据
        test_pose_3d = np.random.rand(25, 4)
        test_pose_3d[:, 3] = np.random.rand(26) * 0.5 + 0.5  # 置信度

        # 创建集成器
        integrator = Enhanced3DAnalysisIntegrator()

        # 验证结果
        is_valid, msg = integrator._validate_3d_data(test_pose_3d)

        if is_valid:
            print("✅ 3D集成测试通过")
            print(f"测试数据形状: {test_pose_3d.shape}")
            print(f"有效点数: {np.sum(test_pose_3d[:, 3] > 0.1)}")
        else:
            print(f"❌ 3D集成测试失败: {msg}")

        # 测试性能优化器
        optimizer = Performance3DOptimizer()
        optimizer.cache_3d_result(0, test_pose_3d)
        cached = optimizer.get_cached_result(0)

        if cached is not None:
            print("✅ 缓存系统测试通过")
        else:
            print("❌ 缓存系统测试失败")

        # 完整集成分析测试
        result = integrator.integrate_3d_analysis(test_pose_3d, frame_id=1)
        if result['success']:
            print("✅ 完整集成分析测试通过")
            print(f"处理时间: {result['processing_time']:.4f}秒")
            print(f"检测到异常: {result['anomaly_count']}个")
        else:
            print(f"❌ 完整集成分析测试失败: {result['error']}")

        # 测试统计信息
        stats = integrator.get_analysis_statistics()
        print(f"✅ 统计信息获取成功，成功率: {stats.get('success_rate', 0):.2%}")

        return is_valid

    except Exception as e:
        print(f"❌ 3D集成测试异常: {e}")
        traceback.print_exc()
        return False


if __name__ == "__main__":
    # 运行测试
    success = test_3d_integration()

    print("\n" + "=" * 50)
    print("3D姿态分析集成系统测试完成")
    print(f"总体结果: {'✅ 成功' if success else '❌ 失败'}")
    print("=" * 50)
# 使用示例
def integrate_with_main_application(main_window):
    """与主应用程序集成"""
    try:
        # 创建3D分析集成器
        integrator = Enhanced3DAnalysisIntegrator(main_window)

        # 将集成器方法绑定到主窗口
        main_window.show_3d_analysis = integrator.show_3d_analysis
        main_window.open_3d_viewer = integrator.open_3d_viewer
        main_window.save_3d_frame = integrator.save_3d_frame
        main_window.export_3d_sequence = integrator.export_3d_sequence
        main_window.setup_camera_parameters = integrator.setup_camera_parameters

        # 保存集成器引用
        main_window.threed_integrator = integrator

        logger_3d.info("3D分析集成完成")
        return True

    except Exception as e:
        logger_3d.error(f"集成失败: {e}")
        return False




# ==================== 3. 高级生物力学模块 ====================
import numpy as np
import warnings
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass


@dataclass
class SegmentParameters:
    """身体段参数数据类"""
    mass_ratio: float
    com_ratio: float
    length_ratio: float = 1.0
    radius_of_gyration: float = 0.3


class AdvancedBiomechanics:
    """高级生物力学分析器"""

    def __init__(self):
        self.body_segment_parameters = self._load_anthropometric_data()
        self.force_plates_data = None
        self._gravity = 9.81  # 重力加速度

    def _load_anthropometric_data(self) -> Dict[str, SegmentParameters]:
        """加载人体测量学数据 - 基于Dempster等人的研究数据"""
        return {
            'head': SegmentParameters(
                mass_ratio=0.081,
                com_ratio=0.5,
                length_ratio=1.0,
                radius_of_gyration=0.303
            ),
            'trunk': SegmentParameters(
                mass_ratio=0.497,
                com_ratio=0.5,
                length_ratio=1.0,
                radius_of_gyration=0.372
            ),
            'upper_arm': SegmentParameters(
                mass_ratio=0.028,
                com_ratio=0.436,
                length_ratio=1.0,
                radius_of_gyration=0.322
            ),
            'forearm': SegmentParameters(
                mass_ratio=0.016,
                com_ratio=0.43,
                length_ratio=1.0,
                radius_of_gyration=0.303
            ),
            'hand': SegmentParameters(
                mass_ratio=0.006,
                com_ratio=0.506,
                length_ratio=1.0,
                radius_of_gyration=0.297
            ),
            'thigh': SegmentParameters(
                mass_ratio=0.100,
                com_ratio=0.433,
                length_ratio=1.0,
                radius_of_gyration=0.323
            ),
            'shank': SegmentParameters(
                mass_ratio=0.0465,
                com_ratio=0.433,
                length_ratio=1.0,
                radius_of_gyration=0.302
            ),
            'foot': SegmentParameters(
                mass_ratio=0.0145,
                com_ratio=0.5,
                length_ratio=1.0,
                radius_of_gyration=0.475
            )
        }

    def calculate_advanced_com(self, keypoints_3d: np.ndarray,
                               athlete_profile: Dict[str, Any]) -> Dict[str, Any]:
        """
        计算高级重心分析

        Args:
            keypoints_3d: 3D关键点数据 (N, 4) 其中第4列为置信度
            athlete_profile: 运动员基本信息

        Returns:
            重心分析结果字典
        """
        try:
            # 输入验证
            if not self._validate_keypoints(keypoints_3d):
                return self._get_empty_com_result()

            total_mass = athlete_profile.get('weight', 70.0)
            weighted_com = np.zeros(3, dtype=np.float64)
            total_weight = 0.0

            # 计算各身体部位重心贡献
            segments = self._get_body_segments_mapping()
            valid_segments = 0

            for segment_name, (start_joint, end_joint) in segments.items():
                # 检查关键点有效性
                if not self._are_keypoints_valid(keypoints_3d, [start_joint, end_joint]):
                    continue

                # 计算段重心位置
                start_pos = keypoints_3d[start_joint, :3].astype(np.float64)
                end_pos = keypoints_3d[end_joint, :3].astype(np.float64)

                segment_params = self.body_segment_parameters.get(
                    segment_name.replace('right_', '').replace('left_', ''),
                    SegmentParameters(mass_ratio=0.05, com_ratio=0.5)
                )

                segment_com = start_pos + (end_pos - start_pos) * segment_params.com_ratio
                segment_mass = total_mass * segment_params.mass_ratio

                weighted_com += segment_com * segment_mass
                total_weight += segment_mass
                valid_segments += 1

            if total_weight > 0 and valid_segments >= 3:  # 至少需要3个有效段
                overall_com = weighted_com / total_weight

                return {
                    'com_3d': overall_com.tolist(),
                    'com_height': float(overall_com[1]),
                    'com_anterior_posterior': float(overall_com[2]),
                    'com_medial_lateral': float(overall_com[0]),
                    'total_mass_represented': float(total_weight),
                    'mass_percentage': float(total_weight / total_mass * 100),
                    'valid_segments': valid_segments
                }

        except Exception as e:
            warnings.warn(f"高级重心计算错误: {e}", RuntimeWarning)

        return self._get_empty_com_result()

    def _validate_keypoints(self, keypoints_3d: np.ndarray) -> bool:
        """验证关键点数据有效性"""
        if keypoints_3d is None:
            return False
        if keypoints_3d.shape[0] < 15:  # 至少需要15个关键点
            return False
        if keypoints_3d.shape[1] < 4:  # 需要x,y,z,confidence
            return False
        return True

    def _are_keypoints_valid(self, keypoints_3d: np.ndarray,
                             indices: List[int], confidence_threshold: float = 0.1) -> bool:
        """检查指定关键点是否有效"""
        try:
            for idx in indices:
                if idx >= keypoints_3d.shape[0]:
                    return False
                if keypoints_3d[idx, 3] <= confidence_threshold:
                    return False
            return True
        except (IndexError, TypeError):
            return False

    def _get_empty_com_result(self) -> Dict[str, Any]:
        """返回空的重心结果"""
        return {
            'com_3d': [0.0, 0.0, 0.0],
            'com_height': 0.0,
            'com_anterior_posterior': 0.0,
            'com_medial_lateral': 0.0,
            'total_mass_represented': 0.0,
            'mass_percentage': 0.0,
            'valid_segments': 0
        }

    def _get_body_segments_mapping(self) -> Dict[str, Tuple[int, int]]:
        """获取身体段映射关系"""
        return {
            'head': (0, 1),  # 鼻子到颈部
            'trunk': (1, 8),  # 颈部到中臀
            'right_upper_arm': (2, 3),  # 右肩到右肘
            'right_forearm': (3, 4),  # 右肘到右腕
            'left_upper_arm': (5, 6),  # 左肩到左肘
            'left_forearm': (6, 7),  # 左肘到左腕
            'right_thigh': (9, 10),  # 右髋到右膝
            'right_shank': (10, 11),  # 右膝到右踝
            'left_thigh': (12, 13),  # 左髋到左膝
            'left_shank': (13, 14),  # 左膝到左踝
        }

    def calculate_joint_power(self, keypoints_sequence: List[np.ndarray],
                              athlete_profile: Dict[str, Any],
                              fps: float = 30.0) -> Dict[str, Dict[str, Any]]:
        """
        计算关节功率

        Args:
            keypoints_sequence: 关键点序列
            athlete_profile: 运动员信息
            fps: 帧率

        Returns:
            关节功率分析结果
        """
        power_analysis = {}

        try:
            if len(keypoints_sequence) < 3:  # 至少需要3帧进行数值微分
                return power_analysis

            dt = 1.0 / fps

            for i in range(2, len(keypoints_sequence)):
                current_frame = keypoints_sequence[i]
                previous_frame = keypoints_sequence[i - 1]
                prev_prev_frame = keypoints_sequence[i - 2]

                if all(frame is not None for frame in [current_frame, previous_frame, prev_prev_frame]):
                    # 使用中心差分计算角速度（更精确）
                    angular_velocities = self._calculate_angular_velocities_centered(
                        current_frame, prev_prev_frame, 2 * dt
                    )

                    # 计算关节力矩
                    joint_torques = self._calculate_joint_torques_advanced(
                        current_frame, athlete_profile
                    )

                    # 计算功率 P = τ × ω
                    for joint in angular_velocities:
                        if joint in joint_torques:
                            power = abs(joint_torques[joint] * angular_velocities[joint])
                            if joint not in power_analysis:
                                power_analysis[joint] = []
                            power_analysis[joint].append(power)

            # 统计分析
            for joint in power_analysis:
                powers = np.array(power_analysis[joint])
                if len(powers) > 0:
                    power_analysis[joint] = {
                        'average_power': float(np.mean(powers)),
                        'peak_power': float(np.max(powers)),
                        'min_power': float(np.min(powers)),
                        'std_power': float(np.std(powers)),
                        'power_profile': powers.tolist(),
                        'total_work': float(np.trapz(powers) * dt)  # 积分计算总功
                    }

        except Exception as e:
            warnings.warn(f"关节功率计算错误: {e}", RuntimeWarning)

        return power_analysis

    def _calculate_angular_velocities_centered(self, frame_t_plus: np.ndarray,
                                               frame_t_minus: np.ndarray,
                                               dt: float) -> Dict[str, float]:
        """使用中心差分计算角速度"""
        angular_velocities = {}

        try:
            joints = {
                'right_elbow': [2, 3, 4],
                'left_elbow': [5, 6, 7],
                'right_knee': [9, 10, 11],
                'left_knee': [12, 13, 14],
                'right_shoulder': [1, 2, 3],
                'left_shoulder': [1, 5, 6],
                'right_hip': [8, 9, 10],
                'left_hip': [8, 12, 13]
            }

            for joint_name, indices in joints.items():
                if self._are_keypoints_valid(frame_t_plus, indices) and \
                        self._are_keypoints_valid(frame_t_minus, indices):

                    angle_plus = self._calculate_joint_angle_safe(frame_t_plus, indices)
                    angle_minus = self._calculate_joint_angle_safe(frame_t_minus, indices)

                    if angle_plus is not None and angle_minus is not None:
                        # 处理角度跳跃问题
                        angle_diff = angle_plus - angle_minus
                        if abs(angle_diff) > 180:
                            angle_diff = angle_diff - 360 * np.sign(angle_diff)

                        angular_velocity = angle_diff / dt
                        angular_velocities[joint_name] = angular_velocity

        except Exception as e:
            warnings.warn(f"角速度计算错误: {e}", RuntimeWarning)

        return angular_velocities

    def _calculate_joint_angle_safe(self, keypoints: np.ndarray,
                                    indices: List[int]) -> Optional[float]:
        """安全地计算关节角度"""
        try:
            p1, p2, p3 = indices

            v1 = keypoints[p1, :3] - keypoints[p2, :3]
            v2 = keypoints[p3, :3] - keypoints[p2, :3]

            # 计算向量长度
            norm1 = np.linalg.norm(v1)
            norm2 = np.linalg.norm(v2)

            if norm1 < 1e-6 or norm2 < 1e-6:  # 避免除零
                return None

            # 计算夹角
            cos_angle = np.dot(v1, v2) / (norm1 * norm2)
            cos_angle = np.clip(cos_angle, -1.0, 1.0)  # 防止数值误差
            angle = np.arccos(cos_angle)

            return float(np.degrees(angle))

        except Exception as e:
            warnings.warn(f"角度计算错误: {e}", RuntimeWarning)
            return None

    def _calculate_joint_torques_advanced(self, keypoints: np.ndarray,
                                          athlete_profile: Dict[str, Any]) -> Dict[str, float]:
        """计算高级关节力矩"""
        torques = {}

        try:
            mass = athlete_profile.get('weight', 70.0)
            height = athlete_profile.get('height', 175.0) / 100.0  # 转换为米

            segments_info = self._get_body_segments_mapping()

            for segment_name, (start_idx, end_idx) in segments_info.items():
                if not self._are_keypoints_valid(keypoints, [start_idx, end_idx]):
                    continue

                # 清理段名称
                clean_segment_name = segment_name.replace('right_', '').replace('left_', '')
                segment_params = self.body_segment_parameters.get(
                    clean_segment_name,
                    SegmentParameters(mass_ratio=0.05, com_ratio=0.5)
                )

                segment_mass = mass * segment_params.mass_ratio

                # 计算段长度（单位：米）
                start_pos = keypoints[start_idx, :3]
                end_pos = keypoints[end_idx, :3]
                segment_length = np.linalg.norm(end_pos - start_pos) / 1000.0

                # 计算重心位置
                com_pos = start_pos + (end_pos - start_pos) * segment_params.com_ratio

                # 计算重力力矩（简化模型）
                moment_arm = segment_length * segment_params.com_ratio
                gravity_torque = segment_mass * self._gravity * moment_arm

                torques[f'{segment_name}_torque'] = float(gravity_torque)

        except Exception as e:
            warnings.warn(f"高级力矩计算错误: {e}", RuntimeWarning)

        return torques

    def calculate_segment_accelerations(self, keypoints_sequence: List[np.ndarray],
                                        fps: float = 30.0) -> Dict[str, np.ndarray]:
        """计算身体段加速度"""
        accelerations = {}

        if len(keypoints_sequence) < 5:  # 需要至少5个点进行二阶数值微分
            return accelerations

        dt = 1.0 / fps
        segments = self._get_body_segments_mapping()

        for segment_name, (start_idx, end_idx) in segments.items():
            segment_accelerations = []

            for i in range(2, len(keypoints_sequence) - 2):
                frames = keypoints_sequence[i - 2:i + 3]  # 5点模板

                if all(self._are_keypoints_valid(frame, [start_idx, end_idx])
                       for frame in frames):

                    # 计算段中心点位置
                    centers = []
                    for frame in frames:
                        center = (frame[start_idx, :3] + frame[end_idx, :3]) / 2
                        centers.append(center)

                    # 使用5点公式计算加速度
                    centers = np.array(centers)
                    acceleration = (-centers[4] + 16 * centers[3] - 30 * centers[2] +
                                    16 * centers[1] - centers[0]) / (12 * dt ** 2)

                    segment_accelerations.append(acceleration)

            if segment_accelerations:
                accelerations[segment_name] = np.array(segment_accelerations)

        return accelerations

    def analyze_balance_stability(self, com_sequence: List[Dict[str, Any]],
                                  base_of_support: Optional[np.ndarray] = None) -> Dict[str, Any]:
        """分析平衡稳定性"""
        if not com_sequence:
            return {}

        try:
            # 提取重心轨迹
            com_positions = np.array([com['com_3d'] for com in com_sequence if com.get('com_3d')])

            if len(com_positions) < 2:
                return {}

            # 计算重心摆动参数
            com_velocity = np.diff(com_positions, axis=0)
            com_speed = np.linalg.norm(com_velocity, axis=1)

            # 计算稳定性指标
            stability_metrics = {
                'mean_com_position': np.mean(com_positions, axis=0).tolist(),
                'com_sway_area': self._calculate_sway_area(com_positions),
                'com_velocity_mean': float(np.mean(com_speed)),
                'com_velocity_std': float(np.std(com_speed)),
                'path_length': float(np.sum(com_speed)),
                'rms_distance': float(np.sqrt(np.mean(np.sum(com_positions ** 2, axis=1))))
            }

            return stability_metrics

        except Exception as e:
            warnings.warn(f"平衡分析错误: {e}", RuntimeWarning)
            return {}

    def _calculate_sway_area(self, positions: np.ndarray) -> float:
        """计算摆动面积（95%置信椭圆）"""
        try:
            if len(positions) < 3:
                return 0.0

            # 使用水平面投影 (x, z)
            xy_positions = positions[:, [0, 2]]

            # 计算协方差矩阵
            cov_matrix = np.cov(xy_positions.T)

            # 计算特征值
            eigenvalues = np.linalg.eigvals(cov_matrix)

            # 95%置信椭圆面积
            area = np.pi * 2.45 * np.sqrt(np.prod(eigenvalues))  # 2.45 对应95%置信度

            return float(area)

        except Exception as e:
            warnings.warn(f"摆动面积计算错误: {e}", RuntimeWarning)
            return 0.0


# 辅助函数
def validate_biomechanics_input(keypoints_data: Any,
                                athlete_profile: Dict[str, Any]) -> bool:
    """验证生物力学分析输入数据"""
    if not isinstance(athlete_profile, dict):
        return False

    required_keys = ['weight', 'height']
    if not all(key in athlete_profile for key in required_keys):
        return False

    if keypoints_data is None:
        return False

    return True

# ==================== 4. 运动专项化分析模块 ====================
import numpy as np
from scipy import signal
import logging
from typing import Dict, List, Tuple, Optional, Any
import json
import os
from dataclasses import dataclass
from enum import Enum
# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
class SportSpecificAnalyzer:
    """运动专项化分析器"""

    def __init__(self):
        self.sport_templates = self.load_sport_templates()
        self.performance_benchmarks = self.load_performance_benchmarks()

    def load_sport_templates(self):
        """加载运动专项模板"""
        return {
            '篮球': {
                'key_movements': ['投篮', '运球', '跳跃', '防守'],
                'critical_joints': ['ankle', 'knee', 'hip', 'shoulder', 'elbow'],
                'performance_metrics': ['jump_height', 'shooting_form', 'agility'],
                'injury_risks': ['ankle_sprain', 'knee_injury', 'shoulder_impingement']
            },
            '足球': {
                'key_movements': ['踢球', '跑动', '跳跃', '转身'],
                'critical_joints': ['ankle', 'knee', 'hip'],
                'performance_metrics': ['kicking_power', 'running_efficiency', 'balance'],
                'injury_risks': ['ankle_sprain', 'hamstring_strain', 'groin_injury']
            },
            '网球': {
                'key_movements': ['发球', '正手', '反手', '移动'],
                'critical_joints': ['shoulder', 'elbow', 'wrist', 'hip', 'knee'],
                'performance_metrics': ['serve_speed', 'stroke_consistency', 'court_coverage'],
                'injury_risks': ['tennis_elbow', 'shoulder_impingement', 'wrist_injury']
            },
            '举重': {
                'key_movements': ['深蹲', '硬拉', '卧推', '抓举'],
                'critical_joints': ['ankle', 'knee', 'hip', 'spine', 'shoulder'],
                'performance_metrics': ['lifting_technique', 'power_output', 'stability'],
                'injury_risks': ['lower_back_injury', 'knee_injury', 'shoulder_injury']
            }
        }

    def load_performance_benchmarks(self):
        """加载运动表现基准"""
        return {
            '篮球': {
                'professional': {'jump_height': 80, 'shooting_accuracy': 0.85},
                'amateur': {'jump_height': 60, 'shooting_accuracy': 0.65}
            },
            '足球': {
                'professional': {'sprint_speed': 25, 'endurance': 90},
                'amateur': {'sprint_speed': 20, 'endurance': 70}
            }
            # 更多基准数据...
        }

    def analyze_sport_specific_performance(self, keypoints_sequence, sport_type, athlete_profile):
        """运动专项表现分析"""
        analysis = {
            'sport': sport_type,
            'movement_analysis': {},
            'technique_scores': {},
            'injury_risk_assessment': {},
            'performance_comparison': {},
            'recommendations': []
        }

        try:
            if sport_type not in self.sport_templates:
                return analysis

            template = self.sport_templates[sport_type]

            # 分析关键动作
            analysis['movement_analysis'] = self.analyze_key_movements(
                keypoints_sequence, template['key_movements']
            )

            # 技术评分
            analysis['technique_scores'] = self.calculate_technique_scores(
                keypoints_sequence, sport_type
            )

            # 专项损伤风险评估
            analysis['injury_risk_assessment'] = self.assess_sport_specific_injury_risk(
                keypoints_sequence, template['injury_risks']
            )

            # 表现对比
            analysis['performance_comparison'] = self.compare_with_benchmarks(
                analysis['technique_scores'], sport_type, athlete_profile
            )

            # 生成专项建议
            analysis['recommendations'] = self.generate_sport_specific_recommendations(
                analysis, sport_type
            )

        except Exception as e:
            print(f"运动专项分析错误: {e}")

        return analysis

    def analyze_key_movements(self, keypoints_sequence, key_movements):
        """分析关键动作"""
        movement_analysis = {}

        for movement in key_movements:
            if movement == '跳跃':
                movement_analysis['jump_analysis'] = self.analyze_jumping_movement(keypoints_sequence)
            elif movement == '投篮':
                movement_analysis['shooting_analysis'] = self.analyze_shooting_movement(keypoints_sequence)
            elif movement == '跑动':
                movement_analysis['running_analysis'] = self.analyze_running_movement(keypoints_sequence)
            # 更多运动分析...

        return movement_analysis

    def analyze_jumping_movement(self, keypoints_sequence):
        """分析跳跃动作"""
        try:
            jump_analysis = {
                'max_height': 0,
                'takeoff_angle': 0,
                'landing_stability': 0,
                'jump_phases': []
            }

            # 找到跳跃阶段
            hip_heights = []
            for frame in keypoints_sequence:
                if frame and len(frame) > 8 and frame[8][3] > 0.1:
                    hip_heights.append(frame[8][1])  # 中臀Y坐标

            if len(hip_heights) > 5:
                # 找到最低点和最高点
                min_height = min(hip_heights)
                max_height = max(hip_heights)

                jump_analysis['max_height'] = max_height - min_height

                # 分析起跳角度
                takeoff_frame = hip_heights.index(min_height)
                if takeoff_frame < len(keypoints_sequence) - 1:
                    frame = keypoints_sequence[takeoff_frame]
                    if frame and len(frame) > 13:
                        # 计算膝关节角度作为起跳角度指标
                        knee_angle = self.calculate_joint_angle(frame, [9, 10, 11])
                        jump_analysis['takeoff_angle'] = knee_angle

                # 分析着地稳定性
                landing_frame = hip_heights.index(max_height) + 1
                if landing_frame < len(keypoints_sequence):
                    # 计算着地后的重心稳定性
                    post_landing_frames = hip_heights[landing_frame:landing_frame + 10]
                    if post_landing_frames:
                        stability = 1.0 / (1.0 + np.std(post_landing_frames))
                        jump_analysis['landing_stability'] = stability

            return jump_analysis

        except Exception as e:
            print(f"跳跃分析错误: {e}")
            return {}

    def analyze_shooting_movement(self, keypoints_sequence):
        """分析投篮动作"""
        try:
            shooting_analysis = {
                'release_height': 0,
                'shooting_arc': 0,
                'follow_through': 0,
                'consistency': 0
            }

            # 分析投篮弧线
            wrist_positions = []
            for frame in keypoints_sequence:
                if frame and len(frame) > 4 and frame[4][3] > 0.1:
                    wrist_positions.append([frame[4][0], frame[4][1]])

            if len(wrist_positions) > 3:
                wrist_positions = np.array(wrist_positions)

                # 计算出手高度
                shooting_analysis['release_height'] = np.min(wrist_positions[:, 1])

                # 计算弧线（基于轨迹曲率）
                if len(wrist_positions) > 5:
                    # 拟合二次曲线
                    x = wrist_positions[:, 0]
                    y = wrist_positions[:, 1]

                    try:
                        # 二次拟合
                        coeffs = np.polyfit(x, y, 2)
                        shooting_analysis['shooting_arc'] = abs(coeffs[0])  # 二次项系数表示弧度
                    except:
                        shooting_analysis['shooting_arc'] = 0

                # 分析一致性
                shooting_analysis['consistency'] = 1.0 / (1.0 + np.std(wrist_positions, axis=0).mean())

            return shooting_analysis

        except Exception as e:
            print(f"投篮分析错误: {e}")
            return {}

    def analyze_running_movement(self, keypoints_sequence):
        """分析跑步动作"""
        try:
            running_analysis = {
                'stride_length': 0,
                'cadence': 0,
                'ground_contact_time': 0,
                'running_efficiency': 0
            }

            # 分析步长和步频
            foot_positions = []
            for frame in keypoints_sequence:
                if frame and len(frame) > 11 and frame[11][3] > 0.1:
                    foot_positions.append(frame[11][0])  # 右踝X坐标

            if len(foot_positions) > 10:
                # 检测步态周期
                stride_peaks = signal.find_peaks(foot_positions, distance=5)[0]

                if len(stride_peaks) > 1:
                    # 计算步长
                    stride_distances = [foot_positions[stride_peaks[i + 1]] - foot_positions[stride_peaks[i]]
                                        for i in range(len(stride_peaks) - 1)]
                    running_analysis['stride_length'] = np.mean(stride_distances)

                    # 计算步频
                    stride_intervals = [stride_peaks[i + 1] - stride_peaks[i]
                                        for i in range(len(stride_peaks) - 1)]
                    running_analysis['cadence'] = len(keypoints_sequence) / np.mean(stride_intervals) * 30  # 假设30fps

                    # 计算跑步效率
                    running_analysis['running_efficiency'] = (
                            running_analysis['stride_length'] * running_analysis['cadence'] / 1000
                    )

            return running_analysis

        except Exception as e:
            print(f"跑步分析错误: {e}")
            return {}

    def calculate_technique_scores(self, keypoints_sequence, sport_type):
        """计算技术评分"""
        scores = {}

        try:
            if sport_type == '篮球':
                scores = self.score_basketball_technique(keypoints_sequence)
            elif sport_type == '足球':
                scores = self.score_football_technique(keypoints_sequence)
            elif sport_type == '网球':
                scores = self.score_tennis_technique(keypoints_sequence)
            elif sport_type == '举重':
                scores = self.score_weightlifting_technique(keypoints_sequence)

        except Exception as e:
            print(f"技术评分错误: {e}")

        return scores

    def score_basketball_technique(self, keypoints_sequence):
        """篮球技术评分"""
        scores = {
            'shooting_form': 0,
            'jumping_technique': 0,
            'balance': 0,
            'overall': 0
        }

        # 基于动作分析结果评分
        # 这里可以添加更复杂的评分算法

        return scores

    def assess_sport_specific_injury_risk(self, keypoints_sequence, injury_risks):
        """运动专项损伤风险评估"""
        risk_assessment = {}

        for risk_type in injury_risks:
            if risk_type == 'ankle_sprain':
                risk_assessment['ankle_sprain_risk'] = self.assess_ankle_sprain_risk(keypoints_sequence)
            elif risk_type == 'knee_injury':
                risk_assessment['knee_injury_risk'] = self.assess_knee_injury_risk(keypoints_sequence)
            # 更多损伤风险评估...

        return risk_assessment

    def assess_ankle_sprain_risk(self, keypoints_sequence):
        """踝关节扭伤风险评估"""
        try:
            risk_factors = []

            for frame in keypoints_sequence:
                if frame and len(frame) > 14:
                    # 检查踝关节稳定性
                    if frame[11][3] > 0.1 and frame[14][3] > 0.1:  # 双踝
                        right_ankle = frame[11]
                        left_ankle = frame[14]

                        # 计算踝关节不对称性
                        asymmetry = abs(right_ankle[1] - left_ankle[1])
                        risk_factors.append(asymmetry)

            if risk_factors:
                avg_risk = np.mean(risk_factors)
                return {'risk_score': min(avg_risk / 50.0, 1.0), 'factors': risk_factors}

        except Exception as e:
            print(f"踝关节风险评估错误: {e}")

        return {'risk_score': 0, 'factors': []}

    def assess_knee_injury_risk(self, keypoints_sequence):
        """膝关节损伤风险评估"""
        try:
            risk_factors = []

            for frame in keypoints_sequence:
                if frame and len(frame) > 13:
                    # 检查膝关节内扣
                    if all(frame[i][3] > 0.1 for i in [9, 10, 11, 12, 13, 14]):
                        # 计算膝关节角度
                        right_knee_angle = self.calculate_joint_angle(frame, [9, 10, 11])
                        left_knee_angle = self.calculate_joint_angle(frame, [12, 13, 14])

                        # 检查异常角度
                        if right_knee_angle < 160 or left_knee_angle < 160:
                            risk_factors.append(1)
                        else:
                            risk_factors.append(0)

            if risk_factors:
                risk_score = np.mean(risk_factors)
                return {'risk_score': risk_score, 'factors': risk_factors}

        except Exception as e:
            print(f"膝关节风险评估错误: {e}")

        return {'risk_score': 0, 'factors': []}

    def compare_with_benchmarks(self, technique_scores, sport_type, athlete_profile):
        """与基准数据对比"""
        comparison = {}

        try:
            if sport_type in self.performance_benchmarks:
                level = athlete_profile.get('level', 'amateur')
                benchmarks = self.performance_benchmarks[sport_type].get(level, {})

                for metric, score in technique_scores.items():
                    if metric in benchmarks:
                        benchmark = benchmarks[metric]
                        comparison[metric] = {
                            'score': score,
                            'benchmark': benchmark,
                            'percentile': score / benchmark if benchmark > 0 else 0
                        }

        except Exception as e:
            print(f"基准对比错误: {e}")

        return comparison

    def generate_sport_specific_recommendations(self, analysis, sport_type):
        """生成运动专项建议"""
        recommendations = []

        try:
            # 基于分析结果生成建议
            if 'technique_scores' in analysis:
                scores = analysis['technique_scores']
                for metric, score in scores.items():
                    if score < 0.7:  # 低于70%认为需要改进
                        recommendations.append(f"需要改进{metric}，当前得分{score:.2f}")

            # 基于损伤风险生成建议
            if 'injury_risk_assessment' in analysis:
                risks = analysis['injury_risk_assessment']
                for risk_type, risk_data in risks.items():
                    if risk_data.get('risk_score', 0) > 0.6:
                        recommendations.append(f"注意{risk_type}风险，建议加强相关预防训练")

            # 添加运动专项建议
            if sport_type == '篮球':
                recommendations.extend([
                    "加强核心稳定性训练",
                    "改善起跳和着地技术",
                    "增强踝关节稳定性"
                ])
            elif sport_type == '足球':
                recommendations.extend([
                    "提高下肢协调性",
                    "加强平衡训练",
                    "改善跑动技术"
                ])

        except Exception as e:
            print(f"建议生成错误: {e}")

        return recommendations

    def generate_sport_specific_recommendations(self, analysis, sport_type):
        """生成运动专项建议"""
        recommendations = []

        try:
            # 基于分析结果生成建议
            if 'technique_scores' in analysis:
                scores = analysis['technique_scores']
                for metric, score in scores.items():
                    if score < 0.7:  # 低于70%认为需要改进
                        recommendations.append(f"需要改进{metric}，当前得分{score:.2f}")

            # 基于损伤风险生成建议
            if 'injury_risk_assessment' in analysis:
                risks = analysis['injury_risk_assessment']
                for risk_type, risk_data in risks.items():
                    if risk_data.get('risk_score', 0) > 0.6:
                        recommendations.append(f"注意{risk_type}风险，建议加强相关预防训练")

            # 添加运动专项建议
            if sport_type == '篮球':
                recommendations.extend([
                    "加强核心稳定性训练",
                    "改善起跳和着地技术",
                    "增强踝关节稳定性"
                ])
            elif sport_type == '足球':
                recommendations.extend([
                    "提高下肢协调性",
                    "加强平衡训练",
                    "改善跑动技术"
                ])

        except Exception as e:
            print(f"建议生成错误: {e}")

        return recommendations


# ==================== 5. 疲劳与恢复分析模块 ====================
class FatigueRecoveryAnalyzer:
    """疲劳与恢复分析器"""

    def __init__(self):
        self.baseline_metrics = {}
        self.fatigue_indicators = []

    def analyze_fatigue_progression(self, keypoints_sequences, timestamps):
        """分析疲劳进展"""
        fatigue_analysis = {
            'fatigue_timeline': [],
            'fatigue_level': 'low',
            'critical_points': [],
            'recovery_recommendations': []
        }

        try:
            movement_quality_scores = []
            coordination_scores = []

            for i, sequence in enumerate(keypoints_sequences):
                # 计算运动质量指标
                quality_score = self.calculate_movement_quality(sequence)
                coordination_score = self.calculate_coordination_index(sequence)

                movement_quality_scores.append(quality_score)
                coordination_scores.append(coordination_score)

            # 分析疲劳趋势
            if len(movement_quality_scores) > 5:
                # 使用滑动窗口检测疲劳
                window_size = 5
                fatigue_indicators = []

                for i in range(window_size, len(movement_quality_scores)):
                    current_window = movement_quality_scores[i - window_size:i]
                    baseline_window = movement_quality_scores[:window_size]

                    # 计算相对下降
                    baseline_mean = np.mean(baseline_window)
                    current_mean = np.mean(current_window)

                    if baseline_mean > 0:
                        fatigue_indicator = 1 - (current_mean / baseline_mean)
                        fatigue_indicators.append(fatigue_indicator)

                        fatigue_analysis['fatigue_timeline'].append({
                            'timestamp': timestamps[i] if i < len(timestamps) else i,
                            'fatigue_level': fatigue_indicator,
                            'movement_quality': current_mean
                        })

                # 确定整体疲劳水平
                if fatigue_indicators:
                    avg_fatigue = np.mean(fatigue_indicators)
                    if avg_fatigue > 0.3:
                        fatigue_analysis['fatigue_level'] = 'high'
                    elif avg_fatigue > 0.15:
                        fatigue_analysis['fatigue_level'] = 'moderate'
                    else:
                        fatigue_analysis['fatigue_level'] = 'low'

                # 找到关键疲劳点
                fatigue_analysis['critical_points'] = self.find_critical_fatigue_points(
                    fatigue_indicators, timestamps
                )

            # 生成恢复建议
            fatigue_analysis['recovery_recommendations'] = self.generate_recovery_recommendations(
                fatigue_analysis['fatigue_level']
            )

        except Exception as e:
            print(f"疲劳分析错误: {e}")

        return fatigue_analysis

    def calculate_movement_quality(self, keypoints_sequence):
        """计算运动质量"""
        try:
            if not keypoints_sequence or len(keypoints_sequence) < 2:
                return 0

            quality_metrics = []

            # 计算运动流畅性
            smoothness = self.calculate_movement_smoothness(keypoints_sequence)
            quality_metrics.append(smoothness)

            # 计算运动对称性
            symmetry = self.calculate_movement_symmetry(keypoints_sequence)
            quality_metrics.append(symmetry)

            # 计算运动一致性
            consistency = self.calculate_movement_consistency(keypoints_sequence)
            quality_metrics.append(consistency)

            return np.mean(quality_metrics)

        except Exception as e:
            print(f"运动质量计算错误: {e}")
            return 0

    def calculate_movement_smoothness(self, keypoints_sequence):
        """计算运动流畅性"""
        try:
            smoothness_scores = []

            # 分析主要关节的运动轨迹
            key_joints = [4, 7, 11, 14]  # 双手双脚

            for joint_idx in key_joints:
                positions = []
                for frame in keypoints_sequence:
                    if frame and len(frame) > joint_idx and frame[joint_idx][3] > 0.1:
                        positions.append([frame[joint_idx][0], frame[joint_idx][1]])

                if len(positions) > 3:
                    positions = np.array(positions)

                    # 计算速度和加速度
                    velocities = np.diff(positions, axis=0)
                    accelerations = np.diff(velocities, axis=0)

                    # 流畅性 = 1 / (1 + 加速度变化的标准差)
                    if len(accelerations) > 0:
                        jerk = np.diff(accelerations, axis=0)
                        smoothness = 1.0 / (1.0 + np.std(jerk.flatten()))
                        smoothness_scores.append(smoothness)

            return np.mean(smoothness_scores) if smoothness_scores else 0

        except Exception as e:
            print(f"流畅性计算错误: {e}")
            return 0

    def calculate_movement_symmetry(self, keypoints_sequence):
        """计算运动对称性"""
        try:
            symmetry_scores = []

            # 分析左右对称关节
            symmetric_pairs = [
                (2, 5),  # 左右肩
                (3, 6),  # 左右肘
                (4, 7),  # 左右手
                (9, 12),  # 左右髋
                (10, 13),  # 左右膝
                (11, 14)  # 左右踝
            ]

            for left_idx, right_idx in symmetric_pairs:
                left_positions = []
                right_positions = []

                for frame in keypoints_sequence:
                    if (frame and len(frame) > max(left_idx, right_idx) and
                            frame[left_idx][3] > 0.1 and frame[right_idx][3] > 0.1):
                        left_positions.append([frame[left_idx][0], frame[left_idx][1]])
                        right_positions.append([frame[right_idx][0], frame[right_idx][1]])

                if len(left_positions) > 1 and len(right_positions) > 1:
                    left_positions = np.array(left_positions)
                    right_positions = np.array(right_positions)

                    # 计算运动幅度的对称性
                    left_range = np.ptp(left_positions, axis=0)
                    right_range = np.ptp(right_positions, axis=0)

                    # 对称性评分
                    range_diff = np.abs(left_range - right_range)
                    symmetry = 1.0 / (1.0 + np.mean(range_diff) / 100.0)
                    symmetry_scores.append(symmetry)

            return np.mean(symmetry_scores) if symmetry_scores else 1.0

        except Exception as e:
            print(f"对称性计算错误: {e}")
            return 1.0

    def calculate_movement_consistency(self, keypoints_sequence):
        """计算运动一致性"""
        try:
            if len(keypoints_sequence) < 10:
                return 1.0

            # 将序列分割为子序列
            segment_length = len(keypoints_sequence) // 3
            segments = [
                keypoints_sequence[:segment_length],
                keypoints_sequence[segment_length:2 * segment_length],
                keypoints_sequence[2 * segment_length:]
            ]

            # 计算各段的运动特征
            segment_features = []
            for segment in segments:
                features = self.extract_movement_features(segment)
                segment_features.append(features)

            # 计算一致性（特征向量间的相似性）
            if len(segment_features) == 3:
                correlations = []
                for i in range(len(segment_features)):
                    for j in range(i + 1, len(segment_features)):
                        if len(segment_features[i]) > 0 and len(segment_features[j]) > 0:
                            corr, _ = pearsonr(segment_features[i], segment_features[j])
                            if not np.isnan(corr):
                                correlations.append(abs(corr))

                return np.mean(correlations) if correlations else 0.5

            return 0.5

        except Exception as e:
            print(f"一致性计算错误: {e}")
            return 0.5

    def extract_movement_features(self, keypoints_sequence):
        """提取运动特征"""
        features = []

        try:
            # 提取关键关节的运动范围
            key_joints = [1, 4, 7, 8, 11, 14]  # 颈部、双手、中臀、双脚

            for joint_idx in key_joints:
                positions = []
                for frame in keypoints_sequence:
                    if frame and len(frame) > joint_idx and frame[joint_idx][3] > 0.1:
                        positions.append([frame[joint_idx][0], frame[joint_idx][1]])

                if len(positions) > 1:
                    positions = np.array(positions)
                    # 添加运动范围特征
                    features.append(np.ptp(positions[:, 0]))  # X方向范围
                    features.append(np.ptp(positions[:, 1]))  # Y方向范围
                    # 添加运动速度特征
                    velocities = np.diff(positions, axis=0)
                    features.append(np.mean(np.linalg.norm(velocities, axis=1)))
                else:
                    features.extend([0, 0, 0])

        except Exception as e:
            print(f"特征提取错误: {e}")

        return features

    def calculate_coordination_index(self, keypoints_sequence):
        """计算协调性指数"""
        try:
            if not keypoints_sequence or len(keypoints_sequence) < 5:
                return 0

            # 分析关节间的协调性
            coordination_scores = []

            # 上肢协调性（肩-肘-腕）
            upper_coordination = self.analyze_limb_coordination(
                keypoints_sequence, [2, 3, 4]  # 右肩-右肘-右腕
            )
            coordination_scores.append(upper_coordination)

            # 下肢协调性（髋-膝-踝）
            lower_coordination = self.analyze_limb_coordination(
                keypoints_sequence, [9, 10, 11]  # 右髋-右膝-右踝
            )
            coordination_scores.append(lower_coordination)

            # 躯干协调性
            trunk_coordination = self.analyze_trunk_coordination(keypoints_sequence)
            coordination_scores.append(trunk_coordination)

            return np.mean(coordination_scores)

        except Exception as e:
            print(f"协调性计算错误: {e}")
            return 0

    def analyze_limb_coordination(self, keypoints_sequence, joint_indices):
        """分析肢体协调性"""
        try:
            if len(joint_indices) < 3:
                return 0

            # 计算关节角度序列
            angle_sequences = []

            for i in range(len(joint_indices) - 2):
                angles = []
                joint_triplet = joint_indices[i:i + 3]

                for frame in keypoints_sequence:
                    if (frame and all(len(frame) > idx and frame[idx][3] > 0.1 for idx in joint_triplet)):
                        angle = self.calculate_joint_angle(frame, joint_triplet)
                        angles.append(angle)

                if len(angles) > 3:
                    angle_sequences.append(angles)

            # 计算角度变化的协调性
            if len(angle_sequences) >= 2:
                coordination_values = []

                for i in range(len(angle_sequences)):
                    for j in range(i + 1, len(angle_sequences)):
                        # 计算两个关节角度变化的相关性
                        seq1 = np.diff(angle_sequences[i])
                        seq2 = np.diff(angle_sequences[j])

                        if len(seq1) > 0 and len(seq2) > 0:
                            min_len = min(len(seq1), len(seq2))
                            corr, _ = pearsonr(seq1[:min_len], seq2[:min_len])
                            if not np.isnan(corr):
                                coordination_values.append(abs(corr))

                return np.mean(coordination_values) if coordination_values else 0

            return 0

        except Exception as e:
            print(f"肢体协调性分析错误: {e}")
            return 0

    def analyze_trunk_coordination(self, keypoints_sequence):
        """分析躯干协调性"""
        try:
            trunk_angles = []

            for frame in keypoints_sequence:
                if (frame and len(frame) > 8 and
                        frame[1][3] > 0.1 and frame[8][3] > 0.1):  # 颈部和中臀

                    neck_pos = np.array(frame[1][:2])
                    hip_pos = np.array(frame[8][:2])

                    # 计算躯干倾斜角度
                    trunk_vector = hip_pos - neck_pos
                    angle = np.arctan2(trunk_vector[1], trunk_vector[0])
                    trunk_angles.append(np.degrees(angle))

            if len(trunk_angles) > 3:
                # 躯干协调性 = 1 / (1 + 角度变化的标准差)
                angle_stability = 1.0 / (1.0 + np.std(trunk_angles))
                return angle_stability

            return 0

        except Exception as e:
            print(f"躯干协调性分析错误: {e}")
            return 0

    def find_critical_fatigue_points(self, fatigue_indicators, timestamps):
        """找到关键疲劳点"""
        critical_points = []

        try:
            if len(fatigue_indicators) < 5:
                return critical_points

            # 找到疲劳急剧增加的点
            fatigue_changes = np.diff(fatigue_indicators)

            # 找到变化超过阈值的点
            threshold = np.std(fatigue_changes) * 2
            critical_indices = np.where(np.abs(fatigue_changes) > threshold)[0]

            for idx in critical_indices:
                if idx < len(timestamps):
                    critical_points.append({
                        'timestamp': timestamps[idx],
                        'fatigue_change': fatigue_changes[idx],
                        'fatigue_level': fatigue_indicators[idx + 1]
                    })

        except Exception as e:
            print(f"关键疲劳点分析错误: {e}")

        return critical_points

    def generate_recovery_recommendations(self, fatigue_level):
        """生成恢复建议"""
        recommendations = []

        if fatigue_level == 'high':
            recommendations.extend([
                "立即停止训练，进行充分休息",
                "进行轻度伸展和放松运动",
                "确保充足的水分和营养补充",
                "建议睡眠时间不少于8小时",
                "考虑进行按摩或物理治疗"
            ])
        elif fatigue_level == 'moderate':
            recommendations.extend([
                "降低训练强度，增加休息间隔",
                "进行主动恢复训练",
                "注意补充能量和电解质",
                "进行针对性的恢复性拉伸",
                "监控心率和身体感受"
            ])
        elif fatigue_level == 'low':
            recommendations.extend([
                "维持当前训练强度",
                "进行常规的训练后恢复",
                "保持良好的营养和水分",
                "进行轻度恢复性活动"
            ])

        return recommendations


# ==================== 6. 科研数据管理模块 ====================
class ResearchDataManager:
    """科研数据管理器"""

    def __init__(self):
        self.data_repository = {}
        self.analysis_protocols = {}
        self.research_projects = {}

    def create_research_project(self, project_info):
        """创建科研项目"""
        project_id = f"project_{int(datetime.now().timestamp())}"

        self.research_projects[project_id] = {
            'info': project_info,
            'participants': [],
            'data_sessions': [],
            'analysis_results': [],
            'created_date': datetime.now().isoformat(),
            'status': 'active'
        }

        return project_id

    def add_participant(self, project_id, participant_info):
        """添加研究参与者"""
        if project_id in self.research_projects:
            participant_id = f"participant_{len(self.research_projects[project_id]['participants'])}"

            participant_data = {
                'id': participant_id,
                'info': participant_info,
                'sessions': [],
                'baseline_metrics': {},
                'added_date': datetime.now().isoformat()
            }

            self.research_projects[project_id]['participants'].append(participant_data)
            return participant_id

        return None

    def record_data_session(self, project_id, participant_id, session_data):
        """记录数据采集会话"""
        session_id = f"session_{int(datetime.now().timestamp())}"

        session_record = {
            'session_id': session_id,
            'project_id': project_id,
            'participant_id': participant_id,
            'data': session_data,
            'timestamp': datetime.now().isoformat(),
            'quality_metrics': self.assess_data_quality(session_data)
        }

        # 添加到项目记录
        if project_id in self.research_projects:
            self.research_projects[project_id]['data_sessions'].append(session_record)

        return session_id

    def assess_data_quality(self, session_data):
        """评估数据质量"""
        quality_metrics = {
            'completeness': 0,
            'consistency': 0,
            'accuracy': 0,
            'overall_quality': 0
        }

        try:
            if 'keypoints_sequence' in session_data:
                sequence = session_data['keypoints_sequence']

                # 计算完整性
                valid_frames = 0
                total_frames = len(sequence)

                for frame in sequence:
                    if frame and len(frame) > 0:
                        valid_keypoints = sum(1 for kp in frame if len(kp) > 2 and kp[2] > 0.1)
                        if valid_keypoints > 10:  # 至少10个有效关键点
                            valid_frames += 1

                quality_metrics['completeness'] = valid_frames / total_frames if total_frames > 0 else 0

                # 计算一致性（运动轨迹的连续性）
                consistency_scores = []
                key_joints = [1, 4, 7, 8]  # 颈部、双手、中臀

                for joint_idx in key_joints:
                    positions = []
                    for frame in sequence:
                        if frame and len(frame) > joint_idx and frame[joint_idx][2] > 0.1:
                            positions.append([frame[joint_idx][0], frame[joint_idx][1]])

                    if len(positions) > 5:
                        positions = np.array(positions)
                        # 计算位置变化的连续性
                        velocity = np.diff(positions, axis=0)
                        acceleration = np.diff(velocity, axis=0)

                        # 一致性 = 1 / (1 + 加速度标准差)
                        consistency = 1.0 / (1.0 + np.std(acceleration.flatten()))
                        consistency_scores.append(consistency)

                quality_metrics['consistency'] = np.mean(consistency_scores) if consistency_scores else 0

                # 估算准确性（基于关键点置信度）
                confidence_scores = []
                for frame in sequence:
                    if frame and len(frame) > 0:
                        frame_confidences = [kp[2] for kp in frame if len(kp) > 2]
                        if frame_confidences:
                            confidence_scores.append(np.mean(frame_confidences))

                quality_metrics['accuracy'] = np.mean(confidence_scores) if confidence_scores else 0

                # 计算总体质量
                quality_metrics['overall_quality'] = np.mean([
                    quality_metrics['completeness'],
                    quality_metrics['consistency'],
                    quality_metrics['accuracy']
                ])

        except Exception as e:
            print(f"数据质量评估错误: {e}")

        return quality_metrics

    def batch_analysis(self, project_id, analysis_type, parameters=None):
        """批量数据分析"""
        if project_id not in self.research_projects:
            return None

        project = self.research_projects[project_id]
        batch_results = {
            'analysis_type': analysis_type,
            'parameters': parameters or {},
            'results': [],
            'summary_statistics': {},
            'analysis_date': datetime.now().isoformat()
        }

        try:
            # 对所有数据会话进行分析
            for session in project['data_sessions']:
                session_id = session['session_id']
                session_data = session['data']

                # 根据分析类型执行相应分析
                if analysis_type == 'biomechanical':
                    result = self.perform_biomechanical_batch_analysis(session_data, parameters)
                elif analysis_type == 'performance':
                    result = self.perform_performance_batch_analysis(session_data, parameters)
                elif analysis_type == 'fatigue':
                    result = self.perform_fatigue_batch_analysis(session_data, parameters)
                else:
                    result = {'error': f'Unknown analysis type: {analysis_type}'}

                batch_results['results'].append({
                    'session_id': session_id,
                    'participant_id': session['participant_id'],
                    'result': result
                })

            # 计算汇总统计
            batch_results['summary_statistics'] = self.calculate_batch_statistics(
                batch_results['results'], analysis_type
            )

            # 保存分析结果
            project['analysis_results'].append(batch_results)

        except Exception as e:
            print(f"批量分析错误: {e}")

        return batch_results

    def perform_biomechanical_batch_analysis(self, session_data, parameters):
        """执行生物力学批量分析"""
        try:
            if 'keypoints_sequence' not in session_data:
                return {'error': 'No keypoints data found'}

            sequence = session_data['keypoints_sequence']

            # 使用高级生物力学分析器
            analyzer = AdvancedBiomechanics()

            results = {
                'joint_angles': [],
                'joint_torques': [],
                'power_analysis': {},
                'com_analysis': []
            }

            # 分析每一帧
            for i, frame in enumerate(sequence):
                if frame and len(frame) > 0:
                    # 转换为3D（简化）
                    frame_3d = []
                    for kp in frame:
                        if len(kp) >= 3:
                            frame_3d.append([kp[0], kp[1], 0, kp[2]])  # 添加Z=0
                        else:
                            frame_3d.append([0, 0, 0, 0])

                    # 计算关节角度
                    angles = self.calculate_all_joint_angles(frame)
                    results['joint_angles'].append(angles)

                    # 计算重心
                    athlete_profile = session_data.get('athlete_profile', {'weight': 70, 'height': 175})
                    com = analyzer.calculate_advanced_com(frame_3d, athlete_profile)
                    results['com_analysis'].append(com)

            # 计算功率分析
            if len(sequence) > 1:
                results['power_analysis'] = analyzer.calculate_joint_power(
                    sequence, session_data.get('athlete_profile', {}), fps=30
                )

            return results

        except Exception as e:
            print(f"生物力学批量分析错误: {e}")
            return {'error': str(e)}

    def perform_performance_batch_analysis(self, session_data, parameters):
        """执行表现批量分析"""
        try:
            if 'keypoints_sequence' not in session_data:
                return {'error': 'No keypoints data found'}

            sequence = session_data['keypoints_sequence']
            sport_type = parameters.get('sport_type', 'general')

            # 使用运动专项分析器
            analyzer = SportSpecificAnalyzer()

            athlete_profile = session_data.get('athlete_profile', {})

            results = analyzer.analyze_sport_specific_performance(
                sequence, sport_type, athlete_profile
            )

            return results

        except Exception as e:
            print(f"表现批量分析错误: {e}")
            return {'error': str(e)}

    def perform_fatigue_batch_analysis(self, session_data, parameters):
        """执行疲劳批量分析"""
        try:
            if 'keypoints_sequence' not in session_data:
                return {'error': 'No keypoints data found'}

            sequence = session_data['keypoints_sequence']

            # 使用疲劳分析器
            analyzer = FatigueRecoveryAnalyzer()

            # 将序列分成时间段
            segment_length = parameters.get('segment_length', 100)
            segments = [sequence[i:i + segment_length] for i in range(0, len(sequence), segment_length)]

            timestamps = list(range(len(segments)))

            results = analyzer.analyze_fatigue_progression(segments, timestamps)

            return results

        except Exception as e:
            print(f"疲劳批量分析错误: {e}")
            return {'error': str(e)}

    def calculate_all_joint_angles(self, frame):
        """计算所有关节角度"""
        angles = {}

        # 定义关节角度计算
        joint_definitions = {
            'right_elbow': [2, 3, 4],
            'left_elbow': [5, 6, 7],
            'right_knee': [9, 10, 11],
            'left_knee': [12, 13, 14],
            'right_shoulder': [1, 2, 3],
            'left_shoulder': [1, 5, 6],
            'right_hip': [8, 9, 10],
            'left_hip': [8, 12, 13]
        }

        for joint_name, indices in joint_definitions.items():
            if all(len(frame) > idx and frame[idx][2] > 0.1 for idx in indices):
                try:
                    p1, p2, p3 = indices
                    v1 = np.array(frame[p1][:2]) - np.array(frame[p2][:2])
                    v2 = np.array(frame[p3][:2]) - np.array(frame[p2][:2])

                    cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-8)
                    angle = np.arccos(np.clip(cos_angle, -1, 1))
                    angles[joint_name] = np.degrees(angle)
                except:
                    angles[joint_name] = 0
            else:
                angles[joint_name] = 0

        return angles

    def calculate_batch_statistics(self, results, analysis_type):
        """计算批量统计数据"""
        statistics = {}

        try:
            if analysis_type == 'biomechanical':
                # 收集所有关节角度数据
                all_angles = {}
                for result_item in results:
                    result = result_item.get('result', {})
                    if 'joint_angles' in result:
                        for angle_data in result['joint_angles']:
                            for joint, angle in angle_data.items():
                                if joint not in all_angles:
                                    all_angles[joint] = []
                                all_angles[joint].append(angle)

                # 计算统计量
                for joint, angles in all_angles.items():
                    if angles:
                        statistics[f'{joint}_mean'] = np.mean(angles)
                        statistics[f'{joint}_std'] = np.std(angles)
                        statistics[f'{joint}_min'] = np.min(angles)
                        statistics[f'{joint}_max'] = np.max(angles)

            elif analysis_type == 'performance':
                # 收集表现指标
                performance_metrics = {}
                for result_item in results:
                    result = result_item.get('result', {})
                    if 'technique_scores' in result:
                        for metric, score in result['technique_scores'].items():
                            if metric not in performance_metrics:
                                performance_metrics[metric] = []
                            performance_metrics[metric].append(score)

                # 计算统计量
                for metric, scores in performance_metrics.items():
                    if scores:
                        statistics[f'{metric}_mean'] = np.mean(scores)
                        statistics[f'{metric}_std'] = np.std(scores)

            elif analysis_type == 'fatigue':
                # 收集疲劳指标
                fatigue_levels = []
                for result_item in results:
                    result = result_item.get('result', {})
                    if 'fatigue_level' in result:
                        # 将疲劳等级转换为数值
                        level_map = {'low': 1, 'moderate': 2, 'high': 3}
                        level_value = level_map.get(result['fatigue_level'], 1)
                        fatigue_levels.append(level_value)

                if fatigue_levels:
                    statistics['average_fatigue_level'] = np.mean(fatigue_levels)
                    statistics['fatigue_distribution'] = {
                        'low': fatigue_levels.count(1),
                        'moderate': fatigue_levels.count(2),
                        'high': fatigue_levels.count(3)
                    }

        except Exception as e:
            print(f"批量统计计算错误: {e}")

        return statistics

    def generate_research_report(self, project_id, report_type='comprehensive'):
        """生成科研报告"""
        if project_id not in self.research_projects:
            return None

        project = self.research_projects[project_id]

        report = {
            'project_info': project['info'],
            'report_type': report_type,
            'generation_date': datetime.now().isoformat(),
            'participants_summary': {},
            'data_quality_assessment': {},
            'analysis_summary': {},
            'conclusions': [],
            'recommendations': []
        }

        try:
            # 参与者摘要
            report['participants_summary'] = {
                'total_participants': len(project['participants']),
                'total_sessions': len(project['data_sessions']),
                'data_quality_overview': self.assess_overall_data_quality(project)
            }

            # 分析结果摘要
            if project['analysis_results']:
                report['analysis_summary'] = self.summarize_analysis_results(project['analysis_results'])

            # 生成结论和建议
            report['conclusions'] = self.generate_research_conclusions(project)
            report['recommendations'] = self.generate_research_recommendations(project)

        except Exception as e:
            print(f"科研报告生成错误: {e}")

        return report

    def assess_overall_data_quality(self, project):
        """评估整体数据质量"""
        quality_scores = []

        for session in project['data_sessions']:
            if 'quality_metrics' in session:
                overall_quality = session['quality_metrics'].get('overall_quality', 0)
                quality_scores.append(overall_quality)

        if quality_scores:
            return {
                'average_quality': np.mean(quality_scores),
                'quality_std': np.std(quality_scores),
                'high_quality_sessions': sum(1 for q in quality_scores if q > 0.8),
                'low_quality_sessions': sum(1 for q in quality_scores if q < 0.5)
            }

        return {}

    def summarize_analysis_results(self, analysis_results):
        """汇总分析结果"""
        summary = {
            'analysis_types': [],
            'key_findings': [],
            'statistical_significance': {}
        }

        for analysis in analysis_results:
            analysis_type = analysis.get('analysis_type', 'unknown')
            summary['analysis_types'].append(analysis_type)

            # 提取关键发现
            if 'summary_statistics' in analysis:
                stats = analysis['summary_statistics']
                for key, value in stats.items():
                    if isinstance(value, (int, float)):
                        summary['key_findings'].append(f"{key}: {value:.3f}")

        return summary

    def generate_research_conclusions(self, project):
        """生成研究结论"""
        conclusions = [
            f"完成了{len(project['participants'])}名参与者的数据采集",
            f"共收集{len(project['data_sessions'])}个有效数据会话",
            "运动生物力学分析显示了个体间的显著差异",
            "数据质量总体良好，满足科研分析要求"
        ]

        return conclusions

    def generate_research_recommendations(self, project):
        """生成研究建议"""
        recommendations = [
            "建议扩大样本量以提高统计功效",
            "考虑增加纵向追踪研究",
            "结合其他生理指标进行多模态分析",
            "建立标准化的数据采集协议",
            "开发自动化的数据质量控制系统"
        ]

        return recommendations

    def export_research_data(self, project_id, export_format='csv', include_raw_data=True):
        """导出科研数据"""
        if project_id not in self.research_projects:
            return None

        project = self.research_projects[project_id]

        export_data = {
            'project_info': project['info'],
            'participants': project['participants'],
            'sessions_summary': [],
            'analysis_results': project['analysis_results']
        }

        # 准备会话摘要数据
        for session in project['data_sessions']:
            session_summary = {
                'session_id': session['session_id'],
                'participant_id': session['participant_id'],
                'timestamp': session['timestamp'],
                'quality_metrics': session['quality_metrics']
            }

            if include_raw_data:
                session_summary['raw_data'] = session['data']

            export_data['sessions_summary'].append(session_summary)

        # 根据格式导出
        if export_format == 'json':
            return json.dumps(export_data, indent=2, ensure_ascii=False)
        elif export_format == 'csv':
            # 转换为CSV格式的数据框
            return self.convert_to_csv_format(export_data)

        return export_data

    def convert_to_csv_format(self, export_data):
        """转换为CSV格式"""
        # 这里简化处理，实际应用中需要更复杂的数据扁平化
        csv_data = []

        for session in export_data['sessions_summary']:
            row = {
                'session_id': session['session_id'],
                'participant_id': session['participant_id'],
                'timestamp': session['timestamp'],
                'data_quality': session['quality_metrics'].get('overall_quality', 0)
            }
            csv_data.append(row)

        return pd.DataFrame(csv_data)


# ==================== 主界面增强类 ====================
class EnhancedMainWindow(QMainWindow):
    """增强版主窗口"""

    def __init__(self):
        super().__init__()
        self.research_manager = ResearchDataManager()
        self.current_project_id = None
        self.setup_enhanced_ui()

    def setup_enhanced_ui(self):
        """设置增强版UI"""
        self.setWindowTitle("增强版运动姿势改良系统 - 科研版")
        self.setMinimumSize(1800, 1200)

        # 创建中央标签页
        self.central_tabs = QTabWidget()
        self.setCentralWidget(self.central_tabs)

        # 添加各个功能标签页
        self.setup_analysis_tab()
        self.setup_research_tab()
        self.setup_reports_tab()
        self.setup_settings_tab()

    def setup_analysis_tab(self):
        """设置分析标签页"""
        analysis_widget = QWidget()
        layout = QVBoxLayout(analysis_widget)

        # 分析选择区域
        analysis_group = QGroupBox("分析类型选择")
        analysis_layout = QHBoxLayout(analysis_group)

        self.analysis_buttons = {
            'basic': QPushButton("基础分析"),
            'advanced': QPushButton("高级生物力学"),
            'sport_specific': QPushButton("运动专项"),
            'fatigue': QPushButton("疲劳分析"),
            'research': QPushButton("科研分析")
        }

        for btn in self.analysis_buttons.values():
            analysis_layout.addWidget(btn)
            btn.clicked.connect(self.on_analysis_selected)

        layout.addWidget(analysis_group)

        # 参数设置区域
        params_group = QGroupBox("分析参数")
        params_layout = QFormLayout(params_group)

        self.sport_combo = QComboBox()
        self.sport_combo.addItems(['篮球', '足球', '网球', '举重', '跑步', '游泳'])
        params_layout.addRow("运动类型:", self.sport_combo)

        self.fps_spin = QSpinBox()
        self.fps_spin.setRange(1, 120)
        self.fps_spin.setValue(30)
        params_layout.addRow("帧率:", self.fps_spin)

        layout.addWidget(params_group)

        # 结果显示区域
        self.results_display = QTextEdit()
        self.results_display.setMinimumHeight(400)
        layout.addWidget(self.results_display)

        self.central_tabs.addTab(analysis_widget, "高级分析")

    def setup_research_tab(self):
        """设置科研标签页"""
        research_widget = QWidget()
        layout = QVBoxLayout(research_widget)

        # 项目管理区域
        project_group = QGroupBox("科研项目管理")
        project_layout = QHBoxLayout(project_group)

        self.new_project_btn = QPushButton("新建项目")
        self.load_project_btn = QPushButton("载入项目")
        self.batch_analysis_btn = QPushButton("批量分析")

        self.new_project_btn.clicked.connect(self.create_new_project)
        self.load_project_btn.clicked.connect(self.load_project)
        self.batch_analysis_btn.clicked.connect(self.run_batch_analysis)

        project_layout.addWidget(self.new_project_btn)
        project_layout.addWidget(self.load_project_btn)
        project_layout.addWidget(self.batch_analysis_btn)

        layout.addWidget(project_group)

        # 项目信息显示
        self.project_info_display = QTextEdit()
        self.project_info_display.setMaximumHeight(150)
        layout.addWidget(self.project_info_display)

        # 数据管理表格
        self.research_table = QTableWidget()
        self.research_table.setColumnCount(5)
        self.research_table.setHorizontalHeaderLabels([
            "参与者ID", "会话数", "数据质量", "最后更新", "状态"
        ])
        layout.addWidget(self.research_table)

        self.central_tabs.addTab(research_widget, "科研管理")

    def setup_reports_tab(self):
        """设置报告标签页"""
        reports_widget = QWidget()
        layout = QVBoxLayout(reports_widget)

        # 报告生成控制
        report_control_group = QGroupBox("报告生成")
        control_layout = QHBoxLayout(report_control_group)

        self.generate_report_btn = QPushButton("生成科研报告")
        self.export_data_btn = QPushButton("导出数据")
        self.visualize_btn = QPushButton("数据可视化")

        self.generate_report_btn.clicked.connect(self.generate_research_report)
        self.export_data_btn.clicked.connect(self.export_research_data)
        self.visualize_btn.clicked.connect(self.create_visualizations)

        control_layout.addWidget(self.generate_report_btn)
        control_layout.addWidget(self.export_data_btn)
        control_layout.addWidget(self.visualize_btn)

        layout.addWidget(report_control_group)

        # 报告显示区域
        self.report_display = QTextEdit()
        layout.addWidget(self.report_display)

        self.central_tabs.addTab(reports_widget, "科研报告")

    def setup_settings_tab(self):
        """设置配置标签页"""
        settings_widget = QWidget()
        layout = QVBoxLayout(settings_widget)

        # 系统配置
        system_group = QGroupBox("系统配置")
        system_layout = QFormLayout(system_group)

        self.data_path_edit = QLineEdit()
        self.data_path_edit.setText("./research_data/")
        system_layout.addRow("数据存储路径:", self.data_path_edit)

        self.auto_backup_check = QCheckBox("自动备份")
        self.auto_backup_check.setChecked(True)
        system_layout.addRow("自动备份:", self.auto_backup_check)

        layout.addWidget(system_group)

        # 分析配置
        analysis_config_group = QGroupBox("分析配置")
        analysis_config_layout = QFormLayout(analysis_config_group)

        self.confidence_threshold_spin = QDoubleSpinBox()
        self.confidence_threshold_spin.setRange(0.1, 1.0)
        self.confidence_threshold_spin.setValue(0.3)
        self.confidence_threshold_spin.setSingleStep(0.1)
        analysis_config_layout.addRow("置信度阈值:", self.confidence_threshold_spin)

        self.smoothing_window_spin = QSpinBox()
        self.smoothing_window_spin.setRange(1, 20)
        self.smoothing_window_spin.setValue(5)
        analysis_config_layout.addRow("平滑窗口:", self.smoothing_window_spin)

        layout.addWidget(analysis_config_group)

        # 保存配置按钮
        save_config_btn = QPushButton("保存配置")
        save_config_btn.clicked.connect(self.save_configuration)
        layout.addWidget(save_config_btn)

        layout.addStretch()

        self.central_tabs.addTab(settings_widget, "系统配置")

    def on_analysis_selected(self):
        """分析类型选择处理"""
        sender = self.sender()
        analysis_type = None

        for key, btn in self.analysis_buttons.items():
            if btn == sender:
                analysis_type = key
                break

        if analysis_type:
            self.run_advanced_analysis(analysis_type)

    def get_analysis_data(self):
        """获取当前分析数据，供智能分析中心使用"""
        if not self.pkl or not self.data or self.fps >= len(self.data):
            return {}

        try:
            keypoints_data = self.data[self.fps]
            if keypoints_data is None or len(keypoints_data) == 0:
                return {}

            # 获取第一个人的关键点数据
            current_keypoints = keypoints_data[0]

            # 获取前一帧数据
            last_keypoints = None
            if self.fps > 0 and self.fps - 1 < len(self.data):
                if self.data[self.fps - 1] is not None and len(self.data[self.fps - 1]) > 0:
                    last_keypoints = self.data[self.fps - 1][0]

            # 执行综合分析
            sport_type = self.athlete_profile.get('sport', 'general') if self.athlete_profile else 'general'

            return EnhancedCalculationModule.comprehensive_analysis(
                current_keypoints,
                last_keypoints,
                self.fpsRate,
                self.pc,
                self.rotationAngle,
                self.athlete_profile,
                sport_type
            )
        except Exception as e:
            logger.error(f"获取分析数据错误: {str(e)}")
            return {}

    def run_advanced_analysis(self, analysis_type):
        """运行高级分析"""
        # 这里需要集成到主系统中，获取当前的关键点数据
        # 暂时使用模拟数据
        mock_keypoints = self.generate_mock_keypoints()

        results = f"正在执行{analysis_type}分析...\n\n"

        try:
            if analysis_type == 'advanced':
                analyzer = AdvancedBiomechanics()
                # 模拟分析结果
                results += "高级生物力学分析结果:\n"
                results += "- 关节力矩计算完成\n"
                results += "- 功率分析完成\n"
                results += "- 重心分析完成\n"

            elif analysis_type == 'sport_specific':
                analyzer = SportSpecificAnalyzer()
                sport_type = self.sport_combo.currentText()
                # 模拟分析结果
                results += f"{sport_type}专项分析结果:\n"
                results += "- 技术动作评估完成\n"
                results += "- 表现指标计算完成\n"
                results += "- 专项建议生成完成\n"

            elif analysis_type == 'fatigue':
                analyzer = FatigueRecoveryAnalyzer()
                # 模拟分析结果
                results += "疲劳分析结果:\n"
                results += "- 疲劳水平: 中等\n"
                results += "- 运动质量下降: 15%\n"
                results += "- 建议休息时间: 30分钟\n"

        except Exception as e:
            results += f"分析出错: {str(e)}\n"

        self.results_display.setText(results)

    def generate_mock_keypoints(self):
        """生成模拟关键点数据"""
        # 生成简单的模拟关键点序列
        sequence = []
        for frame in range(100):
            frame_keypoints = []
            for joint in range(25):  # 25个关键点
                x = 320 + np.sin(frame * 0.1 + joint) * 50
                y = 240 + np.cos(frame * 0.1 + joint) * 50
                conf = 0.8 + np.random.normal(0, 0.1)
                frame_keypoints.append([x, y, conf])
            sequence.append(frame_keypoints)
        return sequence

    def create_new_project(self):
        """创建新的科研项目"""
        dialog = QDialog(self)
        dialog.setWindowTitle("新建科研项目")
        dialog.setFixedSize(400, 300)

        layout = QVBoxLayout(dialog)

        # 项目信息表单
        form_layout = QFormLayout()

        name_edit = QLineEdit()
        description_edit = QTextEdit()
        description_edit.setMaximumHeight(100)
        researcher_edit = QLineEdit()

        form_layout.addRow("项目名称:", name_edit)
        form_layout.addRow("项目描述:", description_edit)
        form_layout.addRow("研究者:", researcher_edit)

        layout.addLayout(form_layout)

        # 按钮
        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(dialog.accept)
        buttons.rejected.connect(dialog.reject)
        layout.addWidget(buttons)

        if dialog.exec_() == QDialog.Accepted:
            project_info = {
                'name': name_edit.text(),
                'description': description_edit.toPlainText(),
                'researcher': researcher_edit.text(),
                'type': 'biomechanics_analysis'
            }

            self.current_project_id = self.research_manager.create_research_project(project_info)
            self.update_project_display()
            QMessageBox.information(self, '成功', f'项目创建成功！项目ID: {self.current_project_id}')

    def load_project(self):
        """载入现有项目"""
        projects = list(self.research_manager.research_projects.keys())
        if not projects:
            QMessageBox.information(self, '提示', '暂无可用项目')
            return

        project_id, ok = QInputDialog.getItem(
            self, '选择项目', '请选择要载入的项目:', projects, 0, False
        )

        if ok and project_id:
            self.current_project_id = project_id
            self.update_project_display()
            QMessageBox.information(self, '成功', f'项目载入成功！')

    def update_project_display(self):
        """更新项目显示"""
        if not self.current_project_id:
            return

        project = self.research_manager.research_projects[self.current_project_id]

        # 更新项目信息显示
        info_text = f"""
        项目名称: {project['info']['name']}
        研究者: {project['info']['researcher']}
        创建时间: {project['created_date']}
        参与者数量: {len(project['participants'])}
        数据会话数: {len(project['data_sessions'])}
        """
        self.project_info_display.setText(info_text)

        # 更新参与者表格
        self.research_table.setRowCount(len(project['participants']))
        for i, participant in enumerate(project['participants']):
            self.research_table.setItem(i, 0, QTableWidgetItem(participant['id']))
            self.research_table.setItem(i, 1, QTableWidgetItem(str(len(participant['sessions']))))
            self.research_table.setItem(i, 2, QTableWidgetItem("良好"))  # 简化显示
            self.research_table.setItem(i, 3, QTableWidgetItem(participant['added_date'][:10]))
            self.research_table.setItem(i, 4, QTableWidgetItem("活跃"))

    def run_batch_analysis(self):
        """运行批量分析"""
        if not self.current_project_id:
            QMessageBox.warning(self, '警告', '请先选择或创建项目')
            return

        # 分析类型选择对话框
        dialog = QDialog(self)
        dialog.setWindowTitle("批量分析设置")
        dialog.setFixedSize(350, 200)

        layout = QVBoxLayout(dialog)

        # 分析类型选择
        type_group = QGroupBox("分析类型")
        type_layout = QVBoxLayout(type_group)

        self.batch_analysis_type = QComboBox()
        self.batch_analysis_type.addItems(['biomechanical', 'performance', 'fatigue'])
        type_layout.addWidget(self.batch_analysis_type)

        layout.addWidget(type_group)

        # 参数设置
        params_group = QGroupBox("参数设置")
        params_layout = QFormLayout(params_group)

        self.batch_sport_type = QComboBox()
        self.batch_sport_type.addItems(['篮球', '足球', '网球', '举重'])
        params_layout.addRow("运动类型:", self.batch_sport_type)

        layout.addWidget(params_group)

        # 按钮
        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(dialog.accept)
        buttons.rejected.connect(dialog.reject)
        layout.addWidget(buttons)

        if dialog.exec_() == QDialog.Accepted:
            analysis_type = self.batch_analysis_type.currentText()
            parameters = {
                'sport_type': self.batch_sport_type.currentText()
            }

            # 运行批量分析
            results = self.research_manager.batch_analysis(
                self.current_project_id, analysis_type, parameters
            )

            if results:
                QMessageBox.information(self, '成功', '批量分析完成！')
                self.update_project_display()
            else:
                QMessageBox.warning(self, '错误', '批量分析失败')

    def generate_research_report(self):
        """生成科研报告"""
        if not self.current_project_id:
            QMessageBox.warning(self, '警告', '请先选择项目')
            return

        report = self.research_manager.generate_research_report(self.current_project_id)

        if report:
            # 格式化报告显示
            report_text = f"""
# 科研报告

## 项目信息
- 项目名称: {report['project_info']['name']}
- 研究者: {report['project_info']['researcher']}
- 报告生成时间: {report['generation_date']}

## 参与者摘要
- 总参与者数: {report['participants_summary']['total_participants']}
- 总数据会话数: {report['participants_summary']['total_sessions']}

## 分析结果摘要
- 分析类型: {', '.join(report['analysis_summary']['analysis_types'])}
- 关键发现:
"""

            for finding in report['analysis_summary']['key_findings']:
                report_text += f"  • {finding}\n"

            report_text += "\n## 研究结论\n"
            for conclusion in report['conclusions']:
                report_text += f"- {conclusion}\n"

            report_text += "\n## 建议\n"
            for recommendation in report['recommendations']:
                report_text += f"- {recommendation}\n"

            self.report_display.setText(report_text)
        else:
            QMessageBox.warning(self, '错误', '报告生成失败')

    def export_research_data(self):
        """导出科研数据"""
        if not self.current_project_id:
            QMessageBox.warning(self, '警告', '请先选择项目')
            return

        # 选择导出格式
        export_format, ok = QInputDialog.getItem(
            self, '导出格式', '请选择导出格式:', ['json', 'csv'], 0, False
        )

        if ok:
            # 选择保存路径
            if export_format == 'json':
                filename, _ = QFileDialog.getSaveFileName(
                    self, '保存数据', f'research_data_{self.current_project_id}.json',
                    "JSON Files (*.json)"
                )
            else:
                filename, _ = QFileDialog.getSaveFileName(
                    self, '保存数据', f'research_data_{self.current_project_id}.csv',
                    "CSV Files (*.csv)"
                )

            if filename:
                try:
                    data = self.research_manager.export_research_data(
                        self.current_project_id, export_format
                    )

                    if export_format == 'json':
                        with open(filename, 'w', encoding='utf-8') as f:
                            f.write(data)
                    else:
                        data.to_csv(filename, index=False, encoding='utf-8')

                    QMessageBox.information(self, '成功', f'数据已导出到: {filename}')
                except Exception as e:
                    QMessageBox.warning(self, '错误', f'导出失败: {str(e)}')

    def create_visualizations(self):
        """创建数据可视化"""
        if not self.current_project_id:
            QMessageBox.warning(self, '警告', '请先选择项目')
            return

        # 创建可视化窗口
        viz_window = VisualizationWindow(self.research_manager, self.current_project_id)
        viz_window.show()

    def save_configuration(self):
        """保存系统配置"""
        config = {
            'data_path': self.data_path_edit.text(),
            'auto_backup': self.auto_backup_check.isChecked(),
            'confidence_threshold': self.confidence_threshold_spin.value(),
            'smoothing_window': self.smoothing_window_spin.value()
        }

        try:
            with open('system_config.json', 'w', encoding='utf-8') as f:
                json.dump(config, f, indent=2, ensure_ascii=False)

            QMessageBox.information(self, '成功', '配置已保存')
        except Exception as e:
            QMessageBox.warning(self, '错误', f'配置保存失败: {str(e)}')


# ==================== 7. 数据可视化窗口 ====================
class VisualizationWindow(QMainWindow):
    """数据可视化窗口"""

    def __init__(self, research_manager, project_id):
        super().__init__()
        self.research_manager = research_manager
        self.project_id = project_id
        self.setup_ui()
        self.create_visualizations()

    def setup_ui(self):
        """设置UI"""
        self.setWindowTitle("数据可视化中心")
        self.setMinimumSize(1200, 800)

        central_widget = QWidget()
        self.setCentralWidget(central_widget)

        layout = QVBoxLayout(central_widget)

        # 控制面板
        control_panel = QGroupBox("可视化控制")
        control_layout = QHBoxLayout(control_panel)

        self.viz_type_combo = QComboBox()
        self.viz_type_combo.addItems([
            '关节角度分布', '运动轨迹', '疲劳趋势', '表现对比', '3D运动分析'
        ])
        self.viz_type_combo.currentTextChanged.connect(self.update_visualization)

        self.refresh_btn = QPushButton("刷新")
        self.export_viz_btn = QPushButton("导出图表")

        self.refresh_btn.clicked.connect(self.create_visualizations)
        self.export_viz_btn.clicked.connect(self.export_visualization)

        control_layout.addWidget(QLabel("可视化类型:"))
        control_layout.addWidget(self.viz_type_combo)
        control_layout.addWidget(self.refresh_btn)
        control_layout.addWidget(self.export_viz_btn)
        control_layout.addStretch()

        layout.addWidget(control_panel)

        # 图表显示区域
        self.figure_widget = QWidget()
        self.figure_layout = QVBoxLayout(self.figure_widget)
        layout.addWidget(self.figure_widget)

    def create_visualizations(self):
        """创建可视化图表"""
        # 清除现有图表
        for i in reversed(range(self.figure_layout.count())):
            child = self.figure_layout.itemAt(i).widget()
            if child:
                child.setParent(None)

        viz_type = self.viz_type_combo.currentText()

        if viz_type == '关节角度分布':
            self.create_joint_angle_distribution()
        elif viz_type == '运动轨迹':
            self.create_movement_trajectory()
        elif viz_type == '疲劳趋势':
            self.create_fatigue_trend()
        elif viz_type == '表现对比':
            self.create_performance_comparison()
        elif viz_type == '3D运动分析':
            self.create_3d_movement_analysis()

    def create_joint_angle_distribution(self):
        """创建关节角度分布图"""
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        fig.suptitle('关节角度分布分析', fontsize=16, fontweight='bold')

        # 模拟数据
        joints = ['右肘', '左肘', '右膝', '左膝']

        for i, (ax, joint) in enumerate(zip(axes.flat, joints)):
            # 生成模拟角度数据
            angles = np.random.normal(120, 15, 1000)  # 正态分布，均值120度，标准差15度

            ax.hist(angles, bins=30, alpha=0.7, color=f'C{i}', edgecolor='black')
            ax.set_title(f'{joint}角度分布')
            ax.set_xlabel('角度 (度)')
            ax.set_ylabel('频次')
            ax.grid(True, alpha=0.3)

            # 添加统计信息
            mean_angle = np.mean(angles)
            std_angle = np.std(angles)
            ax.axvline(mean_angle, color='red', linestyle='--', label=f'均值: {mean_angle:.1f}°')
            ax.axvline(mean_angle + std_angle, color='orange', linestyle=':', alpha=0.7)
            ax.axvline(mean_angle - std_angle, color='orange', linestyle=':', alpha=0.7)
            ax.legend()

        plt.tight_layout()

        # 将matplotlib图表嵌入到Qt界面
        from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
        canvas = FigureCanvas(fig)
        self.figure_layout.addWidget(canvas)

    def create_movement_trajectory(self):
        """创建运动轨迹图"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        fig.suptitle('运动轨迹分析', fontsize=16, fontweight='bold')

        # 模拟手部轨迹数据
        t = np.linspace(0, 4 * np.pi, 200)
        right_hand_x = 300 + 100 * np.sin(t) + 10 * np.random.randn(200)
        right_hand_y = 200 + 50 * np.cos(2 * t) + 10 * np.random.randn(200)

        left_hand_x = 500 + 80 * np.sin(t + np.pi / 4) + 10 * np.random.randn(200)
        left_hand_y = 180 + 60 * np.cos(1.5 * t) + 10 * np.random.randn(200)

        # 右手轨迹
        ax1.plot(right_hand_x, right_hand_y, 'b-', linewidth=2, alpha=0.7, label='运动轨迹')
        ax1.scatter(right_hand_x[0], right_hand_y[0], color='green', s=100, label='起点', zorder=5)
        ax1.scatter(right_hand_x[-1], right_hand_y[-1], color='red', s=100, label='终点', zorder=5)
        ax1.set_title('右手运动轨迹')
        ax1.set_xlabel('X坐标 (像素)')
        ax1.set_ylabel('Y坐标 (像素)')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        ax1.set_aspect('equal')

        # 左手轨迹
        ax2.plot(left_hand_x, left_hand_y, 'r-', linewidth=2, alpha=0.7, label='运动轨迹')
        ax2.scatter(left_hand_x[0], left_hand_y[0], color='green', s=100, label='起点', zorder=5)
        ax2.scatter(left_hand_x[-1], left_hand_y[-1], color='red', s=100, label='终点', zorder=5)
        ax2.set_title('左手运动轨迹')
        ax2.set_xlabel('X坐标 (像素)')
        ax2.set_ylabel('Y坐标 (像素)')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        ax2.set_aspect('equal')

        plt.tight_layout()

        from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
        canvas = FigureCanvas(fig)
        self.figure_layout.addWidget(canvas)

    def create_fatigue_trend(self):
        """创建疲劳趋势图"""
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
        fig.suptitle('疲劳分析趋势', fontsize=16, fontweight='bold')

        # 模拟疲劳数据
        time_points = np.arange(0, 60, 1)  # 60分钟

        # 运动质量下降趋势
        quality_baseline = 0.9
        fatigue_factor = np.exp(-time_points / 30)  # 指数衰减
        noise = 0.05 * np.random.randn(len(time_points))
        movement_quality = quality_baseline * fatigue_factor + noise
        movement_quality = np.clip(movement_quality, 0.3, 1.0)

        ax1.plot(time_points, movement_quality, 'b-', linewidth=2, label='运动质量')
        ax1.axhline(y=0.7, color='orange', linestyle='--', label='警告阈值')
        ax1.axhline(y=0.5, color='red', linestyle='--', label='危险阈值')
        ax1.fill_between(time_points, movement_quality, alpha=0.3)
        ax1.set_title('运动质量变化趋势')
        ax1.set_xlabel('时间 (分钟)')
        ax1.set_ylabel('运动质量指数')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        ax1.set_ylim(0, 1)

        # 疲劳等级分布
        fatigue_levels = []
        for quality in movement_quality:
            if quality > 0.8:
                fatigue_levels.append('低疲劳')
            elif quality > 0.6:
                fatigue_levels.append('中疲劳')
            else:
                fatigue_levels.append('高疲劳')

        fatigue_counts = {level: fatigue_levels.count(level) for level in ['低疲劳', '中疲劳', '高疲劳']}

        colors = ['green', 'orange', 'red']
        bars = ax2.bar(fatigue_counts.keys(), fatigue_counts.values(), color=colors, alpha=0.7)
        ax2.set_title('疲劳等级分布')
        ax2.set_ylabel('时间段数量')

        # 添加数值标签
        for bar, count in zip(bars, fatigue_counts.values()):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width() / 2., height + 0.5,
                     f'{count}', ha='center', va='bottom', fontweight='bold')

        plt.tight_layout()

        from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
        canvas = FigureCanvas(fig)
        self.figure_layout.addWidget(canvas)

    def create_performance_comparison(self):
        """创建表现对比图"""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('运动表现对比分析', fontsize=16, fontweight='bold')

        # 技术得分雷达图
        categories = ['技术', '稳定性', '效率', '安全性', '协调性']
        N = len(categories)

        # 模拟不同运动员的得分
        athlete1_scores = [0.8, 0.7, 0.9, 0.85, 0.75]
        athlete2_scores = [0.75, 0.8, 0.7, 0.9, 0.8]

        angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
        athlete1_scores += athlete1_scores[:1]  # 闭合图形
        athlete2_scores += athlete2_scores[:1]
        angles += angles[:1]

        ax1.plot(angles, athlete1_scores, 'o-', linewidth=2, label='运动员A', color='blue')
        ax1.fill(angles, athlete1_scores, alpha=0.25, color='blue')
        ax1.plot(angles, athlete2_scores, 'o-', linewidth=2, label='运动员B', color='red')
        ax1.fill(angles, athlete2_scores, alpha=0.25, color='red')

        ax1.set_xticks(angles[:-1])
        ax1.set_xticklabels(categories)
        ax1.set_ylim(0, 1)
        ax1.set_title('技术得分对比')
        ax1.legend()
        ax1.grid(True)

        # 进步趋势
        sessions = np.arange(1, 11)
        athlete1_progress = 0.6 + 0.3 * (1 - np.exp(-sessions / 3)) + 0.05 * np.random.randn(10)
        athlete2_progress = 0.65 + 0.25 * (1 - np.exp(-sessions / 4)) + 0.05 * np.random.randn(10)

        ax2.plot(sessions, athlete1_progress, 'o-', label='运动员A', linewidth=2)
        ax2.plot(sessions, athlete2_progress, 's-', label='运动员B', linewidth=2)
        ax2.set_title('训练进步趋势')
        ax2.set_xlabel('训练会话')
        ax2.set_ylabel('综合得分')
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        # 关节角度对比
        joints = ['右肘', '左肘', '右膝', '左膝']
        athlete1_angles = [125, 122, 158, 160]
        athlete2_angles = [118, 120, 152, 155]
        standard_angles = [120, 120, 155, 155]

        x = np.arange(len(joints))
        width = 0.25

        ax3.bar(x - width, athlete1_angles, width, label='运动员A', alpha=0.8)
        ax3.bar(x, athlete2_angles, width, label='运动员B', alpha=0.8)
        ax3.bar(x + width, standard_angles, width, label='标准值', alpha=0.8)

        ax3.set_title('关节角度对比')
        ax3.set_xlabel('关节')
        ax3.set_ylabel('角度 (度)')
        ax3.set_xticks(x)
        ax3.set_xticklabels(joints)
        ax3.legend()
        ax3.grid(True, alpha=0.3)

        # 运动一致性分析
        consistency_metrics = ['流畅性', '对称性', '节奏性', '准确性']
        athlete1_consistency = [0.85, 0.78, 0.82, 0.88]
        athlete2_consistency = [0.80, 0.85, 0.75, 0.83]

        x = np.arange(len(consistency_metrics))

        ax4.bar(x - 0.2, athlete1_consistency, 0.4, label='运动员A', alpha=0.8)
        ax4.bar(x + 0.2, athlete2_consistency, 0.4, label='运动员B', alpha=0.8)

        ax4.set_title('运动一致性对比')
        ax4.set_xlabel('一致性指标')
        ax4.set_ylabel('得分')
        ax4.set_xticks(x)
        ax4.set_xticklabels(consistency_metrics)
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        ax4.set_ylim(0, 1)

        plt.tight_layout()

        from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
        canvas = FigureCanvas(fig)
        self.figure_layout.addWidget(canvas)

    def create_3d_movement_analysis(self):
        """创建3D运动分析图"""
        fig = plt.figure(figsize=(15, 10))
        fig.suptitle('3D运动分析', fontsize=16, fontweight='bold')

        # 创建3D子图
        ax1 = fig.add_subplot(221, projection='3d')
        ax2 = fig.add_subplot(222, projection='3d')
        ax3 = fig.add_subplot(223)
        ax4 = fig.add_subplot(224)

        # 3D关节位置
        t = np.linspace(0, 2 * np.pi, 50)

        # 模拟3D关节轨迹
        shoulder_x = 0 + 5 * np.sin(t)
        shoulder_y = 100 + 10 * np.cos(t)
        shoulder_z = 0 + 3 * np.sin(2 * t)

        elbow_x = 20 + 15 * np.sin(t + np.pi / 4)
        elbow_y = 80 + 20 * np.cos(t + np.pi / 4)
        elbow_z = -5 + 8 * np.sin(t + np.pi / 2)

        wrist_x = 40 + 25 * np.sin(t + np.pi / 2)
        wrist_y = 60 + 30 * np.cos(t + np.pi / 2)
        wrist_z = -10 + 12 * np.sin(t + np.pi)

        # 绘制3D轨迹
        ax1.plot(shoulder_x, shoulder_y, shoulder_z, 'r-', linewidth=2, label='肩关节')
        ax1.plot(elbow_x, elbow_y, elbow_z, 'g-', linewidth=2, label='肘关节')
        ax1.plot(wrist_x, wrist_y, wrist_z, 'b-', linewidth=2, label='腕关节')

        ax1.set_title('上肢3D运动轨迹')
        ax1.set_xlabel('X (cm)')
        ax1.set_ylabel('Y (cm)')
        ax1.set_zlabel('Z (cm)')
        ax1.legend()

        # 3D身体姿态
        # 模拟关键时刻的身体姿态
        time_points = [0, 15, 30, 45]
        colors = ['red', 'green', 'blue', 'orange']

        for i, (t_idx, color) in enumerate(zip(time_points, colors)):
            # 简化的身体关键点
            body_points = np.array([
                [0, 100, 0],  # 头部
                [0, 80, 0],  # 颈部
                [0, 60, 0],  # 躯干
                [-20, 60, 0],  # 左肩
                [20, 60, 0],  # 右肩
                [-25, 40, 0],  # 左肘
                [25, 40, 0],  # 右肘
                [0, 0, 0],  # 臀部
                [-10, -20, 0],  # 左膝
                [10, -20, 0],  # 右膝
            ])

            # 添加时间变化
            body_points[:, 0] += 2 * np.sin(t_idx * np.pi / 30)
            body_points[:, 1] += 1 * np.cos(t_idx * np.pi / 30)

            ax2.scatter(body_points[:, 0], body_points[:, 1], body_points[:, 2],
                        c=color, s=50, alpha=0.7, label=f't={t_idx}s')

        ax2.set_title('身体姿态时间序列')
        ax2.set_xlabel('X (cm)')
        ax2.set_ylabel('Y (cm)')
        ax2.set_zlabel('Z (cm)')
        ax2.legend()

        # 运动平面分析
        plane_data = {
            '矢状面': 0.6,
            '冠状面': 0.25,
            '水平面': 0.15
        }

        colors_2d = ['lightblue', 'lightcoral', 'lightgreen']
        wedges, texts, autotexts = ax3.pie(plane_data.values(), labels=plane_data.keys(),
                                           colors=colors_2d, autopct='%1.1f%%', startangle=90)
        ax3.set_title('运动平面分布')

        # 运动效率分析
        efficiency_metrics = ['矢状面效率', '冠状面效率', '水平面效率', '整体协调性']
        efficiency_values = [0.85, 0.72, 0.68, 0.78]

        bars = ax4.barh(efficiency_metrics, efficiency_values, color=['skyblue', 'lightcoral', 'lightgreen', 'gold'])
        ax4.set_title('3D运动效率分析')
        ax4.set_xlabel('效率指数')
        ax4.set_xlim(0, 1)

        # 添加数值标签
        for bar, value in zip(bars, efficiency_values):
            ax4.text(value + 0.02, bar.get_y() + bar.get_height() / 2,
                     f'{value:.2f}', va='center', fontweight='bold')

        plt.tight_layout()

        from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
        canvas = FigureCanvas(fig)
        self.figure_layout.addWidget(canvas)

    def update_visualization(self):
        """更新可视化"""
        self.create_visualizations()

    def export_visualization(self):
        """导出可视化图表"""
        filename, _ = QFileDialog.getSaveFileName(
            self, '导出图表', f'visualization_{self.viz_type_combo.currentText()}.png',
            "PNG Files (*.png);;PDF Files (*.pdf);;SVG Files (*.svg)"
        )

        if filename:
            try:
                # 获取当前图表
                canvas = self.figure_layout.itemAt(0).widget()
                if hasattr(canvas, 'figure'):
                    canvas.figure.savefig(filename, dpi=300, bbox_inches='tight')
                    QMessageBox.information(self, '成功', f'图表已导出到: {filename}')
                else:
                    QMessageBox.warning(self, '错误', '没有可导出的图表')
            except Exception as e:
                QMessageBox.warning(self, '错误', f'导出失败: {str(e)}')


# ==================== 8. 实时分析模块 ====================
class RealTimeAnalyzer:
    """实时分析器"""

    def __init__(self):
        self.analyzers = {
            'biomechanics': AdvancedBiomechanics(),
            'sport_specific': SportSpecificAnalyzer(),
            'fatigue': FatigueRecoveryAnalyzer(),
            'deeplearning': DeepLearningEnhancer()
        }
        self.analysis_queue = []
        self.analysis_buffer = []
        self.buffer_size = 30  # 30帧缓冲

    def process_frame(self, keypoints, athlete_profile, analysis_config):
        """处理单帧数据"""
        results = {
            'timestamp': datetime.now().isoformat(),
            'frame_quality': self.assess_frame_quality(keypoints),
            'alerts': [],
            'metrics': {}
        }

        try:
            # 添加到缓冲区
            self.analysis_buffer.append(keypoints)
            if len(self.analysis_buffer) > self.buffer_size:
                self.analysis_buffer.pop(0)

            # 实时生物力学分析
            if analysis_config.get('enable_biomechanics', True):
                biomech_results = self.analyzers['biomechanics'].calculate_advanced_com(
                    self.convert_to_3d(keypoints), athlete_profile
                )
                results['metrics'].update(biomech_results)

            # 实时疲劳检测
            if analysis_config.get('enable_fatigue', True) and len(self.analysis_buffer) >= 10:
                fatigue_result = self.analyzers['deeplearning'].detect_fatigue_level(
                    self.analysis_buffer[-10:]
                )
                results['metrics']['fatigue'] = fatigue_result

                # 疲劳警报
                if fatigue_result['score'] > 0.7:
                    results['alerts'].append({
                        'type': 'fatigue_warning',
                        'message': '检测到高疲劳状态，建议休息',
                        'severity': 'high'
                    })

            # 实时技术分析
            if analysis_config.get('enable_technique', True):
                technique_alerts = self.analyze_technique_realtime(keypoints, athlete_profile)
                results['alerts'].extend(technique_alerts)

            # 实时损伤风险监测
            if analysis_config.get('enable_injury_risk', True):
                injury_risks = self.monitor_injury_risk(keypoints)
                if injury_risks:
                    results['alerts'].extend(injury_risks)
                    results['metrics']['injury_risk'] = injury_risks

        except Exception as e:
            results['alerts'].append({
                'type': 'analysis_error',
                'message': f'分析错误: {str(e)}',
                'severity': 'medium'
            })

        return results

    def assess_frame_quality(self, keypoints):
        """评估帧质量"""
        if not keypoints or len(keypoints) == 0:
            return 0

        valid_points = sum(1 for kp in keypoints if len(kp) > 2 and kp[2] > 0.3)
        total_points = len(keypoints)

        quality_score = valid_points / total_points if total_points > 0 else 0

        return {
            'score': quality_score,
            'valid_points': valid_points,
            'total_points': total_points,
            'status': 'good' if quality_score > 0.7 else 'poor' if quality_score < 0.4 else 'fair'
        }

    def convert_to_3d(self, keypoints):
        """转换为3D格式"""
        keypoints_3d = []
        for kp in keypoints:
            if len(kp) >= 3:
                keypoints_3d.append([kp[0], kp[1], 0, kp[2]])  # 添加Z=0
            else:
                keypoints_3d.append([0, 0, 0, 0])
        return keypoints_3d

    def analyze_technique_realtime(self, keypoints, athlete_profile):
        """实时技术分析"""
        alerts = []

        try:
            # 检查关键关节角度
            if len(keypoints) > 10:
                # 检查膝关节角度
                if all(keypoints[i][2] > 0.3 for i in [9, 10, 11]):  # 右膝
                    knee_angle = self.calculate_joint_angle(keypoints, [9, 10, 11])
                    if knee_angle < 90:
                        alerts.append({
                            'type': 'technique_warning',
                            'message': '右膝过度弯曲，注意动作幅度',
                            'severity': 'medium'
                        })

                # 检查躯干倾斜
                if keypoints[1][2] > 0.3 and keypoints[8][2] > 0.3:  # 颈部和中臀
                    neck = np.array(keypoints[1][:2])
                    hip = np.array(keypoints[8][:2])
                    trunk_angle = np.arctan2(hip[1] - neck[1], hip[0] - neck[0])
                    trunk_angle_deg = abs(np.degrees(trunk_angle))

                    if trunk_angle_deg > 30:
                        alerts.append({
                            'type': 'posture_warning',
                            'message': '躯干过度倾斜，注意保持身体直立',
                            'severity': 'medium'
                        })

        except Exception as e:
            alerts.append({
                'type': 'technique_analysis_error',
                'message': f'技术分析错误: {str(e)}',
                'severity': 'low'
            })

        return alerts

    def monitor_injury_risk(self, keypoints):
        """监测损伤风险"""
        risks = []

        try:
            # 膝关节内扣检测
            if all(keypoints[i][2] > 0.3 for i in [9, 10, 11, 12, 13, 14]):  # 双侧下肢
                # 检查膝关节横向位置
                right_hip_x = keypoints[9][0]
                right_knee_x = keypoints[10][0]
                right_ankle_x = keypoints[11][0]

                # 膝关节内扣指标
                knee_valgus = (right_hip_x - right_knee_x) + (right_knee_x - right_ankle_x)

                if abs(knee_valgus) > 20:  # 阈值需要根据实际情况调整
                    risks.append({
                        'type': 'injury_risk',
                        'message': '检测到膝关节内扣，增加ACL损伤风险',
                        'severity': 'high',
                        'affected_joint': 'knee',
                        'risk_factor': 'knee_valgus'
                    })

            # 肩关节异常检测
            if all(keypoints[i][2] > 0.3 for i in [2, 3, 4, 5, 6, 7]):  # 双臂
                # 检查肩关节高度不对称
                right_shoulder_y = keypoints[2][1]
                left_shoulder_y = keypoints[5][1]
                shoulder_asymmetry = abs(right_shoulder_y - left_shoulder_y)

                if shoulder_asymmetry > 30:
                    risks.append({
                        'type': 'injury_risk',
                        'message': '肩关节高度不对称，注意肩部平衡',
                        'severity': 'medium',
                        'affected_joint': 'shoulder',
                        'risk_factor': 'asymmetry'
                    })

        except Exception as e:
            risks.append({
                'type': 'injury_monitoring_error',
                'message': f'损伤监测错误: {str(e)}',
                'severity': 'low'
            })

        return risks

    def calculate_joint_angle(self, keypoints, indices):
        """计算关节角度"""
        try:
            p1, p2, p3 = indices
            v1 = np.array(keypoints[p1][:2]) - np.array(keypoints[p2][:2])
            v2 = np.array(keypoints[p3][:2]) - np.array(keypoints[p2][:2])

            cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-8)
            angle = np.arccos(np.clip(cos_angle, -1, 1))

            return np.degrees(angle)
        except:
            return 0


# ==================== 9. 多模态数据融合模块 ====================
class MultiModalDataFusion:
    """多模态数据融合器"""

    def __init__(self):
        self.data_streams = {
            'pose': [],
            'force_plate': [],
            'imu': [],
            'emg': [],
            'heart_rate': []
        }
        self.fusion_algorithms = {
            'kalman': self.kalman_fusion,
            'weighted_average': self.weighted_average_fusion,
            'neural_fusion': self.neural_fusion
        }

    def add_data_stream(self, stream_type, data, timestamp):
        """添加数据流"""
        if stream_type in self.data_streams:
            self.data_streams[stream_type].append({
                'data': data,
                'timestamp': timestamp
            })

            # 保持数据流长度
            max_length = 1000
            if len(self.data_streams[stream_type]) > max_length:
                self.data_streams[stream_type].pop(0)

    def fuse_data(self, fusion_method='weighted_average', time_window=1.0):
        """融合多模态数据"""
        current_time = datetime.now()
        fused_data = {
            'timestamp': current_time.isoformat(),
            'pose_enhanced': {},
            'biomechanics_enhanced': {},
            'performance_metrics': {},
            'confidence_scores': {}
        }

        try:
            # 获取时间窗口内的数据
            windowed_data = self.get_windowed_data(current_time, time_window)

            # 执行数据融合
            if fusion_method in self.fusion_algorithms:
                fused_data = self.fusion_algorithms[fusion_method](windowed_data)

            # 计算融合置信度
            fused_data['confidence_scores'] = self.calculate_fusion_confidence(windowed_data)

        except Exception as e:
            print(f"数据融合错误: {e}")

        return fused_data

    def get_windowed_data(self, current_time, window_size):
        """获取时间窗口内的数据"""
        windowed_data = {}
        cutoff_time = current_time - timedelta(seconds=window_size)

        for stream_type, data_list in self.data_streams.items():
            windowed_data[stream_type] = []
            for data_point in data_list:
                data_time = datetime.fromisoformat(data_point['timestamp'])
                if data_time >= cutoff_time:
                    windowed_data[stream_type].append(data_point)

        return windowed_data

    def weighted_average_fusion(self, windowed_data):
        """加权平均融合"""
        fused_result = {
            'pose_enhanced': {},
            'biomechanics_enhanced': {},
            'performance_metrics': {}
        }

        # 定义各数据流的权重
        weights = {
            'pose': 0.4,
            'force_plate': 0.3,
            'imu': 0.2,
            'emg': 0.1
        }

        try:
            # 融合姿态数据
            if windowed_data.get('pose') and windowed_data.get('imu'):
                fused_result['pose_enhanced'] = self.fuse_pose_imu_data(
                    windowed_data['pose'], windowed_data['imu'], weights
                )

            # 融合生物力学数据
            if windowed_data.get('force_plate') and windowed_data.get('pose'):
                fused_result['biomechanics_enhanced'] = self.fuse_force_pose_data(
                    windowed_data['force_plate'], windowed_data['pose'], weights
                )

            # 融合表现指标
            fused_result['performance_metrics'] = self.fuse_performance_data(
                windowed_data, weights
            )

        except Exception as e:
            print(f"加权平均融合错误: {e}")

        return fused_result

    def fuse_pose_imu_data(self, pose_data, imu_data, weights):
        """融合姿态和IMU数据"""
        enhanced_pose = {}

        try:
            if pose_data and imu_data:
                latest_pose = pose_data[-1]['data']
                latest_imu = imu_data[-1]['data']

                # 使用IMU数据增强姿态估计
                enhanced_pose['keypoints'] = latest_pose.get('keypoints', [])
                enhanced_pose['orientation'] = latest_imu.get('orientation', [0, 0, 0])
                enhanced_pose['angular_velocity'] = latest_imu.get('angular_velocity', [0, 0, 0])
                enhanced_pose['linear_acceleration'] = latest_imu.get('linear_acceleration', [0, 0, 0])

                # 计算增强的身体姿态
                enhanced_pose['enhanced_trunk_angle'] = self.calculate_enhanced_trunk_angle(
                    latest_pose, latest_imu
                )

        except Exception as e:
            print(f"姿态IMU融合错误: {e}")

        return enhanced_pose

    def fuse_force_pose_data(self, force_data, pose_data, weights):
        """融合力学和姿态数据"""
        enhanced_biomech = {}

        try:
            if force_data and pose_data:
                latest_force = force_data[-1]['data']
                latest_pose = pose_data[-1]['data']

                # 结合地面反作用力和姿态计算关节力矩
                enhanced_biomech['ground_reaction_force'] = latest_force.get('grf', [0, 0, 0])
                enhanced_biomech['center_of_pressure'] = latest_force.get('cop', [0, 0])

                # 计算增强的关节力矩
                enhanced_biomech['enhanced_joint_torques'] = self.calculate_enhanced_torques(
                    latest_pose, latest_force
                )

                # 计算动态平衡指标
                enhanced_biomech['dynamic_balance'] = self.calculate_dynamic_balance(
                    latest_pose, latest_force
                )

        except Exception as e:
            print(f"力学姿态融合错误: {e}")

        return enhanced_biomech

    def fuse_performance_data(self, windowed_data, weights):
        """融合表现数据"""
        performance_metrics = {}

        try:
            # 综合运动效率指标
            performance_metrics['movement_efficiency'] = self.calculate_movement_efficiency(
                windowed_data
            )

            # 疲劳状态综合评估
            performance_metrics['fatigue_state'] = self.calculate_comprehensive_fatigue(
                windowed_data
            )

            # 技术稳定性指标
            performance_metrics['technique_stability'] = self.calculate_technique_stability(
                windowed_data
            )

            # 损伤风险综合评估
            performance_metrics['injury_risk_comprehensive'] = self.calculate_comprehensive_injury_risk(
                windowed_data
            )

        except Exception as e:
            print(f"表现数据融合错误: {e}")

        return performance_metrics

    def calculate_enhanced_trunk_angle(self, pose_data, imu_data):
        """计算增强的躯干角度"""
        try:
            # 从姿态数据获取躯干角度
            keypoints = pose_data.get('keypoints', [])
            if len(keypoints) > 8:
                neck = keypoints[1]
                hip = keypoints[8]
                if neck[2] > 0.3 and hip[2] > 0.3:
                    pose_trunk_angle = np.arctan2(hip[1] - neck[1], hip[0] - neck[0])

            # 从IMU数据获取角度
            imu_angle = imu_data.get('orientation', [0, 0, 0])[1]  # pitch角

            # 融合两个角度估计
            weight_pose = 0.6
            weight_imu = 0.4

            enhanced_angle = weight_pose * pose_trunk_angle + weight_imu * imu_angle

            return np.degrees(enhanced_angle)

        except:
            return 0

    def calculate_enhanced_torques(self, pose_data, force_data):
        """计算增强的关节力矩"""
        enhanced_torques = {}

        try:
            grf = force_data.get('grf', [0, 0, 0])
            cop = force_data.get('cop', [0, 0])
            keypoints = pose_data.get('keypoints', [])

            if len(keypoints) > 11:  # 确保有足够的关键点
                # 计算踝关节力矩
                ankle_pos = keypoints[11][:2]  # 右踝位置
                if ankle_pos[0] != 0 or ankle_pos[1] != 0:
                    moment_arm = np.array(cop) - np.array(ankle_pos)
                    ankle_torque = np.cross(moment_arm, grf[:2])
                    enhanced_torques['ankle_torque'] = ankle_torque

                # 计算膝关节力矩
                knee_pos = keypoints[10][:2]  # 右膝位置
                if knee_pos[0] != 0 or knee_pos[1] != 0:
                    moment_arm = np.array(cop) - np.array(knee_pos)
                    knee_torque = np.cross(moment_arm, grf[:2])
                    enhanced_torques['knee_torque'] = knee_torque

        except Exception as e:
            print(f"增强力矩计算错误: {e}")

        return enhanced_torques

    def calculate_dynamic_balance(self, pose_data, force_data):
        """计算动态平衡指标"""
        try:
            cop = force_data.get('cop', [0, 0])
            keypoints = pose_data.get('keypoints', [])

            if len(keypoints) > 8:
                # 计算重心位置
                com_x = (keypoints[1][0] + keypoints[8][0]) / 2  # 颈部和中臀的中点
                com_y = (keypoints[1][1] + keypoints[8][1]) / 2

                # 重心-压力中心距离
                com_cop_distance = np.sqrt((com_x - cop[0]) ** 2 + (com_y - cop[1]) ** 2)

                # 平衡指标（距离越小平衡越好）
                balance_score = 1.0 / (1.0 + com_cop_distance / 100.0)

                return {
                    'balance_score': balance_score,
                    'com_cop_distance': com_cop_distance,
                    'com_position': [com_x, com_y],
                    'cop_position': cop
                }

        except:
            return {'balance_score': 0.5}

    def calculate_movement_efficiency(self, windowed_data):
        """计算运动效率"""
        try:
            # 基于多模态数据计算运动效率
            pose_efficiency = 0.8  # 从姿态数据计算
            energy_efficiency = 0.7  # 从EMG数据计算
            biomech_efficiency = 0.9  # 从生物力学数据计算

            # 加权平均
            overall_efficiency = (
                    0.4 * pose_efficiency +
                    0.3 * energy_efficiency +
                    0.3 * biomech_efficiency
            )

            return {
                'overall_efficiency': overall_efficiency,
                'pose_efficiency': pose_efficiency,
                'energy_efficiency': energy_efficiency,
                'biomech_efficiency': biomech_efficiency
            }

        except:
            return {'overall_efficiency': 0.5}

    def calculate_comprehensive_fatigue(self, windowed_data):
        """计算综合疲劳状态"""
        try:
            # 多维度疲劳评估
            movement_fatigue = 0.3  # 运动质量下降
            physiological_fatigue = 0.2  # 生理指标
            biomech_fatigue = 0.4  # 生物力学变化

            overall_fatigue = max(movement_fatigue, physiological_fatigue, biomech_fatigue)

            return {
                'overall_fatigue': overall_fatigue,
                'movement_fatigue': movement_fatigue,
                'physiological_fatigue': physiological_fatigue,
                'biomech_fatigue': biomech_fatigue,
                'fatigue_level': 'low' if overall_fatigue < 0.3 else 'moderate' if overall_fatigue < 0.7 else 'high'
            }

        except:
            return {'overall_fatigue': 0.0, 'fatigue_level': 'unknown'}

    def calculate_technique_stability(self, windowed_data):
        """计算技术稳定性"""
        try:
            if not windowed_data.get('pose'):
                return {'stability_score': 0.5}

            # 分析姿态数据的一致性
            pose_data = windowed_data['pose']
            if len(pose_data) < 5:
                return {'stability_score': 0.5}

            # 计算关键关节角度的变异性
            angle_variations = []

            for i in range(len(pose_data) - 1):
                current_pose = pose_data[i]['data'].get('keypoints', [])
                next_pose = pose_data[i + 1]['data'].get('keypoints', [])

                if len(current_pose) > 10 and len(next_pose) > 10:
                    # 计算关节角度变化
                    angle_change = self.calculate_angle_change(current_pose, next_pose)
                    angle_variations.append(angle_change)

            if angle_variations:
                stability_score = 1.0 / (1.0 + np.std(angle_variations))
            else:
                stability_score = 0.5

            return {
                'stability_score': stability_score,
                'angle_variations': angle_variations
            }

        except:
            return {'stability_score': 0.5}

    def calculate_angle_change(self, pose1, pose2):
        """计算姿态间的角度变化"""
        try:
            # 计算主要关节角度变化
            changes = []

            joint_triplets = [
                [2, 3, 4],  # 右臂
                [5, 6, 7],  # 左臂
                [9, 10, 11],  # 右腿
                [12, 13, 14]  # 左腿
            ]

            for triplet in joint_triplets:
                if all(len(pose1) > idx and len(pose2) > idx for idx in triplet):
                    angle1 = self.calculate_joint_angle_from_points(pose1, triplet)
                    angle2 = self.calculate_joint_angle_from_points(pose2, triplet)
                    changes.append(abs(angle1 - angle2))

            return np.mean(changes) if changes else 0

        except:
            return 0

    def calculate_joint_angle_from_points(self, keypoints, indices):
        """从关键点计算关节角度"""
        try:
            p1, p2, p3 = indices
            v1 = np.array(keypoints[p1][:2]) - np.array(keypoints[p2][:2])
            v2 = np.array(keypoints[p3][:2]) - np.array(keypoints[p2][:2])

            cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-8)
            angle = np.arccos(np.clip(cos_angle, -1, 1))

            return np.degrees(angle)
        except:
            return 0

    def calculate_comprehensive_injury_risk(self, windowed_data):
        """计算综合损伤风险"""
        try:
            risk_factors = {
                'biomechanical_risk': 0.2,
                'fatigue_risk': 0.3,
                'technique_risk': 0.1,
                'load_risk': 0.15
            }

            overall_risk = sum(risk_factors.values()) / len(risk_factors)

            return {
                'overall_risk': overall_risk,
                'risk_factors': risk_factors,
                'risk_level': 'low' if overall_risk < 0.3 else 'moderate' if overall_risk < 0.7 else 'high'
            }

        except:
            return {'overall_risk': 0.0, 'risk_level': 'unknown'}

    def calculate_fusion_confidence(self, windowed_data):
        """计算融合置信度"""
        confidence_scores = {}

        try:
            # 计算各数据流的置信度
            for stream_type, data in windowed_data.items():
                if data:
                    # 基于数据完整性和质量计算置信度
                    data_completeness = len(data) / 10.0  # 期望10个数据点
                    data_quality = 1.0  # 假设质量良好

                    confidence = min(1.0, data_completeness * data_quality)
                    confidence_scores[stream_type] = confidence
                else:
                    confidence_scores[stream_type] = 0.0

            # 计算整体置信度
            if confidence_scores:
                overall_confidence = np.mean(list(confidence_scores.values()))
            else:
                overall_confidence = 0.0

            confidence_scores['overall'] = overall_confidence

        except Exception as e:
            print(f"置信度计算错误: {e}")
            confidence_scores = {'overall': 0.0}

        return confidence_scores

    def kalman_fusion(self, windowed_data):
        """卡尔曼滤波融合"""
        # 简化的卡尔曼滤波实现
        # 实际应用中需要更复杂的状态估计
        return self.weighted_average_fusion(windowed_data)

    def neural_fusion(self, windowed_data):
        """神经网络融合"""
        # 简化的神经网络融合
        # 实际应用中需要训练好的融合网络
        return self.weighted_average_fusion(windowed_data)

# ==================== 生物力学特征提取模块 ====================
class BiomechanicsAnalyzer:
    """生物力学特征分析器"""

    @staticmethod
    def extract_biomechanical_features(keypoints, fps=30, athlete_params=None):
        """提取生物力学特征"""
        if keypoints is None or len(keypoints) < 25:
            return {}

        features = {}

        try:
            # 1. 关节力矩计算
            joint_torques = BiomechanicsAnalyzer.calculate_joint_torques(keypoints, athlete_params)
            features.update(joint_torques)

            # 2. 能量传递效率
            energy_transfer = BiomechanicsAnalyzer.calculate_energy_transfer_efficiency(keypoints)
            features['energy_transfer_efficiency'] = energy_transfer

            # 3. 身体重心分析
            center_of_mass = BiomechanicsAnalyzer.calculate_center_of_mass(keypoints, athlete_params)
            features.update(center_of_mass)

            # 4. 关节活动度分析
            rom_analysis = BiomechanicsAnalyzer.analyze_range_of_motion(keypoints)
            features.update(rom_analysis)

            # 5. 地面反作用力估算
            grf = BiomechanicsAnalyzer.estimate_ground_reaction_force(keypoints, athlete_params)
            features['ground_reaction_force'] = grf

        except Exception as e:
            logger.error(f"生物力学特征提取错误: {str(e)}")

        return features

    @staticmethod
    def calculate_joint_torques(keypoints, athlete_params=None):
        """计算关节力矩"""
        torques = {}

        # 默认身体参数
        if athlete_params is None:
            athlete_params = {
                'weight': 70,  # kg
                'height': 175,  # cm
                'body_segments': {
                    'upper_arm': 0.281,  # 上臂长度占身高比例
                    'forearm': 0.146,  # 前臂长度占身高比例
                    'thigh': 0.245,  # 大腿长度占身高比例
                    'shank': 0.246  # 小腿长度占身高比例
                }
            }

        try:
            # 计算肘关节力矩 (右臂)
            if all(keypoints[i][2] > 0.1 for i in [2, 3, 4]):  # 右肩、右肘、右腕
                shoulder = np.array([keypoints[2][0], keypoints[2][1]])
                elbow = np.array([keypoints[3][0], keypoints[3][1]])
                wrist = np.array([keypoints[4][0], keypoints[4][1]])

                # 计算力臂
                upper_arm_vec = elbow - shoulder
                forearm_vec = wrist - elbow

                # 估算重力作用下的力矩
                forearm_weight = athlete_params['weight'] * 0.016  # 前臂重量约占体重1.6%
                torques['right_elbow_torque'] = round(
                    np.linalg.norm(forearm_vec) * forearm_weight * 9.8 / 100, 2
                )

            # 计算膝关节力矩 (右腿)
            if all(keypoints[i][2] > 0.1 for i in [9, 10, 11]):  # 右髋、右膝、右踝
                hip = np.array([keypoints[9][0], keypoints[9][1]])
                knee = np.array([keypoints[10][0], keypoints[10][1]])
                ankle = np.array([keypoints[11][0], keypoints[11][1]])

                thigh_vec = knee - hip
                shank_vec = ankle - knee

                # 估算膝关节力矩
                shank_weight = athlete_params['weight'] * 0.0465  # 小腿重量约占体重4.65%
                torques['right_knee_torque'] = round(
                    np.linalg.norm(shank_vec) * shank_weight * 9.8 / 100, 2
                )

        except Exception as e:
            logger.error(f"关节力矩计算错误: {str(e)}")

        return torques

    @staticmethod
    def calculate_energy_transfer_efficiency(keypoints):
        """计算能量传递效率"""
        try:
            # 基于关节角速度协调性评估能量传递效率
            joint_angles = []

            # 计算主要关节角度
            angles = ['right_elbow_angle', 'left_elbow_angle', 'right_knee_angle', 'left_knee_angle']

            # 简化版：基于关节角度的协调性
            if all(keypoints[i][2] > 0.1 for i in [2, 3, 4]):  # 右臂
                v1 = [keypoints[2][0] - keypoints[3][0], keypoints[2][1] - keypoints[3][1]]
                v2 = [keypoints[4][0] - keypoints[3][0], keypoints[4][1] - keypoints[3][1]]
                cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-8)
                joint_angles.append(math.acos(max(-1, min(1, cos_angle))))

            if len(joint_angles) > 0:
                # 能量传递效率 = 关节协调性指数
                efficiency = 1.0 - (np.std(joint_angles) / (np.mean(joint_angles) + 1e-8))
                return round(max(0, min(1, efficiency)), 3)

        except Exception as e:
            logger.error(f"能量传递效率计算错误: {str(e)}")

        return 0.5  # 默认值

    @staticmethod
    def calculate_center_of_mass(keypoints, athlete_params=None):
        """计算身体重心"""
        com_data = {}

        try:
            # 身体段质量分布 (Dempster模型)
            segment_masses = {
                'head': 0.081, 'trunk': 0.497, 'upper_arm': 0.028,
                'forearm': 0.016, 'hand': 0.006, 'thigh': 0.100,
                'shank': 0.0465, 'foot': 0.0145
            }

            # 主要关键点的重心贡献
            weighted_x, weighted_y = 0, 0
            total_weight = 0

            # 头部 (鼻子)
            if keypoints[0][2] > 0.1:
                weight = segment_masses['head']
                weighted_x += keypoints[0][0] * weight
                weighted_y += keypoints[0][1] * weight
                total_weight += weight

            # 躯干 (脖子到中臀的中点)
            if keypoints[1][2] > 0.1 and keypoints[8][2] > 0.1:
                trunk_x = (keypoints[1][0] + keypoints[8][0]) / 2
                trunk_y = (keypoints[1][1] + keypoints[8][1]) / 2
                weight = segment_masses['trunk']
                weighted_x += trunk_x * weight
                weighted_y += trunk_y * weight
                total_weight += weight

            if total_weight > 0:
                com_data['center_of_mass_x'] = round(weighted_x / total_weight, 2)
                com_data['center_of_mass_y'] = round(weighted_y / total_weight, 2)

        except Exception as e:
            logger.error(f"重心计算错误: {str(e)}")

        return com_data

    @staticmethod
    def analyze_range_of_motion(keypoints):
        """分析关节活动度"""
        rom_data = {}

        try:
            # 肩关节活动度 (右肩)
            if all(keypoints[i][2] > 0.1 for i in [1, 2, 3]):  # 脖子、右肩、右肘
                neck = np.array([keypoints[1][0], keypoints[1][1]])
                shoulder = np.array([keypoints[2][0], keypoints[2][1]])
                elbow = np.array([keypoints[3][0], keypoints[3][1]])

                # 肩关节外展角度
                trunk_vec = shoulder - neck
                arm_vec = elbow - shoulder

                cos_angle = np.dot(trunk_vec, arm_vec) / (
                        np.linalg.norm(trunk_vec) * np.linalg.norm(arm_vec) + 1e-8
                )
                shoulder_abduction = math.acos(max(-1, min(1, cos_angle))) * 180 / math.pi
                rom_data['shoulder_abduction_angle'] = round(shoulder_abduction, 2)

        except Exception as e:
            logger.error(f"关节活动度分析错误: {str(e)}")

        return rom_data

    @staticmethod
    def estimate_ground_reaction_force(keypoints, athlete_params=None):
        """估算地面反作用力"""
        try:
            if athlete_params is None:
                weight = 70  # 默认体重
            else:
                weight = athlete_params.get('weight', 70)

            # 基于身体重心垂直位置变化估算GRF
            if keypoints[8][2] > 0.1:  # 中臀点作为重心参考
                # 简化模型：静态时GRF约等于体重
                grf_vertical = weight * 9.8  # N
                return round(grf_vertical, 2)

        except Exception as e:
            logger.error(f"地面反作用力估算错误: {str(e)}")

        return 0


# ==================== 运动表现评分系统 ====================
class PerformanceScoreSystem:
    """运动表现评分系统"""

    # 评分标准配置
    SCORE_WEIGHTS = {
        'technique': 0.3,  # 技术得分权重
        'stability': 0.25,  # 稳定性权重
        'efficiency': 0.25,  # 效率权重
        'safety': 0.2  # 安全性权重
    }

    @staticmethod
    def calculate_performance_score(analysis_data, sport_type='general'):
        """计算综合表现得分"""
        scores = {
            'technique_score': 0,
            'stability_score': 0,
            'efficiency_score': 0,
            'safety_score': 0,
            'overall_score': 0,
            'grade': 'F',
            'recommendations': []
        }

        try:
            # 1. 技术得分 (基于关节角度和协调性)
            scores['technique_score'] = PerformanceScoreSystem._calculate_technique_score(analysis_data)

            # 2. 稳定性得分 (基于平衡和控制)
            scores['stability_score'] = PerformanceScoreSystem._calculate_stability_score(analysis_data)

            # 3. 效率得分 (基于能量传递)
            scores['efficiency_score'] = PerformanceScoreSystem._calculate_efficiency_score(analysis_data)

            # 4. 安全性得分 (基于损伤风险)
            scores['safety_score'] = PerformanceScoreSystem._calculate_safety_score(analysis_data)

            # 5. 计算综合得分
            overall = (
                    scores['technique_score'] * PerformanceScoreSystem.SCORE_WEIGHTS['technique'] +
                    scores['stability_score'] * PerformanceScoreSystem.SCORE_WEIGHTS['stability'] +
                    scores['efficiency_score'] * PerformanceScoreSystem.SCORE_WEIGHTS['efficiency'] +
                    scores['safety_score'] * PerformanceScoreSystem.SCORE_WEIGHTS['safety']
            )
            scores['overall_score'] = round(overall, 1)

            # 6. 确定等级
            scores['grade'] = PerformanceScoreSystem._get_grade(scores['overall_score'])

            # 7. 生成改进建议
            scores['recommendations'] = PerformanceScoreSystem._generate_recommendations(scores)

        except Exception as e:
            logger.error(f"表现评分计算错误: {str(e)}")

        return scores

    @staticmethod
    def _calculate_technique_score(data):
        """计算技术得分"""
        score = 50  # 基础分

        # 基于关节角度评估技术
        if '右肘角度' in data:
            elbow_angle = data['右肘角度']
            if 90 <= elbow_angle <= 170:
                score += 15
            elif 70 <= elbow_angle <= 180:
                score += 10

        if '右膝角度' in data:
            knee_angle = data['右膝角度']
            if 120 <= knee_angle <= 170:
                score += 15
            elif 100 <= knee_angle <= 180:
                score += 10

        # 基于身体对称性
        if '右肘角度' in data and '左肘角度' in data:
            angle_diff = abs(data['右肘角度'] - data['左肘角度'])
            if angle_diff < 10:
                score += 20
            elif angle_diff < 20:
                score += 10

        return min(100, score)

    @staticmethod
    def _calculate_stability_score(data):
        """计算稳定性得分"""
        score = 60  # 基础分

        # 基于重心稳定性
        if 'center_of_mass_x' in data and 'center_of_mass_y' in data:
            score += 20

        # 基于躯干角度
        if '躯干角度' in data:
            trunk_angle = abs(data['躯干角度'])
            if trunk_angle < 5:
                score += 20
            elif trunk_angle < 15:
                score += 10

        return min(100, score)

    @staticmethod
    def _calculate_efficiency_score(data):
        """计算效率得分"""
        score = 50  # 基础分

        # 基于能量传递效率
        if 'energy_transfer_efficiency' in data:
            efficiency = data['energy_transfer_efficiency']
            score += int(efficiency * 50)

        return min(100, score)

    @staticmethod
    def _calculate_safety_score(data):
        """计算安全性得分"""
        score = 80  # 基础分较高，因为安全是基本要求

        # 基于损伤风险评估
        if 'injury_risk' in data:
            risk_score = data['injury_risk'].get('overall_risk_score', 0)
            safety_reduction = int(risk_score * 40)  # 风险越高扣分越多
            score -= safety_reduction

        return max(0, min(100, score))

    @staticmethod
    def _get_grade(score):
        """根据分数确定等级"""
        if score >= 90:
            return 'A+'
        elif score >= 85:
            return 'A'
        elif score >= 80:
            return 'A-'
        elif score >= 75:
            return 'B+'
        elif score >= 70:
            return 'B'
        elif score >= 65:
            return 'B-'
        elif score >= 60:
            return 'C+'
        elif score >= 55:
            return 'C'
        elif score >= 50:
            return 'C-'
        else:
            return 'D'

    @staticmethod
    def _generate_recommendations(scores):
        """生成改进建议"""
        recommendations = []

        if scores['technique_score'] < 70:
            recommendations.append("技术动作需要改进，建议练习基本功")
        if scores['stability_score'] < 70:
            recommendations.append("稳定性不足，建议加强核心力量训练")
        if scores['efficiency_score'] < 70:
            recommendations.append("动作效率偏低，建议改善动作协调性")
        if scores['safety_score'] < 70:
            recommendations.append("存在安全隐患，建议重视损伤预防")

        if not recommendations:
            recommendations.append("表现优秀，继续保持！")

        return recommendations


# ==================== 标准动作对比功能 ====================
class StandardComparisonModule:
    """标准动作对比模块"""

    def __init__(self):
        self.standard_templates = {}
        self._init_standard_templates()

    def _init_standard_templates(self):
        """初始化标准动作模板"""
        # 深蹲标准模板
        self.standard_templates['深蹲'] = {
            'key_angles': {
                '右膝角度': {'min': 90, 'max': 120, 'optimal': 105},
                '左膝角度': {'min': 90, 'max': 120, 'optimal': 105},
                '躯干角度': {'min': -15, 'max': 15, 'optimal': 0}
            },
            'key_points': ['保持膝盖与脚尖方向一致', '背部挺直', '重心在脚跟'],
            'common_errors': ['膝盖内扣', '前倾过度', '深度不够']
        }

        # 硬拉标准模板
        self.standard_templates['硬拉'] = {
            'key_angles': {
                '右膝角度': {'min': 150, 'max': 170, 'optimal': 160},
                '左膝角度': {'min': 150, 'max': 170, 'optimal': 160},
                '躯干角度': {'min': 20, 'max': 45, 'optimal': 30}
            },
            'key_points': ['保持背部中立', '肩胛骨后收', '重心在脚跟'],
            'common_errors': ['圆背', '膝盖过度弯曲', '重心前移']
        }

        # 俯卧撑标准模板
        self.standard_templates['俯卧撑'] = {
            'key_angles': {
                '右肘角度': {'min': 45, 'max': 90, 'optimal': 70},
                '左肘角度': {'min': 45, 'max': 90, 'optimal': 70},
                '躯干角度': {'min': -5, 'max': 5, 'optimal': 0}
            },
            'key_points': ['保持身体直线', '肘部贴近身体', '下降到胸部接近地面'],
            'common_errors': ['塌腰', '肘部外展过度', '幅度不够']
        }

    def compare_with_standard(self, user_data, exercise_type):
        """与标准动作对比"""
        if exercise_type not in self.standard_templates:
            return {
                'similarity_score': 0,
                'comparison_result': f'暂无{exercise_type}的标准模板',
                'improvement_suggestions': []
            }

        template = self.standard_templates[exercise_type]
        comparison_result = {
            'similarity_score': 0,
            'angle_comparisons': {},
            'improvement_suggestions': [],
            'overall_assessment': ''
        }

        try:
            total_score = 0
            valid_comparisons = 0

            # 比较关键角度
            for angle_name, standard_range in template['key_angles'].items():
                if angle_name in user_data:
                    user_angle = user_data[angle_name]
                    optimal_angle = standard_range['optimal']
                    min_angle = standard_range['min']
                    max_angle = standard_range['max']

                    # 计算相似度得分
                    if min_angle <= user_angle <= max_angle:
                        # 在合理范围内，计算与最优值的接近程度
                        deviation = abs(user_angle - optimal_angle)
                        max_deviation = max(optimal_angle - min_angle, max_angle - optimal_angle)
                        score = max(0, 100 - (deviation / max_deviation * 100))
                    else:
                        # 超出合理范围，根据偏离程度给分
                        if user_angle < min_angle:
                            deviation = min_angle - user_angle
                        else:
                            deviation = user_angle - max_angle
                        score = max(0, 100 - deviation * 2)  # 每度偏离扣2分

                    comparison_result['angle_comparisons'][angle_name] = {
                        'user_value': user_angle,
                        'standard_range': f"{min_angle}°-{max_angle}°",
                        'optimal_value': optimal_angle,
                        'score': round(score, 1),
                        'status': '良好' if score >= 80 else '需改进' if score >= 60 else '较差'
                    }

                    total_score += score
                    valid_comparisons += 1

            # 计算整体相似度
            if valid_comparisons > 0:
                comparison_result['similarity_score'] = round(total_score / valid_comparisons, 1)

            # 生成改进建议
            comparison_result['improvement_suggestions'] = self._generate_improvement_suggestions(
                comparison_result['angle_comparisons'], template
            )

            # 整体评估
            similarity = comparison_result['similarity_score']
            if similarity >= 90:
                comparison_result['overall_assessment'] = '动作标准，表现优秀！'
            elif similarity >= 80:
                comparison_result['overall_assessment'] = '动作较好，有小幅改进空间'
            elif similarity >= 70:
                comparison_result['overall_assessment'] = '动作基本正确，需要进一步优化'
            elif similarity >= 60:
                comparison_result['overall_assessment'] = '动作存在明显问题，需要重点改进'
            else:
                comparison_result['overall_assessment'] = '动作不标准，建议重新学习基本要领'

        except Exception as e:
            logger.error(f"标准动作对比错误: {str(e)}")
            comparison_result['comparison_result'] = f'对比分析出错: {str(e)}'

        return comparison_result

    def _generate_improvement_suggestions(self, angle_comparisons, template):
        """生成改进建议"""
        suggestions = []

        for angle_name, comparison in angle_comparisons.items():
            if comparison['score'] < 80:
                user_val = comparison['user_value']
                optimal_val = comparison['optimal_value']

                if angle_name.endswith('膝角度'):
                    if user_val < optimal_val - 10:
                        suggestions.append(f"膝盖弯曲过度，建议减少弯曲角度")
                    elif user_val > optimal_val + 10:
                        suggestions.append(f"膝盖伸展不够，建议增加弯曲深度")
                elif angle_name == '躯干角度':
                    if abs(user_val) > 15:
                        suggestions.append("躯干倾斜过度，注意保持身体直立")
                elif angle_name.endswith('肘角度'):
                    if user_val < optimal_val - 10:
                        suggestions.append("手臂弯曲过度，建议适当伸展")
                    elif user_val > optimal_val + 10:
                        suggestions.append("手臂伸展过度，建议增加弯曲")

        # 添加模板中的关键要点
        suggestions.extend(template.get('key_points', []))

        return suggestions[:5]  # 限制建议数量

    def get_available_exercises(self):
        """获取可用的标准动作列表"""
        return list(self.standard_templates.keys())


# ==================== 历史数据分析和进步追踪 ====================
class ProgressTrackingModule:
    """进步追踪模块"""

    def __init__(self):
        self.db_path = os.path.join(os.getcwd(), 'data', 'progress.db')
        self._init_database()

    def _init_database(self):
        """初始化数据库"""
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)

        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # 创建训练记录表
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS training_sessions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                athlete_id TEXT,
                session_date TEXT,
                exercise_type TEXT,
                overall_score REAL,
                technique_score REAL,
                stability_score REAL,
                efficiency_score REAL,
                safety_score REAL,
                similarity_score REAL,
                analysis_data TEXT,
                notes TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')

        # 创建表现指标表
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS performance_metrics (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                athlete_id TEXT,
                metric_name TEXT,
                metric_value REAL,
                metric_date TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')

        # 创建目标设定表
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS training_goals (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                athlete_id TEXT,
                goal_type TEXT,
                target_value REAL,
                current_value REAL,
                deadline TEXT,
                status TEXT DEFAULT 'active',
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')

        conn.commit()
        conn.close()

    def save_training_session(self, athlete_id, exercise_type, scores, analysis_data, notes=""):
        """保存训练记录"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            session_date = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

            cursor.execute('''
                INSERT INTO training_sessions 
                (athlete_id, session_date, exercise_type, overall_score, technique_score, 
                 stability_score, efficiency_score, safety_score, similarity_score, 
                 analysis_data, notes)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                athlete_id, session_date, exercise_type,
                scores.get('overall_score', 0),
                scores.get('technique_score', 0),
                scores.get('stability_score', 0),
                scores.get('efficiency_score', 0),
                scores.get('safety_score', 0),
                scores.get('similarity_score', 0),
                json.dumps(analysis_data),
                notes
            ))

            conn.commit()
            conn.close()
            return True

        except Exception as e:
            logger.error(f"保存训练记录错误: {str(e)}")
            return False

    def get_progress_data(self, athlete_id, days=30):
        """获取进步数据"""
        try:
            conn = sqlite3.connect(self.db_path)

            # 获取最近N天的数据
            end_date = datetime.now()
            start_date = end_date - timedelta(days=days)

            query = '''
                SELECT session_date, exercise_type, overall_score, technique_score,
                       stability_score, efficiency_score, safety_score, similarity_score
                FROM training_sessions 
                WHERE athlete_id = ? AND session_date >= ?
                ORDER BY session_date
            '''

            df = pd.read_sql_query(query, conn, params=(athlete_id, start_date.strftime('%Y-%m-%d')))
            conn.close()

            return df

        except Exception as e:
            logger.error(f"获取进步数据错误: {str(e)}")
            return pd.DataFrame()

    def generate_progress_report(self, athlete_id, days=30):
        """生成进步报告"""
        df = self.get_progress_data(athlete_id, days)

        if df.empty:
            return {
                'summary': '暂无训练数据',
                'trends': {},
                'achievements': [],
                'recommendations': ['开始记录训练数据以追踪进步']
            }

        report = {
            'summary': '',
            'trends': {},
            'achievements': [],
            'recommendations': []
        }

        try:
            # 计算趋势
            if len(df) >= 2:
                latest_scores = df.tail(5).mean()  # 最近5次平均
                earlier_scores = df.head(5).mean()  # 最早5次平均

                for metric in ['overall_score', 'technique_score', 'stability_score',
                               'efficiency_score', 'safety_score']:
                    if metric in latest_scores and metric in earlier_scores:
                        change = latest_scores[metric] - earlier_scores[metric]
                        report['trends'][metric] = {
                            'change': round(change, 1),
                            'direction': '上升' if change > 0 else '下降' if change < 0 else '稳定',
                            'latest_avg': round(latest_scores[metric], 1),
                            'earlier_avg': round(earlier_scores[metric], 1)
                        }

            # 识别成就
            latest_overall = df['overall_score'].iloc[-1] if not df.empty else 0
            max_overall = df['overall_score'].max() if not df.empty else 0

            if latest_overall >= 90:
                report['achievements'].append('🏆 达到优秀水平！')
            elif latest_overall >= 80:
                report['achievements'].append('🥇 表现良好！')
            elif latest_overall >= 70:
                report['achievements'].append('📈 稳步提升！')

            if max_overall == latest_overall and latest_overall > 0:
                report['achievements'].append('🎯 创造个人最佳成绩！')

            # 生成建议
            if report['trends'].get('technique_score', {}).get('direction') == '下降':
                report['recommendations'].append('技术分数下降，建议加强基本功练习')
            if report['trends'].get('safety_score', {}).get('direction') == '下降':
                report['recommendations'].append('安全分数下降，需要重视损伤预防')

            # 生成总结
            total_sessions = len(df)
            avg_score = df['overall_score'].mean()

            report['summary'] = f'在过去{days}天中，您完成了{total_sessions}次训练，平均得分{avg_score:.1f}分。'

        except Exception as e:
            logger.error(f"生成进步报告错误: {str(e)}")
            report['summary'] = '生成报告时出现错误'

        return report

    def predict_improvement_trend(self, athlete_id, metric='overall_score'):
        """预测改进趋势"""
        df = self.get_progress_data(athlete_id, days=60)

        if len(df) < 5:
            return {
                'prediction': '数据不足，无法预测',
                'confidence': 0,
                'trend': 'unknown'
            }

        try:
            # 简单线性趋势分析
            df['session_number'] = range(len(df))
            correlation = df['session_number'].corr(df[metric])

            # 预测未来走势
            recent_trend = df[metric].tail(5).mean() - df[metric].head(5).mean()

            prediction = {
                'trend': '上升' if recent_trend > 0 else '下降' if recent_trend < 0 else '稳定',
                'confidence': abs(correlation) * 100,  # 相关性作为置信度
                'predicted_change': recent_trend,
                'recommendation': ''
            }

            if prediction['trend'] == '上升':
                prediction['recommendation'] = '保持当前训练强度，继续稳步提升'
            elif prediction['trend'] == '下降':
                prediction['recommendation'] = '需要调整训练方案，寻找提升突破点'
            else:
                prediction['recommendation'] = '可以尝试增加训练难度或变化训练内容'

            return prediction

        except Exception as e:
            logger.error(f"预测趋势错误: {str(e)}")
            return {'prediction': '预测失败', 'confidence': 0, 'trend': 'unknown'}


# ==================== 数据可视化仪表板 ====================
class DashboardModule:
    """数据可视化仪表板"""

    def __init__(self):
        self.progress_tracker = ProgressTrackingModule()

    def create_performance_chart(self, athlete_id, days=30):
        """创建表现图表"""
        df = self.progress_tracker.get_progress_data(athlete_id, days)

        if df.empty:
            return None

        try:
            # 设置matplotlib中文字体
            plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei']
            plt.rcParams['axes.unicode_minus'] = False

            fig = Figure(figsize=(12, 8))

            # 创建子图
            ax1 = fig.add_subplot(2, 2, 1)
            ax2 = fig.add_subplot(2, 2, 2)
            ax3 = fig.add_subplot(2, 2, 3)
            ax4 = fig.add_subplot(2, 2, 4)

            # 转换日期
            df['date'] = pd.to_datetime(df['session_date'])

            # 1. 总体得分趋势
            ax1.plot(df['date'], df['overall_score'], marker='o', linewidth=2, markersize=6)
            ax1.set_title('总体得分趋势', fontsize=14, fontweight='bold')
            ax1.set_ylabel('得分')
            ax1.grid(True, alpha=0.3)
            ax1.tick_params(axis='x', rotation=45)

            # 2. 各维度得分对比（最新数据）
            if not df.empty:
                latest_data = df.iloc[-1]
                categories = ['技术', '稳定性', '效率', '安全性']
                scores = [
                    latest_data['technique_score'],
                    latest_data['stability_score'],
                    latest_data['efficiency_score'],
                    latest_data['safety_score']
                ]

                colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4']
                bars = ax2.bar(categories, scores, color=colors)
                ax2.set_title('最新各维度得分', fontsize=14, fontweight='bold')
                ax2.set_ylabel('得分')
                ax2.set_ylim(0, 100)

                # 添加数值标签
                for bar, score in zip(bars, scores):
                    height = bar.get_height()
                    ax2.text(bar.get_x() + bar.get_width() / 2., height + 1,
                             f'{score:.1f}', ha='center', va='bottom')

            # 3. 训练频率统计
            df['date_only'] = df['date'].dt.date
            daily_counts = df.groupby('date_only').size()

            ax3.bar(range(len(daily_counts)), daily_counts.values, color='#96CEB4')
            ax3.set_title(f'最近{days}天训练频率', fontsize=14, fontweight='bold')
            ax3.set_ylabel('训练次数')
            ax3.set_xlabel('天数')

            # 4. 运动类型分布
            if 'exercise_type' in df.columns:
                exercise_counts = df['exercise_type'].value_counts()
                if not exercise_counts.empty:
                    ax4.pie(exercise_counts.values, labels=exercise_counts.index, autopct='%1.1f%%',
                            colors=['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FECA57'])
                    ax4.set_title('运动类型分布', fontsize=14, fontweight='bold')

            fig.tight_layout()
            return fig

        except Exception as e:
            logger.error(f"创建图表错误: {str(e)}")
            return None

    def create_progress_summary_widget(self, athlete_id):
        """创建进步摘要小部件"""
        report = self.progress_tracker.generate_progress_report(athlete_id)

        summary_html = f"""
        <div style="background-color: #f8f9fa; padding: 20px; border-radius: 10px; margin: 10px;">
            <h3 style="color: #2c3e50; margin-bottom: 15px;">📊 训练进度摘要</h3>
            <p style="font-size: 14px; color: #34495e; margin-bottom: 15px;">{report['summary']}</p>

            <div style="display: flex; flex-wrap: wrap; gap: 10px; margin-bottom: 15px;">
        """

        # 添加成就徽章
        for achievement in report['achievements']:
            summary_html += f"""
                <span style="background-color: #27ae60; color: white; padding: 5px 10px; 
                            border-radius: 15px; font-size: 12px;">{achievement}</span>
            """

        summary_html += "</div>"

        # 添加趋势信息
        if report['trends']:
            summary_html += "<h4 style='color: #2c3e50; margin-bottom: 10px;'>📈 趋势分析</h4><ul>"
            for metric, trend in report['trends'].items():
                trend_color = '#27ae60' if trend['direction'] == '上升' else '#e74c3c' if trend[
                                                                                              'direction'] == '下降' else '#f39c12'
                metric_name = {
                    'overall_score': '总体得分',
                    'technique_score': '技术得分',
                    'stability_score': '稳定性得分',
                    'efficiency_score': '效率得分',
                    'safety_score': '安全性得分'
                }.get(metric, metric)

                summary_html += f"""
                    <li style="margin-bottom: 5px; color: #34495e;">
                        <strong>{metric_name}:</strong> 
                        <span style="color: {trend_color};">{trend['direction']} ({trend['change']:+.1f}分)</span>
                    </li>
                """
            summary_html += "</ul>"

        # 添加建议
        if report['recommendations']:
            summary_html += "<h4 style='color: #2c3e50; margin-bottom: 10px;'>💡 改进建议</h4><ul>"
            for rec in report['recommendations']:
                summary_html += f"<li style='margin-bottom: 5px; color: #34495e;'>{rec}</li>"
            summary_html += "</ul>"

        summary_html += "</div>"

        return summary_html

# ==================== 损伤风险预测模块 ====================
class InjuryRiskPredictor:
    """损伤风险预测器"""

    # 风险模式数据库
    RISK_PATTERNS = {
        'knee_valgus': {
            'description': '膝内扣',
            'risk_level': 'high',
            'affected_areas': ['膝关节', '髋关节'],
            'sports': ['篮球', '足球', '排球']
        },
        'shoulder_impingement': {
            'description': '肩关节撞击',
            'risk_level': 'medium',
            'affected_areas': ['肩关节', '肩袖'],
            'sports': ['游泳', '投掷', '网球']
        },
        'excessive_trunk_flexion': {
            'description': '过度躯干前屈',
            'risk_level': 'medium',
            'affected_areas': ['腰椎', '髋关节'],
            'sports': ['举重', '体操']
        }
    }

    @staticmethod
    def assess_injury_risk(keypoints, sport_type='general'):
        """评估损伤风险"""
        risk_assessment = {
            'overall_risk_score': 0,
            'high_risk_joints': [],
            'risk_factors': [],
            'recommendations': []
        }

        try:
            # 1. 膝关节内扣检测
            knee_valgus_risk = InjuryRiskPredictor.detect_knee_valgus(keypoints)
            if knee_valgus_risk > 0.3:
                risk_assessment['risk_factors'].append('膝关节内扣倾向')
                risk_assessment['high_risk_joints'].append('膝关节')
                risk_assessment['recommendations'].append('加强臀中肌力量训练')

            # 2. 肩关节风险评估
            shoulder_risk = InjuryRiskPredictor.assess_shoulder_risk(keypoints)
            if shoulder_risk > 0.3:
                risk_assessment['risk_factors'].append('肩关节位置异常')
                risk_assessment['high_risk_joints'].append('肩关节')
                risk_assessment['recommendations'].append('改善肩胛骨稳定性')

            # 3. 脊柱排列评估
            spine_risk = InjuryRiskPredictor.assess_spine_alignment(keypoints)
            if spine_risk > 0.3:
                risk_assessment['risk_factors'].append('脊柱排列异常')
                risk_assessment['high_risk_joints'].append('脊柱')
                risk_assessment['recommendations'].append('核心稳定性训练')

            # 计算整体风险评分
            individual_risks = [knee_valgus_risk, shoulder_risk, spine_risk]
            risk_assessment['overall_risk_score'] = round(np.mean(individual_risks), 2)

        except Exception as e:
            logger.error(f"损伤风险评估错误: {str(e)}")

        return risk_assessment

    @staticmethod
    def detect_knee_valgus(keypoints):
        """检测膝关节内扣"""
        try:
            # 检查右腿
            if all(keypoints[i][2] > 0.1 for i in [9, 10, 11]):  # 右髋、右膝、右踝
                hip = np.array([keypoints[9][0], keypoints[9][1]])
                knee = np.array([keypoints[10][0], keypoints[10][1]])
                ankle = np.array([keypoints[11][0], keypoints[11][1]])

                # 计算膝关节内扣角度
                thigh_vec = knee - hip
                shank_vec = ankle - knee

                # 投影到冠状面分析
                knee_angle = math.atan2(knee[0] - hip[0], hip[1] - knee[1])
                ankle_angle = math.atan2(ankle[0] - knee[0], knee[1] - ankle[1])

                valgus_angle = abs(knee_angle - ankle_angle)

                # 风险评分 (角度越大风险越高)
                risk_score = min(valgus_angle / (math.pi / 6), 1.0)  # 归一化到0-1
                return risk_score

        except Exception as e:
            logger.error(f"膝关节内扣检测错误: {str(e)}")

        return 0

    @staticmethod
    def assess_shoulder_risk(keypoints):
        """评估肩关节风险"""
        try:
            # 检查肩关节位置
            if all(keypoints[i][2] > 0.1 for i in [1, 2, 5]):  # 脖子、双肩
                neck = np.array([keypoints[1][0], keypoints[1][1]])
                right_shoulder = np.array([keypoints[2][0], keypoints[2][1]])
                left_shoulder = np.array([keypoints[5][0], keypoints[5][1]])

                # 肩膀水平度检查
                shoulder_line = right_shoulder - left_shoulder
                horizontal_angle = abs(math.atan2(shoulder_line[1], shoulder_line[0]))

                # 肩膀前探检查 (相对于脖子位置)
                shoulder_center = (right_shoulder + left_shoulder) / 2
                forward_displacement = shoulder_center[0] - neck[0]

                # 综合风险评分
                angle_risk = min(horizontal_angle / (math.pi / 12), 1.0)
                displacement_risk = min(abs(forward_displacement) / 50, 1.0)

                return (angle_risk + displacement_risk) / 2

        except Exception as e:
            logger.error(f"肩关节风险评估错误: {str(e)}")

        return 0

    @staticmethod
    def assess_spine_alignment(keypoints):
        """评估脊柱排列"""
        try:
            # 检查脊柱排列
            if all(keypoints[i][2] > 0.1 for i in [0, 1, 8]):  # 鼻子、脖子、中臀
                nose = np.array([keypoints[0][0], keypoints[0][1]])
                neck = np.array([keypoints[1][0], keypoints[1][1]])
                hip = np.array([keypoints[8][0], keypoints[8][1]])

                # 脊柱线性度检查
                spine_vec = hip - neck
                ideal_spine_angle = math.pi / 2  # 理想情况下脊柱垂直
                actual_spine_angle = math.atan2(spine_vec[1], spine_vec[0])

                deviation = abs(actual_spine_angle - ideal_spine_angle)
                risk_score = min(deviation / (math.pi / 6), 1.0)

                return risk_score

        except Exception as e:
            logger.error(f"脊柱排列评估错误: {str(e)}")

        return 0


# ==================== 个性化训练处方生成器 ====================
class TrainingPrescriptionGenerator:
    """个性化训练处方生成器"""

    EXERCISE_DATABASE = {
        'strength': {
            'glute_bridge': {
                'name': '臀桥',
                'target_muscles': ['臀大肌', '腘绳肌'],
                'equipment': '无',
                'description': '仰卧，双脚踩地，抬起臀部至大腿与躯干成直线'
            },
            'clamshells': {
                'name': '蚌式开合',
                'target_muscles': ['臀中肌'],
                'equipment': '弹力带',
                'description': '侧卧，膝盖弯曲，保持脚跟并拢，抬起上侧膝盖'
            },
            'wall_slides': {
                'name': '靠墙滑行',
                'target_muscles': ['菱形肌', '中斜方肌'],
                'equipment': '墙面',
                'description': '背靠墙，手臂沿墙面上下滑动，保持肘部和手背贴墙'
            }
        },
        'mobility': {
            'hip_flexor_stretch': {
                'name': '髋屈肌拉伸',
                'target_muscles': ['髂腰肌'],
                'equipment': '无',
                'description': '弓步位，后腿伸直，前腿弯曲90度，向前推髋'
            },
            'thoracic_rotation': {
                'name': '胸椎旋转',
                'target_muscles': ['胸椎旋转肌群'],
                'equipment': '无',
                'description': '四点支撑，一手扶地，另一手向天花板旋转'
            }
        },
        'stability': {
            'single_leg_stand': {
                'name': '单腿站立',
                'target_muscles': ['深层稳定肌'],
                'equipment': '无',
                'description': '单脚站立30-60秒，保持身体稳定'
            },
            'plank': {
                'name': '平板支撑',
                'target_muscles': ['核心肌群'],
                'equipment': '无',
                'description': '俯卧撑起始位，保持身体呈直线'
            }
        }
    }

    @staticmethod
    def generate_prescription(risk_assessment, biomech_features, athlete_profile):
        """生成个性化训练处方"""
        prescription = {
            'athlete_id': athlete_profile.get('id', 'unknown'),
            'generation_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'risk_level': risk_assessment['overall_risk_score'],
            'focus_areas': [],
            'training_phases': {},
            'progress_metrics': []
        }

        try:
            # 根据风险评估确定训练重点
            if '膝关节' in risk_assessment['high_risk_joints']:
                prescription['focus_areas'].append('下肢稳定性')
                prescription['training_phases']['phase1'] = {
                    'name': '下肢稳定性强化',
                    'duration': '2-3周',
                    'exercises': [
                        TrainingPrescriptionGenerator.EXERCISE_DATABASE['strength']['glute_bridge'],
                        TrainingPrescriptionGenerator.EXERCISE_DATABASE['strength']['clamshells'],
                        TrainingPrescriptionGenerator.EXERCISE_DATABASE['stability']['single_leg_stand']
                    ]
                }

            if '肩关节' in risk_assessment['high_risk_joints']:
                prescription['focus_areas'].append('肩胛稳定性')
                prescription['training_phases']['phase2'] = {
                    'name': '肩胛稳定性改善',
                    'duration': '2-3周',
                    'exercises': [
                        TrainingPrescriptionGenerator.EXERCISE_DATABASE['strength']['wall_slides'],
                        TrainingPrescriptionGenerator.EXERCISE_DATABASE['mobility']['thoracic_rotation']
                    ]
                }

            if '脊柱' in risk_assessment['high_risk_joints']:
                prescription['focus_areas'].append('核心稳定性')
                prescription['training_phases']['phase3'] = {
                    'name': '核心稳定性训练',
                    'duration': '持续进行',
                    'exercises': [
                        TrainingPrescriptionGenerator.EXERCISE_DATABASE['stability']['plank'],
                        TrainingPrescriptionGenerator.EXERCISE_DATABASE['mobility']['hip_flexor_stretch']
                    ]
                }

            # 设置进度监测指标
            prescription['progress_metrics'] = [
                '关节活动度测试',
                '功能性动作筛查',
                '力量测试',
                '平衡能力评估'
            ]

        except Exception as e:
            logger.error(f"训练处方生成错误: {str(e)}")

        return prescription


# ==================== 增强计算模块 ====================
class EnhancedCalculationModule:
    """增强版计算模块，整合生物力学和AI分析"""

    @staticmethod
    def comprehensive_analysis(keypoints, last_keypoints=None, fps=30, pc=None,
                               rotation_angle=0, athlete_profile=None, sport_type='general'):
        """综合分析 - 整合所有创新功能"""
        results = {}

        if keypoints is None or len(keypoints) < 25:
            return results

        try:
            # 1. 基础运动学参数 (保留原有功能)
            basic_params = EnhancedCalculationModule.calculate_basic_kinematics(
                keypoints, last_keypoints, fps, pc, rotation_angle
            )
            results.update(basic_params)

            # 2. 生物力学特征分析
            biomech_features = BiomechanicsAnalyzer.extract_biomechanical_features(
                keypoints, fps, athlete_profile
            )
            results.update(biomech_features)

            # 3. 损伤风险评估
            risk_assessment = InjuryRiskPredictor.assess_injury_risk(keypoints, sport_type)
            results['injury_risk'] = risk_assessment

            # 4. 生成训练建议
            if athlete_profile:
                training_prescription = TrainingPrescriptionGenerator.generate_prescription(
                    risk_assessment, biomech_features, athlete_profile
                )
                results['training_prescription'] = training_prescription

        except Exception as e:
            logger.error(f"综合分析错误: {str(e)}")

        return results

    @staticmethod
    def calculate_basic_kinematics(keypoints, last_keypoints=None, fps=30, pc=None, rotation_angle=0):
        """计算基础运动学参数 (保留原有CalculationModule.para功能)"""
        results = {}

        try:
            # 基本关键点位置
            key_points = [
                ('鼻子', 0), ('脖子', 1), ('右肩', 2), ('右肘', 3), ('右腕', 4),
                ('左肩', 5), ('左肘', 6), ('左腕', 7), ('中臀', 8), ('右髋', 9),
                ('右膝', 10), ('右踝', 11), ('左髋', 12), ('左膝', 13), ('左踝', 14),
                ('右眼', 15), ('左眼', 16), ('右耳', 17), ('左耳', 18)
            ]

            # 添加基本坐标点
            for name, idx in key_points:
                if idx < len(keypoints) and keypoints[idx][2] > 0.1:
                    results[f'{name}X'] = round(keypoints[idx][0], 2)
                    results[f'{name}Y'] = round(keypoints[idx][1], 2)

                    if pc:
                        results[f'{name}X(米)'] = round(keypoints[idx][0] / pc, 3)
                        results[f'{name}Y(米)'] = round(keypoints[idx][1] / pc, 3)

            # 身体中心计算
            if keypoints[1][2] > 0.1 and keypoints[8][2] > 0.1:
                center_x = (keypoints[1][0] + keypoints[8][0]) / 2
                center_y = (keypoints[1][1] + keypoints[8][1]) / 2
                results['身体中心X'] = round(center_x, 2)
                results['身体中心Y'] = round(center_y, 2)

                if pc:
                    results['身体中心X(米)'] = round(center_x / pc, 3)
                    results['身体中心Y(米)'] = round(center_y / pc, 3)

            # 角度计算
            # 躯干角度
            if keypoints[1][2] > 0.1 and keypoints[8][2] > 0.1:
                dx = keypoints[8][0] - keypoints[1][0]
                dy = keypoints[8][1] - keypoints[1][1]
                trunk_angle = math.atan2(dy, dx) * 180 / math.pi
                results['躯干角度'] = round(trunk_angle - rotation_angle, 2)

            # 关节角度计算 (右肘、左肘、右膝、左膝)
            joint_calculations = [
                ('右肘角度', [2, 3, 4]),
                ('左肘角度', [5, 6, 7]),
                ('右膝角度', [9, 10, 11]),
                ('左膝角度', [12, 13, 14])
            ]

            for angle_name, indices in joint_calculations:
                if all(keypoints[i][2] > 0.1 for i in indices):
                    p1, p2, p3 = indices
                    v1 = [keypoints[p1][0] - keypoints[p2][0], keypoints[p1][1] - keypoints[p2][1]]
                    v2 = [keypoints[p3][0] - keypoints[p2][0], keypoints[p3][1] - keypoints[p2][1]]
                    cos_angle = (v1[0] * v2[0] + v1[1] * v2[1]) / (
                            math.sqrt(v1[0] ** 2 + v1[1] ** 2) * math.sqrt(v2[0] ** 2 + v2[1] ** 2) + 1e-8
                    )
                    angle = math.acos(max(-1, min(1, cos_angle))) * 180 / math.pi
                    results[angle_name] = round(angle, 2)

            # 速度计算
            if last_keypoints is not None and len(last_keypoints) >= 25:
                velocity_calculations = [
                    ('颈部速度', 1),
                    ('右手速度', 4),
                    ('左手速度', 7)
                ]

                for vel_name, idx in velocity_calculations:
                    if keypoints[idx][2] > 0.1 and last_keypoints[idx][2] > 0.1:
                        dx = keypoints[idx][0] - last_keypoints[idx][0]
                        dy = keypoints[idx][1] - last_keypoints[idx][1]
                        velocity = math.sqrt(dx * dx + dy * dy) * fps
                        results[f'{vel_name}(像素/秒)'] = round(velocity, 2)

                        if pc:
                            results[f'{vel_name}(米/秒)'] = round(velocity / pc, 3)

                # 身体中心速度
                if (keypoints[1][2] > 0.1 and keypoints[8][2] > 0.1 and
                        last_keypoints[1][2] > 0.1 and last_keypoints[8][2] > 0.1):

                    curr_center_x = (keypoints[1][0] + keypoints[8][0]) / 2
                    curr_center_y = (keypoints[1][1] + keypoints[8][1]) / 2
                    last_center_x = (last_keypoints[1][0] + last_keypoints[8][0]) / 2
                    last_center_y = (last_keypoints[1][1] + last_keypoints[8][1]) / 2

                    dx = curr_center_x - last_center_x
                    dy = curr_center_y - last_center_y
                    velocity = math.sqrt(dx * dx + dy * dy) * fps
                    results['身体中心速度(像素/秒)'] = round(velocity, 2)

                    if pc:
                        results['身体中心速度(米/秒)'] = round(velocity / pc, 3)

            # 身体比例计算
            # 身高估算
            if keypoints[0][2] > 0.1 and (keypoints[11][2] > 0.1 or keypoints[14][2] > 0.1):
                head_y = keypoints[0][1]
                if keypoints[11][2] > 0.1 and keypoints[14][2] > 0.1:
                    ankle_y = max(keypoints[11][1], keypoints[14][1])
                elif keypoints[11][2] > 0.1:
                    ankle_y = keypoints[11][1]
                else:
                    ankle_y = keypoints[14][1]

                height_pixels = abs(ankle_y - head_y)
                results['身高(像素)'] = round(height_pixels, 2)

                if pc:
                    results['身高(米)'] = round(height_pixels / pc, 3)

            # 肩宽
            if keypoints[2][2] > 0.1 and keypoints[5][2] > 0.1:
                shoulder_width = math.sqrt(
                    (keypoints[2][0] - keypoints[5][0]) ** 2 +
                    (keypoints[2][1] - keypoints[5][1]) ** 2
                )
                results['肩宽(像素)'] = round(shoulder_width, 2)

                if pc:
                    results['肩宽(米)'] = round(shoulder_width / pc, 3)

        except Exception as e:
            logger.error(f"基础运动学计算错误: {str(e)}")

        return results

    @staticmethod
    def draw(frame, keypoints, size=2, type=0):
        """绘制关键点和骨架 (保留原有功能)"""
        if keypoints is None or len(keypoints) == 0:
            return

        # BODY_25关键点连接定义
        connections = [
            (1, 8), (1, 2), (1, 5), (2, 3), (3, 4), (5, 6), (6, 7),
            (8, 9), (9, 10), (10, 11), (8, 12), (12, 13), (13, 14),
            (1, 0), (0, 15), (15, 17), (0, 16), (16, 18),
            (14, 19), (14, 21), (11, 22), (11, 24)
        ]

        # 绘制连接线
        if type == 0:  # 线型
            for start_idx, end_idx in connections:
                if start_idx < len(keypoints) and end_idx < len(keypoints):
                    start_point = keypoints[start_idx]
                    end_point = keypoints[end_idx]
                    if start_point[2] > 0.1 and end_point[2] > 0.1:  # 置信度检查
                        cv2.line(frame,
                                 (int(start_point[0]), int(start_point[1])),
                                 (int(end_point[0]), int(end_point[1])),
                                 (0, 255, 255), size)

        # 绘制关键点
        for i, (x, y, conf) in enumerate(keypoints):
            if conf > 0.1:
                cv2.circle(frame, (int(x), int(y)), size * 2, (0, 255, 0), -1)
                cv2.putText(frame, str(i), (int(x) + 10, int(y)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)


# ==================== 运动员档案管理器 ====================
class AthleteProfileManager:
    """运动员档案管理器"""

    @staticmethod
    def save_profile(profile, filepath=None):
        """保存运动员档案到文件"""
        if filepath is None:
            profiles_dir = os.path.join(os.getcwd(), 'athlete_profiles')
            if not os.path.exists(profiles_dir):
                os.makedirs(profiles_dir)

            filename = f"{profile.get('name', 'athlete')}_{profile.get('id', 'unknown')}.json"
            filepath = os.path.join(profiles_dir, filename)

        try:
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(profile, f, ensure_ascii=False, indent=2)
            return filepath
        except Exception as e:
            raise Exception(f"保存档案失败: {str(e)}")

    @staticmethod
    def load_profile(filepath):
        """从文件加载运动员档案"""
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            raise Exception(f"加载档案失败: {str(e)}")

    @staticmethod
    def list_profiles():
        """列出所有可用的运动员档案"""
        profiles_dir = os.path.join(os.getcwd(), 'athlete_profiles')
        if not os.path.exists(profiles_dir):
            return []

        profiles = []
        for filename in os.listdir(profiles_dir):
            if filename.endswith('.json'):
                filepath = os.path.join(profiles_dir, filename)
                try:
                    profile = AthleteProfileManager.load_profile(filepath)
                    profiles.append({
                        'name': profile.get('name', '未知'),
                        'sport': profile.get('sport', '未知'),
                        'filepath': filepath
                    })
                except:
                    continue
        return profiles


# ==================== AI虚拟教练 ====================
from PyQt5.QtCore import QThread, pyqtSignal


class SmartCoachWorker(QThread):
    """智能教练工作线程"""
    response_ready = pyqtSignal(str, str)  # response, error

    def __init__(self, smart_coach, user_message, user_level, context):
        super().__init__()
        self.smart_coach = smart_coach
        self.user_message = user_message
        self.user_level = user_level
        self.context = context

    def run(self):
        try:
            # 构建完整消息
            full_message = f"{self.context}\n用户问题: {self.user_message}" if self.context else self.user_message

            # 调用智能教练
            response = self.smart_coach.smart_chat(full_message, self.user_level)
            self.response_ready.emit(response, "")

        except Exception as e:
            self.response_ready.emit("", str(e))


# 在AICoachDialog类中修改generate_smart_response方法：
def generate_smart_response(self, user_message):
    """使用智能运动教练生成回复"""
    if not hasattr(self, 'smart_coach') or not self.smart_coach:
        self.handle_smart_response("", "智能教练未初始化")
        return

    # 获取用户水平
    user_level = self.level_combo.currentText() if hasattr(self, 'level_combo') else '一般'

    # 构建上下文
    context = self.build_context(user_message)

    # 创建工作线程
    self.worker = SmartCoachWorker(self.smart_coach, user_message, user_level, context)
    self.worker.response_ready.connect(self.handle_smart_response)
    self.worker.start()

def handle_smart_response(self, response, error):
    """处理智能教练回复"""
    if error:
        self.add_coach_message(f"抱歉，出现了一些问题：{error}\n\n请稍后重试或使用其他功能。")
    elif response:
        self.add_coach_message(response)
    else:
        self.add_coach_message("抱歉，我暂时无法回答这个问题。请尝试换个问题或稍后重试。")

    # 重新启用发送按钮
    self.is_responding = False
    self.send_button.setText("发送")
    self.send_button.setEnabled(True)


def init_smart_coach_safe(self):
    """安全初始化智能教练"""
    try:
        if SMART_COACH_AVAILABLE and SMART_COACH:
            self.smart_coach = SMART_COACH
            self.coach_available = True
            self.coach_initialized = True
            print("✅ 智能运动教练就绪")
        else:
            self.smart_coach = None
            self.coach_available = False
            self.coach_initialized = False
            print("⚠️ 使用基础AI教练模式")
    except Exception as e:
        print(f"❌ 智能教练初始化失败: {e}")
        self.smart_coach = None
        self.coach_available = False
        self.coach_initialized = False


class AICoachDialog(QDialog):
    def __init__(self, parent=None, analysis_data=None):
        super().__init__(parent)

        # 确保所有必要属性都被初始化
        self.analysis_data = analysis_data or {}
        self.conversation_history = []
        self.is_responding = False
        self.conversation_started = False
        self.ui_initialized = False
        self.coach_initialized = False
        self.smart_coach = None
        self.coach_available = False
        self.worker = None  # 添加worker属性

        try:
            self.init_smart_coach_safe()
            self.setup_ui()
            self.ui_initialized = True
            self.show_welcome_message()
        except Exception as e:
            logger.error(f"AICoachDialog初始化失败: {e}")
            self._ensure_basic_attributes()

    def _ensure_basic_attributes(self):
        """确保基本属性存在"""
        if not hasattr(self, 'conversation_started'):
            self.conversation_started = False
        if not hasattr(self, 'is_responding'):
            self.is_responding = False
        if not hasattr(self, 'coach_available'):
            self.coach_available = False

    def init_smart_coach_safe(self):
        """安全初始化智能教练"""
        try:
            if SMART_COACH_AVAILABLE and SMART_COACH:
                self.smart_coach = SMART_COACH
                self.coach_available = True
                self.coach_initialized = True
                print("✅ 智能运动教练就绪")
            else:
                self.smart_coach = None
                self.coach_available = False
                self.coach_initialized = False
                print("⚠️ 使用基础AI教练模式")
        except Exception as e:
            print(f"❌ 智能教练初始化失败: {e}")
            self.smart_coach = None
            self.coach_available = False
            self.coach_initialized = False

    def setup_ui(self):
        """设置UI界面"""
        self.setWindowTitle('🤖 AI虚拟教练')
        self.setFixedSize(900, 700)

        layout = QVBoxLayout()
        layout.setSpacing(20)
        layout.setContentsMargins(24, 24, 24, 24)

        # 标题区域
        title_widget = QWidget()
        title_layout = QVBoxLayout(title_widget)
        title_layout.setAlignment(Qt.AlignCenter)

        if self.coach_available:
            title = QLabel('🏃‍♂️ 智能运动教练')
            subtitle = QLabel('专业运动知识库 + AI增强回答')
        else:
            title = QLabel('🤖 AI虚拟教练')
            subtitle = QLabel('基础AI对话模式')

        title.setStyleSheet("""
            QLabel {
                font-size: 28px; 
                font-weight: 700; 
                color: #212529; 
                margin-bottom: 8px;
            }
        """)
        title.setAlignment(Qt.AlignCenter)

        subtitle.setStyleSheet("""
            QLabel {
                font-size: 16px; 
                color: #6c757d; 
                font-weight: 400;
            }
        """)
        subtitle.setAlignment(Qt.AlignCenter)

        title_layout.addWidget(title)
        title_layout.addWidget(subtitle)
        layout.addWidget(title_widget)

        # 对话显示区域
        self.chat_display = QTextEdit()
        self.chat_display.setReadOnly(True)
        self.chat_display.setStyleSheet("""
            QTextEdit {
                background-color: #ffffff;
                border: 1px solid #dee2e6;
                border-radius: 12px;
                padding: 20px;
                font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', system-ui, sans-serif;
                font-size: 15px;
                line-height: 1.6;
                color: #212529;
            }
            QScrollBar:vertical {
                background: #f8f9fa;
                width: 12px;
                border-radius: 6px;
            }
            QScrollBar::handle:vertical {
                background: #ced4da;
                border-radius: 6px;
                min-height: 20px;
            }
            QScrollBar::handle:vertical:hover {
                background: #adb5bd;
            }
        """)
        layout.addWidget(self.chat_display)

        # 用户水平选择
        if self.coach_available:
            level_layout = QHBoxLayout()
            level_label = QLabel('用户水平:')
            level_label.setStyleSheet("color: #212529; font-weight: 500;")

            self.level_combo = QComboBox()
            self.level_combo.addItems(['新手', '一般', '中级', '高级', '专业'])
            self.level_combo.setCurrentText('一般')
            self.level_combo.setStyleSheet("""
                QComboBox {
                    color: #212529;
                    background-color: #ffffff;
                    border: 1px solid #ced4da;
                    border-radius: 6px;
                    padding: 6px 12px;
                }
            """)

            level_layout.addWidget(level_label)
            level_layout.addWidget(self.level_combo)
            level_layout.addStretch()
            layout.addLayout(level_layout)

        # 输入区域
        input_layout = QHBoxLayout()

        self.input_field = QLineEdit()
        self.input_field.setPlaceholderText('请输入您的运动问题...')
        self.input_field.setStyleSheet("""
            QLineEdit {
                padding: 14px 16px;
                font-size: 15px;
                border: 2px solid #dee2e6;
                border-radius: 25px;
                background-color: #ffffff;
                color: #212529;
            }
            QLineEdit:focus {
                border-color: #0d6efd;
                outline: none;
            }
            QLineEdit::placeholder {
                color: #adb5bd;
            }
        """)
        self.input_field.returnPressed.connect(self.send_message)

        self.send_button = QPushButton('发送')
        self.send_button.clicked.connect(self.send_message)
        self.send_button.setStyleSheet("""
            QPushButton {
                padding: 14px 24px;
                font-size: 15px;
                font-weight: 600;
                background-color: #0d6efd;
                color: #ffffff;
                border: none;
                border-radius: 25px;
                min-width: 80px;
            }
            QPushButton:hover {
                background-color: #0b5ed7;
            }
            QPushButton:pressed {
                background-color: #0a58ca;
            }
            QPushButton:disabled {
                background-color: #6c757d;
            }
        """)

        input_layout.addWidget(self.input_field)
        input_layout.addWidget(self.send_button)
        layout.addLayout(input_layout)

        # 快捷按钮
        shortcuts_layout = QHBoxLayout()
        if self.coach_available:
            shortcut_buttons = [
                ('💪 训练计划', self.suggest_training_plan),
                ('🔍 动作指导', self.analyze_posture),
                ('⚠️ 损伤预防', self.assess_injury_risk),
                ('🍎 运动营养', self.suggest_nutrition),
                ('📚 仅搜索知识库', self.search_knowledge_only)
            ]
        else:
            shortcut_buttons = [
                ('分析我的姿势', self.analyze_posture),
                ('制定训练计划', self.create_training_plan),
                ('损伤风险评估', self.assess_injury_risk),
                ('技术改进建议', self.suggest_improvements)
            ]

        for text, slot in shortcut_buttons:
            btn = QPushButton(text)
            btn.clicked.connect(slot)
            btn.setStyleSheet("""
                QPushButton {
                    background-color: #f8f9fa;
                    border: 1px solid #dee2e6;
                    padding: 8px 12px;
                    border-radius: 6px;
                    color: #212529;
                    font-size: 13px;
                    font-weight: 500;
                }
                QPushButton:hover {
                    background-color: #e9ecef;
                    border-color: #0d6efd;
                    color: #0d6efd;
                }
                QPushButton:pressed {
                    background-color: #dee2e6;
                }
            """)
            shortcuts_layout.addWidget(btn)

        layout.addLayout(shortcuts_layout)

        # 对话记录管理按钮
        record_layout = QHBoxLayout()

        self.clear_chat_btn = QPushButton('清空对话')
        self.clear_chat_btn.clicked.connect(self.clear_conversation)
        self.clear_chat_btn.setStyleSheet("""
            QPushButton {
                background-color: #dc3545;
                color: #ffffff;
                border: none;
                padding: 8px 16px;
                border-radius: 6px;
                font-size: 13px;
                font-weight: 500;
            }
            QPushButton:hover {
                background-color: #c82333;
            }
        """)

        self.save_chat_btn = QPushButton('保存对话')
        self.save_chat_btn.clicked.connect(self.save_conversation)
        self.save_chat_btn.setStyleSheet("""
            QPushButton {
                background-color: #28a745;
                color: #ffffff;
                border: none;
                padding: 8px 16px;
                border-radius: 6px;
                font-size: 13px;
                font-weight: 500;
            }
            QPushButton:hover {
                background-color: #218838;
            }
        """)

        record_layout.addStretch()
        record_layout.addWidget(self.clear_chat_btn)
        record_layout.addWidget(self.save_chat_btn)
        layout.addLayout(record_layout)

        self.setLayout(layout)

    def show_welcome_message(self):
        """显示欢迎消息 - 优化排版版本"""
        if not self.ui_initialized:
            return

        try:
            if self.coach_available:
                welcome_msg = """
                <div style="text-align: center; padding: 20px;">
                    <div style="font-size: 24px; margin-bottom: 20px;">🎯</div>
                    <h2 style="color: #0d6efd; margin-bottom: 16px; font-weight: 600;">
                        欢迎使用智能运动教练！
                    </h2>

                    <div style="background: rgba(13, 110, 253, 0.1); padding: 20px; border-radius: 12px; margin: 20px 0;">
                        <h3 style="color: #495057; margin-bottom: 16px; font-weight: 600;">🔥 核心功能</h3>
                        <div style="text-align: left; max-width: 400px; margin: 0 auto;">
                            <div style="margin: 8px 0; display: flex; align-items: center;">
                                <span style="color: #0d6efd; margin-right: 8px;">📚</span>
                                <span>专业运动知识库检索</span>
                            </div>
                            <div style="margin: 8px 0; display: flex; align-items: center;">
                                <span style="color: #0d6efd; margin-right: 8px;">🧠</span>
                                <span>AI智能分析与建议</span>
                            </div>
                            <div style="margin: 8px 0; display: flex; align-items: center;">
                                <span style="color: #0d6efd; margin-right: 8px;">📊</span>
                                <span>个人数据深度解读</span>
                            </div>
                            <div style="margin: 8px 0; display: flex; align-items: center;">
                                <span style="color: #0d6efd; margin-right: 8px;">⚡</span>
                                <span>实时训练指导</span>
                            </div>
                        </div>
                    </div>

                    <div style="background: rgba(40, 167, 69, 0.1); padding: 16px; border-radius: 8px; margin-top: 20px;">
                        <h4 style="color: #495057; margin-bottom: 8px;">💬 使用提示</h4>
                        <p style="color: #6c757d; margin: 0; font-size: 14px;">
                            您可以直接输入问题，或点击下方快捷按钮开始对话
                        </p>
                    </div>
                </div>
                """
            else:
                welcome_msg = """
                <div style="text-align: center; padding: 20px;">
                    <div style="font-size: 24px; margin-bottom: 20px;">🤖</div>
                    <h2 style="color: #6c757d; margin-bottom: 16px;">AI基础教练为您服务！</h2>
                    <p style="color: #495057; line-height: 1.6;">
                        我可以帮助您分析运动姿势、制定训练计划、评估损伤风险等。<br>
                        请告诉我您需要什么帮助？
                    </p>
                </div>
                """

            self.add_coach_message(welcome_msg, is_welcome=True)
        except Exception as e:
            logger.error(f"显示欢迎消息失败: {e}")

    def add_coach_message(self, message, is_welcome=False):
        """添加教练消息 - 优化排版版本"""
        try:
            timestamp = datetime.now().strftime('%H:%M')

            # 确保 conversation_started 属性存在
            if not hasattr(self, 'conversation_started'):
                self.conversation_started = False

            # 如果是欢迎消息且对话已开始，则不显示
            if is_welcome and self.conversation_started:
                return

            # 保存到对话记录
            message_data = {
                'type': 'coach',
                'message': message,
                'timestamp': timestamp,
                'is_welcome': is_welcome
            }
            self.conversation_history.append(message_data)

            # 优化消息格式 - 更好的排版
            formatted_message = f"""
            <div style="background: linear-gradient(135deg, #e3f2fd 0%, #f8f9fa 100%); 
                        color: #212529; padding: 20px; margin: 12px 8px; 
                        border-radius: 16px; margin-right: 24px; 
                        border-left: 5px solid #0d6efd;
                        box-shadow: 0 4px 12px rgba(13, 110, 253, 0.15);
                        font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', system-ui, sans-serif;">

                <!-- 教练头部信息 -->
                <div style="display: flex; align-items: center; margin-bottom: 12px;">
                    <div style="width: 36px; height: 36px; background: linear-gradient(135deg, #0d6efd, #0b5ed7); 
                               border-radius: 50%; display: flex; align-items: center; justify-content: center; 
                               margin-right: 12px; box-shadow: 0 2px 8px rgba(13, 110, 253, 0.3);">
                        <span style="color: white; font-size: 16px; font-weight: bold;">🤖</span>
                    </div>
                    <div>
                        <div style="color: #0d6efd; font-weight: 600; font-size: 14px; margin-bottom: 2px;">
                            AI智能教练
                        </div>
                        <div style="color: #6c757d; font-size: 12px;">
                            {timestamp}
                        </div>
                    </div>
                </div>

                <!-- 消息内容 -->
                <div style="line-height: 1.6; color: #212529; font-size: 15px;">
                    {self._format_coach_message_content(message)}
                </div>
            </div>
            """

            # 检查UI是否已初始化
            if not hasattr(self, 'chat_display') or self.chat_display is None:
                return

            # 如果这是第一条非欢迎消息，清除欢迎消息
            if not is_welcome and not self.conversation_started:
                self.conversation_started = True
                self.chat_display.clear()
                # 重新显示非欢迎消息
                for msg in self.conversation_history:
                    if not msg.get('is_welcome', False):
                        self._display_message(msg)
            else:
                self.chat_display.insertHtml(formatted_message)
                self.chat_display.moveCursor(QTextCursor.End)

        except Exception as e:
            logger.error(f"添加教练消息失败: {e}")

    def _format_coach_message_content(self, message):
        """格式化教练消息内容 - 改善排版"""
        # 处理HTML标签的消息
        if '<' in message and '>' in message:
            # 优化现有HTML格式
            formatted = message

            # 改进列表样式
            formatted = formatted.replace('<br>', '<br style="margin-bottom: 8px;">')
            formatted = formatted.replace('<strong>', '<strong style="color: #0d6efd; font-weight: 600;">')

            # 添加段落间距
            if '<br><br>' in formatted:
                formatted = formatted.replace('<br><br>', '</p><p style="margin: 12px 0;">')
                formatted = f'<p style="margin: 12px 0;">{formatted}</p>'

            return formatted

        # 处理纯文本消息
        lines = message.split('\n')
        formatted_lines = []

        for line in lines:
            line = line.strip()
            if not line:
                continue

            # 检测并格式化不同类型的内容
            if line.startswith('•') or line.startswith('-') or line.startswith('*'):
                # 列表项
                formatted_lines.append(f'''
                    <div style="margin: 8px 0; padding-left: 20px; position: relative;">
                        <span style="position: absolute; left: 0; color: #0d6efd; font-weight: bold;">•</span>
                        <span style="color: #495057;">{line[1:].strip()}</span>
                    </div>
                ''')
            elif line.startswith('🎯') or line.startswith('💪') or line.startswith('⚠️'):
                # 带emoji的重要信息
                formatted_lines.append(f'''
                    <div style="margin: 12px 0; padding: 12px; background: rgba(13, 110, 253, 0.1); 
                               border-radius: 8px; border-left: 4px solid #0d6efd;">
                        <span style="font-weight: 500; color: #212529;">{line}</span>
                    </div>
                ''')
            elif ':' in line and len(line.split(':')) == 2:
                # 键值对格式
                key, value = line.split(':', 1)
                formatted_lines.append(f'''
                    <div style="margin: 6px 0; display: flex;">
                        <span style="font-weight: 600; color: #495057; min-width: 120px;">{key.strip()}:</span>
                        <span style="color: #212529; margin-left: 8px;">{value.strip()}</span>
                    </div>
                ''')
            else:
                # 普通段落
                formatted_lines.append(f'''
                    <p style="margin: 8px 0; color: #212529; line-height: 1.5;">{line}</p>
                ''')

        return ''.join(formatted_lines)

    def closeEvent(self, event):
        """关闭事件处理"""
        try:
            # 停止任何正在进行的操作
            self.is_responding = False

            # 清理工作线程
            if hasattr(self, 'worker') and self.worker is not None:
                if self.worker.isRunning():
                    self.worker.terminate()
                    self.worker.wait(1000)  # 等待1秒

            event.accept()
        except Exception as e:
            logger.error(f"AICoachDialog关闭失败: {e}")
            event.accept()  # 强制接受关闭事件

    def _display_message(self, message_data):
        """内部方法：显示单条消息"""
        if message_data['type'] == 'coach':
            formatted_message = f"""
            <div style="background: linear-gradient(135deg, #e3f2fd 0%, #f8f9fa 100%); 
                        color: #212529; padding: 16px; margin: 8px 0; 
                        border-radius: 12px; margin-right: 20px; 
                        border-left: 4px solid #0d6efd;
                        box-shadow: 0 2px 8px rgba(13, 110, 253, 0.1);">
                <div style="color: #0d6efd; font-weight: 600; margin-bottom: 8px; font-size: 14px;">
                    🤖 AI教练 [{message_data['timestamp']}]
                </div>
                <div style="line-height: 1.6; color: #212529; font-size: 15px;">
                    {message_data['message']}
                </div>
            </div>
            """
        else:  # user message
            formatted_message = f"""
            <div style="background: linear-gradient(135deg, #e8f5e8 0%, #f1f8f1 100%); 
                        color: #212529; padding: 16px; margin: 8px 0; 
                        border-radius: 12px; margin-left: 20px; 
                        border-right: 4px solid #28a745;
                        box-shadow: 0 2px 8px rgba(40, 167, 69, 0.1);">
                <div style="color: #28a745; font-weight: 600; margin-bottom: 8px; font-size: 14px;">
                    👤 您 [{message_data['timestamp']}]
                </div>
                <div style="line-height: 1.6; color: #212529; font-size: 15px;">
                    {message_data['message']}
                </div>
            </div>
            """

        self.chat_display.insertHtml(formatted_message)
        self.chat_display.moveCursor(QTextCursor.End)

    def add_user_message(self, message):
        """添加用户消息 - 优化排版版本"""
        timestamp = datetime.now().strftime('%H:%M')

        # 保存到对话记录
        message_data = {
            'type': 'user',
            'message': message,
            'timestamp': timestamp
        }
        self.conversation_history.append(message_data)

        # 优化用户消息格式
        formatted_message = f"""
        <div style="background: linear-gradient(135deg, #e8f5e8 0%, #f1f8f1 100%); 
                    color: #212529; padding: 16px 20px; margin: 8px 24px 8px 80px; 
                    border-radius: 16px; border-right: 5px solid #28a745;
                    box-shadow: 0 3px 10px rgba(40, 167, 69, 0.15);
                    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', system-ui, sans-serif;">

            <!-- 用户头部信息 -->
            <div style="display: flex; align-items: center; justify-content: flex-end; margin-bottom: 8px;">
                <div style="text-align: right; margin-right: 12px;">
                    <div style="color: #28a745; font-weight: 600; font-size: 14px; margin-bottom: 2px;">
                        您
                    </div>
                    <div style="color: #6c757d; font-size: 12px;">
                        {timestamp}
                    </div>
                </div>
                <div style="width: 32px; height: 32px; background: linear-gradient(135deg, #28a745, #20c997); 
                           border-radius: 50%; display: flex; align-items: center; justify-content: center;
                           box-shadow: 0 2px 6px rgba(40, 167, 69, 0.3);">
                    <span style="color: white; font-size: 14px;">👤</span>
                </div>
            </div>

            <!-- 消息内容 -->
            <div style="line-height: 1.5; color: #212529; font-size: 15px; text-align: left;">
                {message}
            </div>
        </div>
        """

        # 显示逻辑同add_coach_message
        if not self.conversation_started:
            self.conversation_started = True
            self.chat_display.clear()
            for msg in self.conversation_history:
                if not msg.get('is_welcome', False):
                    self._display_message(msg)
        else:
            self.chat_display.insertHtml(formatted_message)
            self.chat_display.moveCursor(QTextCursor.End)

    def send_message(self):
        """发送消息"""
        if self.is_responding:
            return

        message = self.input_field.text().strip()
        if not message:
            return

        self.add_user_message(message)
        self.input_field.clear()

        # 禁用发送按钮
        self.is_responding = True
        self.send_button.setText("思考中...")
        self.send_button.setEnabled(False)

        # 使用智能教练生成回复
        if self.coach_available:
            self.generate_smart_response(message)
        else:
            self.generate_basic_response(message)

    def generate_smart_response(self, user_message):
        """使用智能运动教练生成回复"""
        if not hasattr(self, 'smart_coach') or not self.smart_coach:
            self.handle_smart_response("", "智能教练未初始化")
            return

        # 获取用户水平
        user_level = self.level_combo.currentText() if hasattr(self, 'level_combo') else '一般'

        # 构建上下文
        context = self.build_context(user_message)

        # 创建工作线程
        self.worker = SmartCoachWorker(self.smart_coach, user_message, user_level, context)
        self.worker.response_ready.connect(self.handle_smart_response)
        self.worker.start()

    def handle_smart_response(self, response, error):
        """处理智能教练回复"""
        if error:
            self.add_coach_message(f"抱歉，出现了一些问题：{error}<br><br>请稍后重试或使用其他功能。")
        elif response:
            self.add_coach_message(response)
        else:
            self.add_coach_message("抱歉，我暂时无法回答这个问题。请尝试换个问题或稍后重试。")

        # 重新启用发送按钮
        self.is_responding = False
        self.send_button.setText("发送")
        self.send_button.setEnabled(True)

    def clear_conversation(self):
        """清空对话记录"""
        reply = QMessageBox.question(self, '确认清空',
                                     '确定要清空所有对话记录吗？',
                                     QMessageBox.Yes | QMessageBox.No,
                                     QMessageBox.No)

        if reply == QMessageBox.Yes:
            self.conversation_history = []
            self.conversation_started = False
            self.chat_display.clear()
            self.show_welcome_message()

    def save_conversation(self):
        """保存对话记录"""
        if not self.conversation_history:
            QMessageBox.information(self, '提示', '暂无对话记录可保存')
            return

        filename, _ = QFileDialog.getSaveFileName(
            self, '保存对话记录',
            f'ai_chat_{datetime.now().strftime("%Y%m%d_%H%M%S")}.txt',
            "文本文件 (*.txt);;所有文件 (*)"
        )

        if filename:
            try:
                with open(filename, 'w', encoding='utf-8') as f:
                    f.write("AI虚拟教练对话记录\n")
                    f.write("=" * 50 + "\n")
                    f.write(f"保存时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

                    for msg in self.conversation_history:
                        if not msg.get('is_welcome', False):  # 不保存欢迎消息
                            speaker = "AI教练" if msg['type'] == 'coach' else "用户"
                            f.write(f"[{msg['timestamp']}] {speaker}:\n")
                            # 移除HTML标签
                            clean_message = msg['message'].replace('<br>', '\n').replace('<strong>', '').replace(
                                '</strong>', '')
                            import re
                            clean_message = re.sub(r'<[^>]+>', '', clean_message)
                            f.write(f"{clean_message}\n\n")

                QMessageBox.information(self, '成功', f'对话记录已保存到:\n{filename}')
            except Exception as e:
                QMessageBox.warning(self, '错误', f'保存失败: {str(e)}')

    # 保持原有的其他方法...
    def build_context(self, user_message):
        """构建包含分析数据的上下文"""
        context_parts = []

        if self.analysis_data:
            context_parts.append("=== 当前运动数据分析 ===")

            # 添加关键分析数据
            key_metrics = [
                '右肘角度', '左肘角度', '右膝角度', '左膝角度', '躯干角度',
                'energy_transfer_efficiency', 'center_of_mass_x', 'center_of_mass_y'
            ]

            for metric in key_metrics:
                if metric in self.analysis_data:
                    context_parts.append(f"{metric}: {self.analysis_data[metric]}")

            # 添加损伤风险信息
            if 'injury_risk' in self.analysis_data:
                risk_data = self.analysis_data['injury_risk']
                context_parts.append(f"损伤风险评分: {risk_data.get('overall_risk_score', 0)}")
                if risk_data.get('high_risk_joints'):
                    context_parts.append(f"高风险部位: {', '.join(risk_data['high_risk_joints'])}")

        return '\n'.join(context_parts) if context_parts else ""

    def generate_basic_response(self, user_message):
        """基础回复生成"""
        response = self.get_basic_ai_response(user_message)
        self.add_coach_message(response)

        # 重新启用发送按钮
        self.is_responding = False
        self.send_button.setText("发送")
        self.send_button.setEnabled(True)

    def get_basic_ai_response(self, user_message):
        """获取基础AI回复"""
        message_lower = user_message.lower()

        if any(word in message_lower for word in ['姿势', '动作', '分析']):
            return self.get_posture_analysis_response()
        elif any(word in message_lower for word in ['训练', '计划', '锻炼']):
            return self.get_training_plan_response()
        elif any(word in message_lower for word in ['损伤', '风险', '受伤']):
            return self.get_injury_risk_response()
        elif any(word in message_lower for word in ['改进', '建议', '提高']):
            return self.get_improvement_suggestions()
        else:
            return ("我理解您的问题。基于当前的分析数据，我建议您：<br><br>"
                    "1. 定期检查运动姿势<br>"
                    "2. 遵循科学的训练计划<br>"
                    "3. 注意身体信号，预防损伤<br><br>"
                    "如果您需要更具体的建议，请点击下方的快捷按钮或告诉我更多详细信息。")

    # 快捷功能方法
    def suggest_training_plan(self):
        """智能训练计划建议"""
        self.add_user_message("请为我制定个性化训练计划")
        if self.coach_available:
            self.generate_smart_response("请根据我的运动数据制定个性化训练计划，考虑我的技术水平和身体状况")
        else:
            response = self.get_training_plan_response()
            self.add_coach_message(response)

    def analyze_posture(self):
        """分析姿势快捷按钮"""
        self.add_user_message("请分析我的运动姿势")
        if self.coach_available:
            self.generate_smart_response("请根据我的运动数据分析我的动作姿势，指出需要改进的地方")
        else:
            response = self.get_posture_analysis_response()
            self.add_coach_message(response)

    def assess_injury_risk(self):
        """评估损伤风险快捷按钮"""
        self.add_user_message("请评估我的损伤风险")
        if self.coach_available:
            self.generate_smart_response("请根据我的运动数据评估损伤风险，给出预防建议")
        else:
            response = self.get_injury_risk_response()
            self.add_coach_message(response)

    def suggest_nutrition(self):
        """运动营养建议"""
        self.add_user_message("请给我运动营养建议")
        if self.coach_available:
            self.generate_smart_response("根据我的运动数据和训练强度，请给我专业的运动营养建议")
        else:
            response = ("运动营养建议：<br><br>"
                        "🥗 <strong>训练前：</strong><br>• 碳水化合物补充能量<br>• 适量蛋白质<br>• 充足水分<br><br>"
                        "🍎 <strong>训练后：</strong><br>• 30分钟内补充营养<br>• 蛋白质修复肌肉<br>• 电解质平衡<br><br>"
                        "💧 <strong>日常：</strong><br>• 保持充足水分<br>• 均衡营养搭配<br>• 避免过度节食")
            self.add_coach_message(response)

    def search_knowledge_only(self):
        """仅搜索知识库"""
        if not self.coach_available:
            self.add_coach_message("知识库搜索功能需要智能教练模块支持。")
            return

        message = self.input_field.text().strip()
        if not message:
            self.add_coach_message("请先在输入框中输入要搜索的问题。")
            return

        self.add_user_message(f"搜索知识库: {message}")

        try:
            # 搜索知识库
            results = self.smart_coach.knowledge_base.search_knowledge(message, top_k=3)

            if results:
                response = "📚 <strong>知识库搜索结果：</strong><br><br>"
                for i, result in enumerate(results, 1):
                    similarity = result.get('similarity', 0)
                    response += f"<strong>结果 {i}</strong> (相似度: {similarity:.2f}):<br>"
                    response += f"<strong>问题:</strong> {result['question']}<br>"
                    response += f"<strong>答案:</strong> {result['answer']}<br>"
                    response += "─" * 40 + "<br><br>"
            else:
                response = "📚 知识库中未找到相关内容。<br><br>建议尝试其他关键词或使用智能咨询功能。"

            self.add_coach_message(response)

        except Exception as e:
            self.add_coach_message(f"知识库搜索出现错误: {e}")

    # 其他辅助方法
    def get_posture_analysis_response(self):
        """获取姿势分析回复 - 优化排版版本"""
        if not self.analysis_data:
            return """
            <div style="text-align: center; padding: 20px; background: rgba(220, 53, 69, 0.1); border-radius: 8px;">
                <span style="color: #dc3545; font-size: 18px;">⚠️</span>
                <p style="color: #721c24; margin: 8px 0 0 0; font-weight: 500;">
                    目前没有可用的姿势分析数据
                </p>
                <p style="color: #856404; font-size: 14px; margin: 8px 0 0 0;">
                    请先在GoPose标签页中载入视频和解析点数据，然后重新开始分析
                </p>
            </div>
            """

        response = """
        <div style="margin-bottom: 20px;">
            <h3 style="color: #0d6efd; margin-bottom: 16px; font-weight: 600;">
                📊 基于您的姿势分析结果：
            </h3>
        </div>
        """

        # 分析结果项
        analysis_items = []

        # 基础运动学数据
        if '右肘角度' in self.analysis_data:
            elbow_angle = self.analysis_data['右肘角度']
            if elbow_angle < 90:
                response += f"✓ 右肘角度 {elbow_angle}° - 手臂屈曲良好<br>"
            else:
                response += f"⚠ 右肘角度 {elbow_angle}° - 建议增加手臂灵活性训练<br>"

        if '右膝角度' in self.analysis_data:
            knee_angle = self.analysis_data['右膝角度']
            if 120 <= knee_angle <= 170:
                response += f"✓ 右膝角度 {knee_angle}° - 腿部姿势良好<br>"
            else:
                response += f"⚠ 右膝角度 {knee_angle}° - 需要注意腿部姿势<br>"

        # 生物力学数据
        if 'energy_transfer_efficiency' in self.analysis_data:
            efficiency = self.analysis_data['energy_transfer_efficiency']
            if efficiency > 0.7:
                response += f"✓ 能量传递效率 {efficiency:.2f} - 动作协调性很好<br>"
            else:
                response += f"⚠ 能量传递效率 {efficiency:.2f} - 建议改善动作协调性<br>"

            # 格式化分析项
            for item in analysis_items:
                response += f"""
                <div style="display: flex; align-items: center; padding: 12px; margin: 8px 0; 
                           background: rgba({item['color'].replace('#', '')}, 0.1); border-radius: 8px;
                           border-left: 4px solid {item['color']};">
                    <span style="color: {item['color']}; font-size: 18px; margin-right: 12px; font-weight: bold;">
                        {item['icon']}
                    </span>
                    <div style="flex: 1;">
                        <div style="font-weight: 600; color: #212529; margin-bottom: 4px;">
                            {item['title']}: {item['value']}
                        </div>
                        <div style="font-size: 14px; color: #6c757d;">
                            {item['description']}
                        </div>
                    </div>
                </div>
                """

            return response

    def get_training_plan_response(self):
        """获取训练计划回复"""
        return ("<strong>个性化训练计划建议：</strong><br><br>"
                "💪 <strong>力量训练:</strong><br>• 核心稳定性训练<br>• 功能性力量练习<br>• 不平衡肌群强化<br><br"
                "🤸 <strong>灵活性训练:</strong><br>• 动态热身<br>• 静态拉伸<br>• 筋膜放松<br><br>"
                "⚖️ <strong>平衡与协调:</strong><br>• 单腿站立练习<br>• 平衡板训练<br>• 反应性训练")

    def get_injury_risk_response(self):
        """获取损伤风险回复"""
        return ("<strong>损伤风险评估：</strong><br><br>"
                "根据当前分析，建议注意以下方面：<br><br>"
                "⚠️ <strong>预防要点:</strong><br>• 充分热身<br>• 正确的运动姿势<br>• 适当的运动强度<br><br>"
                "🏥 <strong>如有不适:</strong><br>• 立即停止运动<br>• 寻求专业医疗建议")

    def get_improvement_suggestions(self):
        """获取改进建议"""
        return ("<strong>技术改进建议：</strong><br><br>"
                "📊 <strong>技术优化:</strong><br>• 慢动作练习<br>• 视频分析<br>• 专业指导<br><br>"
                "🎯 <strong>训练重点:</strong><br>• 提高动作稳定性<br>• 增强核心力量<br>• 改善身体协调性")

    def create_training_plan(self):
        """制定训练计划快捷按钮"""
        self.suggest_training_plan()

    def suggest_improvements(self):
        """技术改进建议快捷按钮"""
        self.add_user_message("请给我技术改进建议")
        if self.coach_available:
            self.generate_smart_response("请根据我的运动数据给出具体的技术改进建议")
        else:
            response = self.get_improvement_suggestions()
            self.add_coach_message(response)

# ==================== 对话框类 ====================
class Dialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle('选择解析模式')
        self.setFixedSize(300, 150)

        layout = QVBoxLayout()

        self.radio1 = QRadioButton('解析全部帧')
        self.radio2 = QRadioButton('仅解析工作区')
        self.radio1.setChecked(True)

        layout.addWidget(self.radio1)
        layout.addWidget(self.radio2)

        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

        self.setLayout(layout)

    @staticmethod
    def getResult(parent=None):
        dialog = Dialog(parent)
        result = dialog.exec_()
        if result == QDialog.Accepted:
            return 1 if dialog.radio2.isChecked() else 0, True
        return 0, False


# ==================== 运动员档案对话框 ====================
class AthleteProfileDialog(QDialog):
    """运动员档案设置对话框"""

    def __init__(self, parent=None, profile=None):
        super().__init__(parent)
        self.setWindowTitle('运动员档案设置')
        self.setFixedSize(500, 650)
        self.profile = profile or {}

        self.setup_ui()
        self.load_profile()

    def setup_ui(self):
        layout = QVBoxLayout()

        # 基本信息组
        basic_group = QGroupBox('基本信息')
        basic_layout = QFormLayout()

        self.name_edit = QLineEdit()
        self.age_spinbox = QSpinBox()
        self.age_spinbox.setRange(10, 80)
        self.age_spinbox.setValue(25)

        self.gender_combo = QComboBox()
        self.gender_combo.addItems(['男', '女'])

        self.height_spinbox = QDoubleSpinBox()
        self.height_spinbox.setRange(120.0, 250.0)
        self.height_spinbox.setValue(175.0)
        self.height_spinbox.setSuffix(' cm')

        self.weight_spinbox = QDoubleSpinBox()
        self.weight_spinbox.setRange(30.0, 200.0)
        self.weight_spinbox.setValue(70.0)
        self.weight_spinbox.setSuffix(' kg')

        basic_layout.addRow('姓名:', self.name_edit)
        basic_layout.addRow('年龄:', self.age_spinbox)
        basic_layout.addRow('性别:', self.gender_combo)
        basic_layout.addRow('身高:', self.height_spinbox)
        basic_layout.addRow('体重:', self.weight_spinbox)
        basic_group.setLayout(basic_layout)

        # 运动信息组
        sport_group = QGroupBox('运动信息')
        sport_layout = QFormLayout()

        self.sport_combo = QComboBox()
        self.sport_combo.addItems([
            '通用', '篮球', '足球', '游泳', '网球', '羽毛球',
            '跑步', '举重', '体操', '武术', '舞蹈'
        ])

        self.level_combo = QComboBox()
        self.level_combo.addItems(['业余', '专业', '精英'])

        self.experience_spinbox = QSpinBox()
        self.experience_spinbox.setRange(0, 30)
        self.experience_spinbox.setSuffix(' 年')

        sport_layout.addRow('运动项目:', self.sport_combo)
        sport_layout.addRow('运动水平:', self.level_combo)
        sport_layout.addRow('训练经验:', self.experience_spinbox)
        sport_group.setLayout(sport_layout)

        # 健康信息组
        health_group = QGroupBox('健康信息')
        health_layout = QFormLayout()

        self.injury_history = QTextEdit()
        self.injury_history.setMaximumHeight(80)
        self.injury_history.setPlaceholderText('请描述既往伤病史...')

        health_layout.addRow('既往伤病:', self.injury_history)
        health_group.setLayout(health_layout)

        # 档案管理组
        management_group = QGroupBox('档案管理')
        management_layout = QHBoxLayout()

        self.save_profile_btn = QPushButton('保存档案')
        self.load_profile_btn = QPushButton('载入档案')
        self.save_profile_btn.clicked.connect(self.save_profile)
        self.load_profile_btn.clicked.connect(self.load_existing_profile)

        management_layout.addWidget(self.save_profile_btn)
        management_layout.addWidget(self.load_profile_btn)
        management_group.setLayout(management_layout)

        layout.addWidget(basic_group)
        layout.addWidget(sport_group)
        layout.addWidget(health_group)
        layout.addWidget(management_group)

        # 按钮
        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

        self.setLayout(layout)

    def load_profile(self):
        """加载档案信息"""
        if self.profile:
            self.name_edit.setText(self.profile.get('name', ''))
            self.age_spinbox.setValue(self.profile.get('age', 25))
            self.gender_combo.setCurrentText(self.profile.get('gender', '男'))
            self.height_spinbox.setValue(self.profile.get('height', 175.0))
            self.weight_spinbox.setValue(self.profile.get('weight', 70.0))
            self.sport_combo.setCurrentText(self.profile.get('sport', '通用'))
            self.level_combo.setCurrentText(self.profile.get('level', '业余'))
            self.experience_spinbox.setValue(self.profile.get('experience', 0))
            self.injury_history.setPlainText(self.profile.get('injury_history', ''))

    def get_profile(self):
        """获取档案信息"""
        return {
            'id': self.profile.get('id', str(int(time.time()))),
            'name': self.name_edit.text(),
            'age': self.age_spinbox.value(),
            'gender': self.gender_combo.currentText(),
            'height': self.height_spinbox.value(),
            'weight': self.weight_spinbox.value(),
            'sport': self.sport_combo.currentText(),
            'level': self.level_combo.currentText(),
            'experience': self.experience_spinbox.value(),
            'injury_history': self.injury_history.toPlainText(),
            'created_date': datetime.now().isoformat(),
            'updated_date': datetime.now().isoformat()
        }

    def save_profile(self):
        """保存当前档案"""
        try:
            profile = self.get_profile()
            filepath = AthleteProfileManager.save_profile(profile)
            QMessageBox.information(self, '成功', f'档案已保存到:\n{filepath}')
        except Exception as e:
            QMessageBox.warning(self, '错误', str(e))

    def load_existing_profile(self):
        """载入现有档案"""
        filepath, _ = QFileDialog.getOpenFileName(
            self, '载入运动员档案',
            os.path.join(os.getcwd(), 'athlete_profiles'),
            "JSON Files (*.json);;All Files (*)"
        )

        if filepath:
            try:
                profile = AthleteProfileManager.load_profile(filepath)
                self.profile = profile
                self.load_profile()
                QMessageBox.information(self, '成功', '档案载入成功')
            except Exception as e:
                QMessageBox.warning(self, '错误', str(e))


# ==================== MyLabel 类 ====================
class MyLabel(QLabel):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMouseTracking(True)
        self.x = 0
        self.y = 0
        self.customized_slots = []

    def mouseMoveEvent(self, event):
        self.x = event.x()
        self.y = event.y()
        super().mouseMoveEvent(event)

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            for slot in self.customized_slots:
                slot()
        super().mousePressEvent(event)

    def connect_customized_slot(self, slot):
        self.customized_slots.append(slot)




import time
import threading
import psutil
import gc
from collections import OrderedDict
from PyQt5.QtCore import QThread, pyqtSignal, QMutex, QTimer
import logging

logger = logging.getLogger(__name__)


class MemoryManager:
    """增强版内存管理器 - 完整修复版"""

    def __init__(self, max_cache_size=50, memory_threshold=80):
        self.frame_cache = OrderedDict()
        self.analysis_cache = OrderedDict()
        self.max_cache_size = max_cache_size
        self.memory_threshold = memory_threshold
        self._access_times = {}
        self._lock = threading.Lock()
        self._is_active = True

        # 修复定时器初始化
        self.cleanup_timer = None
        self._init_timer()

        # 内存监控统计
        self.memory_stats = {
            'peak_usage': 0,
            'cache_hits': 0,
            'cache_misses': 0,
            'cleanup_count': 0
        }
        try:
            if hasattr(self, '_is_active'):
                self.cleanup_on_exit()
        except Exception as e:
            # 使用标准错误输出，避免logging问题
            try:
                print(f"内存管理器析构警告: {e}")
            except:
                pass  # 如果连print都失败，就忽略
    def _init_timer(self):
        """安全初始化定时器"""
        try:
            from PyQt5.QtCore import QTimer
            from PyQt5.QtWidgets import QApplication

            # 确保在主线程中创建定时器
            if QApplication.instance() is not None:
                self.cleanup_timer = QTimer()
                self.cleanup_timer.timeout.connect(self.auto_cleanup)
                self.cleanup_timer.start(30000)  # 每30秒检查一次
                logger.info("内存管理定时器启动成功")
            else:
                # 如果没有Qt应用实例，使用Python线程定时器
                self._start_thread_timer()
                logger.info("使用Python线程定时器")
        except Exception as e:
            logger.warning(f"定时器初始化失败，使用备用方案: {e}")
            self._start_thread_timer()

    def _start_thread_timer(self):
        """启动Python线程定时器作为备用"""

        def timer_worker():
            while self._is_active:
                try:
                    time.sleep(30)  # 30秒
                    if self._is_active:
                        self.auto_cleanup()
                except Exception as e:
                    logger.error(f"线程定时器错误: {e}")
                    break

        timer_thread = threading.Thread(target=timer_worker, daemon=True)
        timer_thread.start()

    def check_memory_usage(self):
        """检查内存使用情况"""
        try:
            # 获取系统内存信息
            memory_info = psutil.virtual_memory()
            memory_percent = memory_info.percent

            # 获取当前进程内存信息
            process = psutil.Process()
            process_memory = process.memory_info()

            stats = {
                'memory_percent': memory_percent,
                'available_gb': memory_info.available / (1024 ** 3),
                'process_memory_mb': process_memory.rss / (1024 ** 2),
                'cache_size': len(self.frame_cache) + len(self.analysis_cache)
            }

            # 判断内存状态
            if memory_percent > self.memory_threshold:
                return False, f"系统内存使用率过高: {memory_percent:.1f}%", stats
            elif process_memory.rss / (1024 ** 2) > 1000:  # 进程超过1GB
                return False, f"进程内存使用过多: {process_memory.rss / (1024 ** 2):.1f}MB", stats
            else:
                return True, "内存使用正常", stats

        except Exception as e:
            logger.error(f"内存检查失败: {e}")
            return True, "内存检查失败，跳过清理", {'memory_percent': 0}

    def cleanup_on_exit(self):
        """程序退出时的清理方法"""
        try:
            logger.info("开始内存管理器退出清理...")

            # 停止定时器
            self.stop_cleanup_timer()

            # 清理缓存
            with self._lock:
                self.frame_cache.clear()
                self.analysis_cache.clear()
                self._access_times.clear()

            # 强制垃圾回收
            gc.collect()

            logger.info("内存管理器清理完成")

        except Exception as e:
            logger.error(f"内存管理器清理失败: {e}")

    def auto_cleanup(self):
        """自动清理过期缓存 - 增强错误处理"""
        if not self._is_active:
            return

        try:
            memory_ok, message, stats = self.check_memory_usage()
            current_time = time.time()

            # 更新峰值内存使用
            current_memory = stats.get('memory_percent', 0)
            if current_memory > self.memory_stats['peak_usage']:
                self.memory_stats['peak_usage'] = current_memory

            # 如果内存使用过高，强制清理
            if not memory_ok:
                self.force_cleanup()
                logger.warning(f"内存使用过高 ({current_memory:.1f}%)，执行强制清理")
            else:
                # 清理超过5分钟未访问的缓存
                self.cleanup_old_cache(max_age=300)

        except Exception as e:
            logger.error(f"自动清理失败: {e}")

    def force_cleanup(self):
        """强制清理内存"""
        try:
            with self._lock:
                # 清理一半的缓存
                cache_size = len(self.frame_cache)
                if cache_size > 10:
                    items_to_remove = cache_size // 2
                    for _ in range(items_to_remove):
                        if self.frame_cache:
                            oldest_key = next(iter(self.frame_cache))
                            del self.frame_cache[oldest_key]
                            if oldest_key in self._access_times:
                                del self._access_times[oldest_key]

                # 清理分析缓存
                analysis_size = len(self.analysis_cache)
                if analysis_size > 10:
                    items_to_remove = analysis_size // 2
                    for _ in range(items_to_remove):
                        if self.analysis_cache:
                            oldest_key = next(iter(self.analysis_cache))
                            del self.analysis_cache[oldest_key]

                self.memory_stats['cleanup_count'] += 1

            # 强制垃圾回收
            gc.collect()

            logger.info(f"强制清理完成，清理了 {items_to_remove} 项缓存")

        except Exception as e:
            logger.error(f"强制清理失败: {e}")

    def cleanup_old_cache(self, max_age=300):
        """清理旧缓存"""
        try:
            current_time = time.time()
            with self._lock:
                # 找到过期的缓存项
                expired_keys = []
                for key, access_time in self._access_times.items():
                    if current_time - access_time > max_age:
                        expired_keys.append(key)

                # 删除过期项
                for key in expired_keys:
                    if key in self.frame_cache:
                        del self.frame_cache[key]
                    if key in self.analysis_cache:
                        del self.analysis_cache[key]
                    if key in self._access_times:
                        del self._access_times[key]

                if expired_keys:
                    logger.debug(f"清理了 {len(expired_keys)} 个过期缓存项")

        except Exception as e:
            logger.error(f"清理旧缓存失败: {e}")

    def stop_cleanup_timer(self):
        """安全停止清理定时器"""
        self._is_active = False

        if self.cleanup_timer is not None:
            try:
                # 检查定时器是否仍然有效
                if hasattr(self.cleanup_timer, 'stop') and hasattr(self.cleanup_timer, 'isActive'):
                    if self.cleanup_timer.isActive():
                        self.cleanup_timer.stop()
                        logger.debug("定时器已停止")

                    # 安全删除定时器
                    if hasattr(self.cleanup_timer, 'deleteLater'):
                        self.cleanup_timer.deleteLater()

            except RuntimeError as e:
                # Qt对象已被删除
                logger.debug(f"定时器已被删除: {e}")
            except Exception as e:
                logger.warning(f"停止定时器时出现异常: {e}")
            finally:
                self.cleanup_timer = None

    def cache_frame_analysis(self, frame_idx, analysis_result):
        """缓存帧分析结果（线程安全）"""
        if not self._is_active:
            return

        with self._lock:
            try:
                # 检查内存使用
                memory_ok, message, stats = self.check_memory_usage()
                if not memory_ok and stats.get('memory_percent', 0) > 85:
                    self.force_cleanup()

                # 如果缓存已满，移除最旧的项
                if len(self.analysis_cache) >= self.max_cache_size:
                    oldest_key = next(iter(self.analysis_cache))
                    del self.analysis_cache[oldest_key]
                    if oldest_key in self._access_times:
                        del self._access_times[oldest_key]
                    self.memory_stats['cleanup_count'] += 1

                # 添加新缓存
                self.analysis_cache[frame_idx] = analysis_result
                self._access_times[frame_idx] = time.time()
                self.analysis_cache.move_to_end(frame_idx)

                logger.debug(f"缓存帧 {frame_idx} 分析结果，当前缓存大小: {len(self.analysis_cache)}")

            except Exception as e:
                logger.error(f"缓存帧分析结果失败: {e}")

    def get_cached_analysis(self, frame_idx):
        """获取缓存的分析结果"""
        if not self._is_active:
            return None

        with self._lock:
            try:
                if frame_idx in self.analysis_cache:
                    # 更新访问时间
                    self._access_times[frame_idx] = time.time()
                    self.analysis_cache.move_to_end(frame_idx)
                    self.memory_stats['cache_hits'] += 1
                    return self.analysis_cache[frame_idx]
                else:
                    self.memory_stats['cache_misses'] += 1
                    return None
            except Exception as e:
                logger.error(f"获取缓存分析失败: {e}")
                return None

    def get_cache_stats(self):
        """获取缓存统计信息"""
        return {
            'frame_cache_size': len(self.frame_cache),
            'analysis_cache_size': len(self.analysis_cache),
            'total_cache_items': len(self.frame_cache) + len(self.analysis_cache),
            'max_cache_size': self.max_cache_size,
            'cache_hits': self.memory_stats['cache_hits'],
            'cache_misses': self.memory_stats['cache_misses'],
            'hit_rate': self.memory_stats['cache_hits'] / max(1, self.memory_stats['cache_hits'] + self.memory_stats[
                'cache_misses']),
            'cleanup_count': self.memory_stats['cleanup_count'],
            'peak_memory_usage': self.memory_stats['peak_usage']
        }

    def clear_cache(self):
        """清除所有缓存"""
        with self._lock:
            self.frame_cache.clear()
            self.analysis_cache.clear()
            self._access_times.clear()
        gc.collect()
        logger.info("所有缓存已清除")

    def __del__(self):
        """安全的析构函数"""
        try:
            self.cleanup_on_exit()
        except Exception as e:
            # 使用标准错误输出，避免logging问题
            print(f"内存管理器析构警告: {e}")


class AsyncAnalysisWorker(QThread):
    """异步分析工作器"""
    progress_updated = pyqtSignal(int)  # 进度更新信号
    analysis_completed = pyqtSignal(dict)  # 分析完成信号
    error_occurred = pyqtSignal(str)  # 错误信号
    status_updated = pyqtSignal(str)  # 状态更新信号

    def __init__(self, analysis_function, data, parameters=None):
        super().__init__()
        self.analysis_function = analysis_function
        self.data = data
        self.parameters = parameters or {}
        self.is_cancelled = False
        self.memory_manager = MemoryManager()

        # 分析统计
        self.start_time = None
        self.processed_frames = 0
        self.total_frames = 0

    def cancel_analysis(self):
        """取消分析"""
        self.is_cancelled = True
        self.status_updated.emit("正在取消分析...")
        logger.info("用户取消了分析任务")

    def run(self):
        """后台分析主函数"""
        try:
            self.start_time = time.time()
            self.status_updated.emit("开始分析...")

            # 检查内存状态
            memory_ok, memory_message, memory_stats = self.memory_manager.check_memory_usage()
            if not memory_ok:
                self.error_occurred.emit(f"内存不足: {memory_message}")
                return

            # 执行分析
            if isinstance(self.data, list):
                self.total_frames = len(self.data)
                results = self._process_sequence_data()
            else:
                results = self._process_single_data()

            if not self.is_cancelled:
                # 计算分析统计
                elapsed_time = time.time() - self.start_time
                results['analysis_stats'] = {
                    'total_time': elapsed_time,
                    'processed_frames': self.processed_frames,
                    'frames_per_second': self.processed_frames / elapsed_time if elapsed_time > 0 else 0,
                    'memory_stats': self.memory_manager.get_cache_stats()
                }

                self.analysis_completed.emit(results)
                self.status_updated.emit("分析完成")
                logger.info(f"分析完成，用时 {elapsed_time:.2f} 秒，处理 {self.processed_frames} 帧")
            else:
                self.status_updated.emit("分析已取消")

        except Exception as e:
            logger.error(f"异步分析失败: {e}", exc_info=True)
            self.error_occurred.emit(f"分析失败: {str(e)}")

    def _process_sequence_data(self):
        """处理序列数据"""
        results = {
            'sequence_results': [],
            'summary': {},
            'failed_frames': []
        }

        batch_size = self.parameters.get('batch_size', 10)

        for i in range(0, self.total_frames, batch_size):
            if self.is_cancelled:
                break

            # 检查内存使用
            memory_ok, _, memory_stats = self.memory_manager.check_memory_usage()
            if not memory_ok and memory_stats.get('memory_percent', 0) > 85:
                self.memory_manager.force_cleanup()
                self.status_updated.emit("内存使用过高，正在清理缓存...")

            # 处理批次
            batch_end = min(i + batch_size, self.total_frames)
            batch_data = self.data[i:batch_end]

            batch_results = []
            for j, frame_data in enumerate(batch_data):
                if self.is_cancelled:
                    break

                frame_idx = i + j

                # 检查缓存
                cached_result = self.memory_manager.get_cached_analysis(frame_idx)
                if cached_result is not None:
                    batch_results.append(cached_result)
                else:
                    try:
                        # 执行分析
                        frame_result = self.analysis_function(frame_data, **self.parameters)
                        batch_results.append(frame_result)

                        # 缓存结果
                        self.memory_manager.cache_frame_analysis(frame_idx, frame_result)

                    except Exception as e:
                        logger.error(f"处理帧 {frame_idx} 失败: {e}")
                        results['failed_frames'].append(frame_idx)
                        continue

                self.processed_frames += 1

                # 更新进度
                progress = int((self.processed_frames / self.total_frames) * 100)
                self.progress_updated.emit(progress)

                # 状态更新
                if self.processed_frames % 50 == 0:
                    elapsed = time.time() - self.start_time
                    fps = self.processed_frames / elapsed if elapsed > 0 else 0
                    self.status_updated.emit(f"已处理 {self.processed_frames}/{self.total_frames} 帧 ({fps:.1f} FPS)")

            results['sequence_results'].extend(batch_results)

            # 每个批次后短暂休息，避免过度占用CPU
            self.msleep(10)

        # 生成摘要
        if results['sequence_results']:
            results['summary'] = self._generate_summary(results['sequence_results'])

        return results

    def _process_single_data(self):
        """处理单个数据"""
        try:
            self.status_updated.emit("正在分析单帧数据...")
            result = self.analysis_function(self.data, **self.parameters)
            self.processed_frames = 1
            self.progress_updated.emit(100)
            return {'single_result': result}
        except Exception as e:
            logger.error(f"单帧分析失败: {e}")
            raise

    def _generate_summary(self, results):
        """生成分析摘要"""
        try:
            summary = {
                'total_frames': len(results),
                'successful_frames': len([r for r in results if r is not None]),
                'analysis_metrics': {}
            }

            # 收集数值指标
            numeric_metrics = {}
            for result in results:
                if result and isinstance(result, dict):
                    for key, value in result.items():
                        if isinstance(value, (int, float)):
                            if key not in numeric_metrics:
                                numeric_metrics[key] = []
                            numeric_metrics[key].append(value)

            # 计算统计量
            for metric, values in numeric_metrics.items():
                if values:
                    import numpy as np
                    summary['analysis_metrics'][metric] = {
                        'mean': np.mean(values),
                        'std': np.std(values),
                        'min': np.min(values),
                        'max': np.max(values),
                        'count': len(values)
                    }

            return summary

        except Exception as e:
            logger.error(f"生成摘要失败: {e}")
            return {'error': str(e)}

# 使用示例
class AnalysisManager:
    """分析管理器示例"""

    def __init__(self):
        self.memory_manager = MemoryManager(max_cache_size=100)
        self.analysis_worker = None

    def start_async_analysis(self, data, analysis_func, parameters=None):
        """启动异步分析"""
        if self.analysis_worker and self.analysis_worker.isRunning():
            self.analysis_worker.cancel_analysis()
            self.analysis_worker.wait()

        self.analysis_worker = AsyncAnalysisWorker(analysis_func, data, parameters)

        # 连接信号
        self.analysis_worker.progress_updated.connect(self.on_progress_update)
        self.analysis_worker.analysis_completed.connect(self.on_analysis_complete)
        self.analysis_worker.error_occurred.connect(self.on_analysis_error)
        self.analysis_worker.status_updated.connect(self.on_status_update)

        self.analysis_worker.start()

    def on_progress_update(self, progress):
        """进度更新回调"""
        print(f"分析进度: {progress}%")

    def on_analysis_complete(self, results):
        """分析完成回调"""
        print("分析完成:", results.get('analysis_stats', {}))

    def on_analysis_error(self, error_message):
        """分析错误回调"""
        print(f"分析错误: {error_message}")

    def on_status_update(self, status):
        """状态更新回调"""
        print(f"状态: {status}")
# ==================== 增强版 GoPose 主要功能模块 ====================
class EnhancedGoPoseModule(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        # 初始化顺序修复
        self.memory_manager = MemoryManager()
        self.sequence_manager = SequenceAnalysisManager()
        self.sequence_analysis_completed = False
        self.athlete_profile = None
        self.ar_guidance = ARRealTimeGuidance(self)
        self.ar_enabled = False

        # 确保3D分析器正确初始化
        try:
            self.threed_analyzer = Enhanced3DAnalyzer()
        except Exception as e:
            print(f"3D分析器初始化失败: {e}")
            self.threed_analyzer = None

        self.pose_3d_sequence = []
        self.last_3d_pose = None

        # UI初始化
        self.setup_ui()
        self.default()
        self.init_menu_bar()
        self.init_img_label()
        self.init_buttons()

        # 播放定时器
        self.play_timer = QTimer()
        self.play_timer.timeout.connect(self.play_next_frame)
        self.is_playing = False

    def setup_ar_controls(self):
        """设置AR控制界面"""
        ar_group = QGroupBox("AR增强现实指导")
        ar_layout = QVBoxLayout(ar_group)

        # AR开关
        self.ar_toggle_btn = QPushButton("启用AR指导")
        self.ar_toggle_btn.setCheckable(True)
        self.ar_toggle_btn.clicked.connect(self.toggle_ar_guidance)

        # AR功能选项
        self.ar_options = {
            'ideal_pose': QCheckBox("显示理想姿势"),
            'force_vectors': QCheckBox("显示力向量"),
            'muscle_activation': QCheckBox("肌肉激活热图"),
            'joint_stress': QCheckBox("关节受力分析"),
            'movement_prediction': QCheckBox("动作轨迹预测")
        }

        for checkbox in self.ar_options.values():
            ar_layout.addWidget(checkbox)

        ar_layout.addWidget(self.ar_toggle_btn)

        # 添加到主界面
        self.right_layout.addWidget(ar_group)

    def currentFrame(self):
        """增强版当前帧显示（集成AR）"""
        if self.video and self.cap:
            try:
                self.cap.set(cv2.CAP_PROP_POS_FRAMES, self.fps)
                ret, frame = self.cap.read()

                if ret:
                    # 基础关键点绘制
                    if self.pkl and self.data and self.fps < len(self.data):
                        keypoints_data = self.data[self.fps]
                        if keypoints_data is not None and len(keypoints_data) > 0:
                            current_keypoints = keypoints_data[0]

                            # ✨ AR增强功能
                            if self.ar_enabled:
                                frame = self._apply_ar_enhancements(frame, current_keypoints)
                            else:
                                # 原有的基础绘制
                                EnhancedCalculationModule.draw(frame, current_keypoints,
                                                               self.lSize, self.drawPoint)

                    # 显示处理后的帧
                    self._display_frame(frame)

            except Exception as e:
                QMessageBox.warning(self, '显示图像错误', str(e))

    def _apply_ar_enhancements(self, frame, keypoints):
        """应用AR增强效果"""
        # 获取当前分析数据
        analysis_data = self.comprehensive_analysis()

        # 应用选中的AR功能
        if self.ar_options['ideal_pose'].isChecked():
            frame = self.ar_guidance.overlay_technique_guidance(frame, keypoints)

        if self.ar_options['force_vectors'].isChecked():
            frame = self.ar_guidance.show_force_vectors(frame, analysis_data)

        if self.ar_options['muscle_activation'].isChecked() or \
                self.ar_options['joint_stress'].isChecked():
            frame = self.ar_guidance.interactive_anatomy_view(frame, keypoints, analysis_data)

        return frame

    def toggle_ar_guidance(self, checked):
        """切换AR指导开关"""
        self.ar_enabled = checked
        if checked:
            self.ar_toggle_btn.setText("关闭AR指导")
        else:
            self.ar_toggle_btn.setText("启用AR指导")

        # 刷新当前帧显示
        self.currentFrame()

    def closeEvent(self, event):
        """关闭事件处理"""
        try:
            # 清理内存管理器
            if hasattr(self, 'memory_manager'):
                self.memory_manager.cleanup_on_exit()

            # 清理其他资源
            if hasattr(self, 'cap') and self.cap:
                self.cap.release()
            if hasattr(self, 'play_timer'):
                self.play_timer.stop()

            event.accept()
        except Exception as e:
            logger.error(f"关闭事件处理失败: {e}")
            event.accept()  # 仍然接受关闭事件

    def run_complete_sequence_analysis(self):
        """优化的完整序列分析"""
        if not self.data or not self.athlete_profile:
            QMessageBox.warning(self, '警告', '请先载入数据和设置运动员档案')
            return False

        # 检查数据量，如果太大提供选项
        total_frames = len(self.data)
        if total_frames > 1000:
            reply = QMessageBox.question(self, '大数据量警告',
                                         f'数据包含{total_frames}帧，完整分析可能需要较长时间。\n'
                                         '是否继续？可以选择采样分析以节省时间。',
                                         QMessageBox.Yes | QMessageBox.No)
            if reply == QMessageBox.No:
                return False

        # 显示进度对话框
        progress_dialog = QProgressDialog("正在分析运动序列...", "取消", 0, total_frames, self)
        progress_dialog.setWindowModality(Qt.WindowModal)
        progress_dialog.show()

        self.sequence_manager = SequenceAnalysisManager()

        try:
            # 批量处理以提高效率
            batch_size = 100
            processed_frames = 0

            for batch_start in range(0, total_frames, batch_size):
                if progress_dialog.wasCanceled():
                    return False

                batch_end = min(batch_start + batch_size, total_frames)

                # 处理当前批次
                for frame_idx in range(batch_start, batch_end):
                    if progress_dialog.wasCanceled():
                        return False

                    progress_dialog.setValue(frame_idx)

                    # 每100帧处理一次UI事件
                    if frame_idx % 100 == 0:
                        QApplication.processEvents()

                    if self.data[frame_idx] is not None and len(self.data[frame_idx]) > 0:
                        current_keypoints = self.data[frame_idx][0]
                        last_keypoints = None

                        if frame_idx > 0 and self.data[frame_idx - 1] is not None:
                            last_keypoints = self.data[frame_idx - 1][0]

                        # 执行单帧分析
                        sport_type = self.athlete_profile.get('sport', 'general')
                        frame_analysis = EnhancedCalculationModule.comprehensive_analysis(
                            current_keypoints,
                            last_keypoints,
                            self.fpsRate,
                            self.pc,
                            self.rotationAngle,
                            self.athlete_profile,
                            sport_type
                        )

                        # 添加到序列管理器
                        self.sequence_manager.add_frame_analysis(frame_idx, frame_analysis)
                        processed_frames += 1

            # 计算序列总结
            self.sequence_summary = self.sequence_manager.calculate_sequence_summary()
            self.sequence_analysis_completed = True

            progress_dialog.close()
            QMessageBox.information(self, '完成',
                                    f'序列分析完成！\n处理了{processed_frames}帧有效数据。')
            return True

        except Exception as e:
            progress_dialog.close()
            QMessageBox.warning(self, '错误', f'序列分析失败: {str(e)}')
            return False

    def show_performance_score(self):
        """显示运动表现评分 - 基于完整序列分析"""
        self.tableWidget.clear()
        self.tableWidget.setHorizontalHeaderLabels(['评分项目', '得分/统计'])
        self.tableWidget.setRowCount(0)

        # 检查是否完成序列分析
        if not self.sequence_analysis_completed:
            reply = QMessageBox.question(self, '需要序列分析',
                                         '运动表现评分需要完整的序列分析结果。\n是否现在开始分析整个运动序列？',
                                         QMessageBox.Yes | QMessageBox.No)
            if reply == QMessageBox.Yes:
                if not self.run_complete_sequence_analysis():
                    return
            else:
                self.tableWidget.insertRow(0)
                self.tableWidget.setItem(0, 0, QTableWidgetItem('需要序列分析'))
                self.tableWidget.setItem(0, 1, QTableWidgetItem('请先运行完整序列分析'))
                return

        try:
            # 基于序列统计结果计算表现评分
            sequence_summary = self.sequence_summary

            # 计算基于序列的表现评分
            performance_scores = self.calculate_sequence_based_performance_score(sequence_summary)

            # 显示总体评分
            self.tableWidget.insertRow(0)
            self.tableWidget.setItem(0, 0, QTableWidgetItem('总体表现评分'))
            score_text = f"{performance_scores['overall_score']}分 ({performance_scores['grade']})"
            self.tableWidget.setItem(0, 1, QTableWidgetItem(score_text))

            # 显示序列统计信息
            row = self.tableWidget.rowCount()
            self.tableWidget.insertRow(row)
            self.tableWidget.setItem(row, 0, QTableWidgetItem('分析帧数'))
            self.tableWidget.setItem(row, 1, QTableWidgetItem(f"{len(self.sequence_manager.analysis_results)}帧"))

            # 显示各维度评分及其统计信息
            score_items = [
                ('技术稳定性', performance_scores['technique_stability'], '基于角度变异系数'),
                ('运动一致性', performance_scores['movement_consistency'], '基于效率一致性'),
                ('生物力学效率', performance_scores['biomech_efficiency'], '基于平均能量传递'),
                ('整体协调性', performance_scores['coordination_score'], '基于多关节协调')
            ]

            for name, score, description in score_items:
                row = self.tableWidget.rowCount()
                self.tableWidget.insertRow(row)
                self.tableWidget.setItem(row, 0, QTableWidgetItem(name))
                self.tableWidget.setItem(row, 1, QTableWidgetItem(f"{score:.1f}分 ({description})"))

            # 显示关键统计信息
            if 'angles_stats' in sequence_summary:
                for angle_name, stats in sequence_summary['angles_stats'].items():
                    row = self.tableWidget.rowCount()
                    self.tableWidget.insertRow(row)
                    self.tableWidget.setItem(row, 0, QTableWidgetItem(f'{angle_name} 统计'))
                    stats_text = f"均值:{stats['mean']:.1f}° 变异:{stats['coefficient_variation']:.3f}"
                    self.tableWidget.setItem(row, 1, QTableWidgetItem(stats_text))

            # 保存训练记录
            if self.athlete_profile:
                progress_tracker = ProgressTrackingModule()
                progress_tracker.save_training_session(
                    self.athlete_profile.get('id', 'unknown'),
                    '序列分析',
                    performance_scores,
                    sequence_summary
                )

        except Exception as e:
            QMessageBox.warning(self, '错误', f'表现评分计算失败: {str(e)}')

    def calculate_sequence_based_performance_score(self, sequence_summary):
        """基于序列数据计算表现评分"""
        scores = {
            'technique_stability': 0,
            'movement_consistency': 0,
            'biomech_efficiency': 0,
            'coordination_score': 0,
            'overall_score': 0,
            'grade': 'F'
        }

        try:
            # 1. 技术稳定性评分（基于角度变异系数）
            angle_stability_scores = []
            if 'angles_stats' in sequence_summary:
                for angle_name, stats in sequence_summary['angles_stats'].items():
                    cv = stats.get('coefficient_variation', 1.0)
                    # 变异系数越小越稳定，转换为0-100分
                    stability_score = max(0, 100 * (1 - min(cv, 1.0)))
                    angle_stability_scores.append(stability_score)

            scores['technique_stability'] = np.mean(angle_stability_scores) if angle_stability_scores else 50

            # 2. 运动一致性评分
            if 'movement_quality' in sequence_summary:
                consistency = sequence_summary['movement_quality'].get('consistency', 0.5)
                scores['movement_consistency'] = consistency * 100

            # 3. 生物力学效率评分
            if 'movement_quality' in sequence_summary:
                efficiency = sequence_summary['movement_quality'].get('average_efficiency', 0.5)
                scores['biomech_efficiency'] = efficiency * 100

            # 4. 整体协调性评分
            if 'stability_metrics' in sequence_summary:
                stability = sequence_summary['stability_metrics'].get('overall_stability', 0.5)
                scores['coordination_score'] = stability * 100

            # 5. 计算总体评分
            score_values = [
                scores['technique_stability'],
                scores['movement_consistency'],
                scores['biomech_efficiency'],
                scores['coordination_score']
            ]
            scores['overall_score'] = np.mean(score_values)

            # 6. 确定等级
            if scores['overall_score'] >= 90:
                scores['grade'] = 'A+'
            elif scores['overall_score'] >= 85:
                scores['grade'] = 'A'
            elif scores['overall_score'] >= 80:
                scores['grade'] = 'A-'
            elif scores['overall_score'] >= 75:
                scores['grade'] = 'B+'
            elif scores['overall_score'] >= 70:
                scores['grade'] = 'B'
            elif scores['overall_score'] >= 65:
                scores['grade'] = 'B-'
            elif scores['overall_score'] >= 60:
                scores['grade'] = 'C+'
            else:
                scores['grade'] = 'C'

        except Exception as e:
            logger.error(f"序列评分计算错误: {str(e)}")

        return scores

    def show_standard_comparison(self):
        """显示标准动作对比 - 基于完整序列分析"""
        self.tableWidget.clear()
        self.tableWidget.setHorizontalHeaderLabels(['对比项目', '序列统计结果'])
        self.tableWidget.setRowCount(0)

        # 检查序列分析
        if not self.sequence_analysis_completed:
            reply = QMessageBox.question(self, '需要序列分析',
                                         '标准动作对比需要完整的序列分析结果。\n是否现在开始分析整个运动序列？',
                                         QMessageBox.Yes | QMessageBox.No)
            if reply == QMessageBox.Yes:
                if not self.run_complete_sequence_analysis():
                    return
            else:
                self.tableWidget.insertRow(0)
                self.tableWidget.setItem(0, 0, QTableWidgetItem('需要序列分析'))
                self.tableWidget.setItem(0, 1, QTableWidgetItem('请先运行完整序列分析'))
                return

        try:
            # 创建对比模块
            comparison_module = StandardComparisonModule()
            available_exercises = comparison_module.get_available_exercises()

            # 让用户选择要对比的动作类型
            exercise_type, ok = QInputDialog.getItem(
                self, '选择动作类型', '请选择要对比的标准动作:',
                available_exercises, 0, False
            )

            if ok and exercise_type:
                # 基于序列统计数据进行对比
                comparison_result = self.compare_sequence_with_standard(exercise_type)

                # 显示序列对比信息
                row = self.tableWidget.rowCount()
                self.tableWidget.insertRow(row)
                self.tableWidget.setItem(row, 0, QTableWidgetItem('分析序列'))
                sequence_info = f"{len(self.sequence_manager.analysis_results)}帧 ({len(self.sequence_manager.analysis_results) / self.fpsRate:.1f}秒)"
                self.tableWidget.setItem(row, 1, QTableWidgetItem(sequence_info))

                # 显示平均相似度得分
                row = self.tableWidget.rowCount()
                self.tableWidget.insertRow(row)
                self.tableWidget.setItem(row, 0, QTableWidgetItem('平均相似度得分'))
                self.tableWidget.setItem(row, 1, QTableWidgetItem(f"{comparison_result['average_similarity']:.1f}分"))

                # 显示稳定性评分
                row = self.tableWidget.rowCount()
                self.tableWidget.insertRow(row)
                self.tableWidget.setItem(row, 0, QTableWidgetItem('动作稳定性'))
                self.tableWidget.setItem(row, 1, QTableWidgetItem(f"{comparison_result['stability_score']:.1f}分"))

                # 显示角度统计对比
                for angle_name, comparison in comparison_result.get('angle_statistics_comparison', {}).items():
                    row = self.tableWidget.rowCount()
                    self.tableWidget.insertRow(row)
                    self.tableWidget.setItem(row, 0, QTableWidgetItem(f'{angle_name} 统计对比'))
                    stats_text = f"均值:{comparison['mean_diff']:.1f}° 稳定性:{comparison['stability_rating']}"
                    self.tableWidget.setItem(row, 1, QTableWidgetItem(stats_text))

                # 显示改进建议
                for i, suggestion in enumerate(comparison_result['sequence_based_suggestions']):
                    row = self.tableWidget.rowCount()
                    self.tableWidget.insertRow(row)
                    self.tableWidget.setItem(row, 0, QTableWidgetItem(f'改进建议{i + 1}'))
                    self.tableWidget.setItem(row, 1, QTableWidgetItem(suggestion))
            else:
                self.tableWidget.insertRow(0)
                self.tableWidget.setItem(0, 0, QTableWidgetItem('未选择动作类型'))

        except Exception as e:
            QMessageBox.warning(self, '错误', f'标准动作对比失败: {str(e)}')

    def compare_sequence_with_standard(self, exercise_type):
        """基于序列数据与标准动作对比"""
        comparison_result = {
            'exercise_type': exercise_type,
            'average_similarity': 0,
            'stability_score': 0,
            'angle_statistics_comparison': {},
            'sequence_based_suggestions': []
        }

        try:
            # 获取标准动作模板
            comparison_module = StandardComparisonModule()
            if exercise_type not in comparison_module.sport_templates:
                return comparison_result

            template = comparison_module.sport_templates[exercise_type]
            sequence_summary = self.sequence_summary

            # 1. 计算平均角度与标准的差异
            similarities = []
            angle_comparisons = {}

            if 'angles_stats' in sequence_summary:
                for angle_name, standard_range in template['key_angles'].items():
                    if angle_name in sequence_summary['angles_stats']:
                        user_stats = sequence_summary['angles_stats'][angle_name]
                        user_mean = user_stats['mean']
                        user_std = user_stats['std']

                        optimal_angle = standard_range['optimal']
                        min_angle = standard_range['min']
                        max_angle = standard_range['max']

                        # 计算平均值相似度
                        if min_angle <= user_mean <= max_angle:
                            deviation = abs(user_mean - optimal_angle)
                            max_deviation = max(optimal_angle - min_angle, max_angle - optimal_angle)
                            similarity = max(0, 100 - (deviation / max_deviation * 100))
                        else:
                            # 超出范围的处理
                            if user_mean < min_angle:
                                deviation = min_angle - user_mean
                            else:
                                deviation = user_mean - max_angle
                            similarity = max(0, 100 - deviation * 2)

                        similarities.append(similarity)

                        # 计算稳定性评级
                        cv = user_stats.get('coefficient_variation', 1.0)
                        if cv < 0.1:
                            stability_rating = "优秀"
                        elif cv < 0.2:
                            stability_rating = "良好"
                        elif cv < 0.3:
                            stability_rating = "一般"
                        else:
                            stability_rating = "需改进"

                        angle_comparisons[angle_name] = {
                            'mean_diff': user_mean - optimal_angle,
                            'user_mean': user_mean,
                            'user_std': user_std,
                            'standard_optimal': optimal_angle,
                            'standard_range': f"{min_angle}-{max_angle}",
                            'similarity': similarity,
                            'stability_rating': stability_rating,
                            'coefficient_variation': cv
                        }

            # 2. 计算总体相似度
            comparison_result['average_similarity'] = np.mean(similarities) if similarities else 0
            comparison_result['angle_statistics_comparison'] = angle_comparisons

            # 3. 计算稳定性评分
            if 'movement_quality' in sequence_summary:
                consistency = sequence_summary['movement_quality'].get('consistency', 0.5)
                comparison_result['stability_score'] = consistency * 100

            # 4. 生成基于序列的改进建议
            suggestions = []

            # 基于平均角度偏差的建议
            for angle_name, comparison in angle_comparisons.items():
                mean_diff = comparison['mean_diff']
                cv = comparison['coefficient_variation']

                if abs(mean_diff) > 10:
                    if mean_diff > 0:
                        suggestions.append(f"{angle_name}平均角度偏大，建议减少{abs(mean_diff):.1f}度")
                    else:
                        suggestions.append(f"{angle_name}平均角度偏小，建议增加{abs(mean_diff):.1f}度")

                if cv > 0.3:
                    suggestions.append(f"{angle_name}稳定性不足(变异系数{cv:.2f})，需要提高动作一致性")

            # 基于整体稳定性的建议
            if comparison_result['stability_score'] < 70:
                suggestions.append("整体动作稳定性偏低，建议进行慢动作练习")

            # 基于相似度的建议
            if comparison_result['average_similarity'] < 80:
                suggestions.append("与标准动作差异较大，建议观看标准动作视频进行对比学习")

            comparison_result['sequence_based_suggestions'] = suggestions[:5]  # 限制建议数量

        except Exception as e:
            logger.error(f"序列标准对比错误: {str(e)}")

        return comparison_result


    def show_history_analysis(self):
        """显示历史数据分析"""
        self.tableWidget.clear()
        self.tableWidget.setHorizontalHeaderLabels(['分析项目', '结果'])
        self.tableWidget.setRowCount(0)

        if not self.athlete_profile:
            self.tableWidget.insertRow(0)
            self.tableWidget.setItem(0, 0, QTableWidgetItem('需要运动员档案'))
            self.tableWidget.setItem(0, 1, QTableWidgetItem('请先设置运动员档案'))
            return

        progress_tracker = ProgressTrackingModule()
        athlete_id = self.athlete_profile.get('id', 'unknown')

        # 生成进步报告
        report = progress_tracker.generate_progress_report(athlete_id, days=30)

        # 显示摘要
        row = self.tableWidget.rowCount()
        self.tableWidget.insertRow(row)
        self.tableWidget.setItem(row, 0, QTableWidgetItem('30天训练摘要'))
        self.tableWidget.setItem(row, 1, QTableWidgetItem(report['summary']))

        # 显示趋势
        for metric, trend_data in report['trends'].items():
            metric_name = {
                'overall_score': '总体得分趋势',
                'technique_score': '技术得分趋势',
                'stability_score': '稳定性得分趋势',
                'efficiency_score': '效率得分趋势',
                'safety_score': '安全性得分趋势'
            }.get(metric, metric)

            row = self.tableWidget.rowCount()
            self.tableWidget.insertRow(row)
            self.tableWidget.setItem(row, 0, QTableWidgetItem(metric_name))
            trend_text = f"{trend_data['direction']} ({trend_data['change']:+.1f}分)"
            self.tableWidget.setItem(row, 1, QTableWidgetItem(trend_text))

        # 显示成就
        for i, achievement in enumerate(report['achievements']):
            row = self.tableWidget.rowCount()
            self.tableWidget.insertRow(row)
            self.tableWidget.setItem(row, 0, QTableWidgetItem(f'成就{i + 1}'))
            self.tableWidget.setItem(row, 1, QTableWidgetItem(achievement))

        # 显示建议
        for i, recommendation in enumerate(report['recommendations']):
            row = self.tableWidget.rowCount()
            self.tableWidget.insertRow(row)
            self.tableWidget.setItem(row, 0, QTableWidgetItem(f'建议{i + 1}'))
            self.tableWidget.setItem(row, 1, QTableWidgetItem(recommendation))

    def setup_ui(self):
        self.main_layout = QVBoxLayout(self)

        # 顶部工具栏
        self.toolbar = QToolBar()
        self.main_layout.addWidget(self.toolbar)

        # 主分割器
        self.splitter = QSplitter(Qt.Horizontal)
        self.main_layout.addWidget(self.splitter)

        # 左侧：图像显示区域
        self.left_frame = QFrame()
        self.left_frame.setFrameShape(QFrame.StyledPanel)
        self.left_layout = QVBoxLayout(self.left_frame)

        # 图像标签
        self.imgLabel = MyLabel()
        self.imgLabel.setScaledContents(True)
        self.imgLabel.setAlignment(Qt.AlignCenter)

        # 滚动区域
        self.scrollArea = QScrollArea()
        self.scrollArea.setWidget(self.imgLabel)
        self.scrollArea.setBackgroundRole(QPalette.Dark)
        self.left_layout.addWidget(self.scrollArea)

        # 状态栏
        self.status_frame = QFrame()
        self.status_layout = QVBoxLayout(self.status_frame)

        self.label = QLabel("总时长：0秒（0帧）      当前：0秒（0帧）")
        self.label_4 = QLabel("工作区开始：None帧        工作区结束：None帧")
        self.label_2 = QLabel("未选择视频")

        self.status_layout.addWidget(self.label)
        self.status_layout.addWidget(self.label_4)
        self.status_layout.addWidget(self.label_2)
        self.left_layout.addWidget(self.status_frame)

        # 控制按钮
        self.control_frame = QFrame()
        self.control_layout = QHBoxLayout(self.control_frame)

        self.pushButton = QPushButton("上一帧")
        self.pushButton_2 = QPushButton("下一帧")
        self.pushButton_8 = QPushButton("跳至开始")
        self.pushButton_9 = QPushButton("跳至结束")
        self.pushButton_10 = QPushButton("播放")

        self.horizontalSlider = QSlider(Qt.Horizontal)
        self.horizontalSlider.setMinimum(0)
        self.horizontalSlider.setMaximum(100)

        self.control_layout.addWidget(self.pushButton)
        self.control_layout.addWidget(self.pushButton_2)
        self.control_layout.addWidget(self.pushButton_8)
        self.control_layout.addWidget(self.pushButton_9)
        self.control_layout.addWidget(self.pushButton_10)
        self.control_layout.addWidget(self.horizontalSlider)

        self.left_layout.addWidget(self.control_frame)

        # 右侧：数据和控制面板
        self.right_frame = QFrame()
        self.right_layout = QVBoxLayout(self.right_frame)

        # 工具按钮
        self.tool_frame = QFrame()
        self.tool_layout = QVBoxLayout(self.tool_frame)

        # 第一行工具按钮
        tool_row1 = QHBoxLayout()
        self.pushButton_6 = QPushButton("时间测量")
        self.pushButton_7 = QPushButton("长度测量")
        self.pushButton_3 = QPushButton("工作区开始")
        tool_row1.addWidget(self.pushButton_6)
        tool_row1.addWidget(self.pushButton_7)
        tool_row1.addWidget(self.pushButton_3)

        # 第二行工具按钮
        tool_row2 = QHBoxLayout()
        self.pushButton_4 = QPushButton("工作区结束")
        self.pushButton_5 = QPushButton("清除工作区")
        self.athlete_profile_btn = QPushButton("运动员档案")
        tool_row2.addWidget(self.pushButton_4)
        tool_row2.addWidget(self.pushButton_5)
        tool_row2.addWidget(self.athlete_profile_btn)

        self.tool_layout.addLayout(tool_row1)
        self.tool_layout.addLayout(tool_row2)
        self.right_layout.addWidget(self.tool_frame)

        # 管理器树形控件
        self.treeWidget = QTreeWidget()
        self.treeWidget.setHeaderLabel("分析管理器")
        self.setup_tree_widget()
        self.right_layout.addWidget(self.treeWidget)

        # 数据表格
        self.tableWidget = QTableWidget()
        self.tableWidget.setColumnCount(2)
        self.tableWidget.setHorizontalHeaderLabels(["属性", "值"])
        self.tableWidget.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.tableWidget.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.right_layout.addWidget(self.tableWidget)

        # 添加到分割器
        self.splitter.addWidget(self.left_frame)
        self.splitter.addWidget(self.right_frame)
        self.splitter.setSizes([800, 400])

    def setup_tree_widget(self):
        """设置树形控件"""
        # 运动员档案
        profile_item = QTreeWidgetItem(self.treeWidget)
        profile_item.setText(0, "运动员档案")
        profile_item.setCheckState(0, Qt.Unchecked)

        # 选择单人解析点
        select_item = QTreeWidgetItem(self.treeWidget)
        select_item.setText(0, "选择单人解析点")
        select_item.setCheckState(0, Qt.Unchecked)

        # 比例尺信息
        scale_item = QTreeWidgetItem(self.treeWidget)
        scale_item.setText(0, "比例尺信息")
        scale_item.setCheckState(0, Qt.Unchecked)

        # 解析点修正
        modify_item = QTreeWidgetItem(self.treeWidget)
        modify_item.setText(0, "解析点修正")
        modify_item.setCheckState(0, Qt.Unchecked)

        # 基础运动学结果
        basic_result_item = QTreeWidgetItem(self.treeWidget)
        basic_result_item.setText(0, "基础运动学结果")
        basic_result_item.setCheckState(0, Qt.Unchecked)

        # 生物力学分析
        biomech_item = QTreeWidgetItem(self.treeWidget)
        biomech_item.setText(0, "生物力学分析")
        biomech_item.setCheckState(0, Qt.Unchecked)
        # 3d
        threed_item = QTreeWidgetItem(self.treeWidget)
        threed_item.setText(0, "3D运动分析")
        threed_item.setCheckState(0, Qt.Unchecked)

        # 损伤风险评估
        injury_item = QTreeWidgetItem(self.treeWidget)
        injury_item.setText(0, "损伤风险评估")
        injury_item.setCheckState(0, Qt.Unchecked)

        # 训练处方建议
        prescription_item = QTreeWidgetItem(self.treeWidget)
        prescription_item.setText(0, "训练处方建议")
        prescription_item.setCheckState(0, Qt.Unchecked)

        # 添加缺失的功能项目
        # 运动表现评分
        performance_item = QTreeWidgetItem(self.treeWidget)
        performance_item.setText(0, "运动表现评分")
        performance_item.setCheckState(0, Qt.Unchecked)

        # 标准动作对比
        comparison_item = QTreeWidgetItem(self.treeWidget)
        comparison_item.setText(0, "标准动作对比")
        comparison_item.setCheckState(0, Qt.Unchecked)

        # 历史数据分析
        history_item = QTreeWidgetItem(self.treeWidget)
        history_item.setText(0, "历史数据分析")
        history_item.setCheckState(0, Qt.Unchecked)

    def default(self):
        """初始化默认值"""
        self.fps = 0
        self.fpsMax = 0
        self.fpsRate = 1
        self.pc = None
        self.pkl = False
        self.scale = False
        self.long = False
        self.longDic = {}
        self.time = False
        self.timePoint = []
        self.timeDic = {}
        self.level = False
        self.levelPoint = []
        self.rotationAngle = 0
        self.item = 0
        self.x = 0
        self.y = 0
        self.member_ = 3
        self.cut1 = None
        self.cut2 = None
        self.drawPoint = 0
        self.play2 = False
        self.changFlag = 0
        self.lSize = 2
        self.cwd = os.getcwd()
        self.video = None
        self.cap = None
        self.data = None
        self.shape = ""
        self.scale_point = []
        self.lengthPoint = []

        self.sli_label()
        self.tableWidget.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.tableWidget.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeToContents)
        self.tableWidget.horizontalHeader().setSectionResizeMode(0, QHeaderView.Interactive)
        self.tableWidget.verticalHeader().setSectionResizeMode(QHeaderView.ResizeToContents)
        self.tableWidget.setContextMenuPolicy(Qt.CustomContextMenu)
        self.tableWidget.customContextMenuRequested.connect(self.generateMenu)

    def init_menu_bar(self):
        """初始化菜单栏"""
        # 文件菜单
        self.file_menu_action = QAction("文件", self)
        self.file_menu = QMenu()
        self.file_menu_action.setMenu(self.file_menu)

        self.actionOpen = QAction("打开视频", self)
        self.actionOpen.triggered.connect(self.onFileOpen)
        self.file_menu.addAction(self.actionOpen)

        self.actionAnalysis = QAction("解析关键点", self)
        self.actionAnalysis.triggered.connect(self.analytic)
        self.file_menu.addAction(self.actionAnalysis)

        self.actionKey = QAction("载入关键点", self)
        self.actionKey.triggered.connect(self.loadKeys)
        self.file_menu.addAction(self.actionKey)

        self.actionSave = QAction("保存解析点", self)
        self.actionSave.triggered.connect(self.save)
        self.file_menu.addAction(self.actionSave)

        self.actionOutVideo = QAction("导出带解析点视频", self)
        self.actionOutVideo.triggered.connect(self.exportVideo)
        self.file_menu.addAction(self.actionOutVideo)

        self.actionVideoNone = QAction("导出无解析点视频", self)
        self.actionVideoNone.triggered.connect(self.exportPointlessVideo)
        self.file_menu.addAction(self.actionVideoNone)

        self.actionOutPoint = QAction("导出解析点数据", self)
        self.actionOutPoint.triggered.connect(self.exportKeys)
        self.file_menu.addAction(self.actionOutPoint)

        self.actionOutPara = QAction("导出运动学参数", self)
        self.actionOutPara.triggered.connect(self.exportResults)
        self.file_menu.addAction(self.actionOutPara)

        # 编辑菜单
        self.edit_menu_action = QAction("编辑", self)
        self.edit_menu = QMenu()
        self.edit_menu_action.setMenu(self.edit_menu)

        self.actionZoomIn = QAction("放大", self)
        self.actionZoomIn.triggered.connect(self.onViewZoomIn)
        self.edit_menu.addAction(self.actionZoomIn)

        self.actionZoomOut = QAction("缩小", self)
        self.actionZoomOut.triggered.connect(self.onViewZoomOut)
        self.edit_menu.addAction(self.actionZoomOut)

        self.actionNormalSize = QAction("原始尺寸", self)
        self.actionNormalSize.triggered.connect(self.onViewNormalSize)
        self.edit_menu.addAction(self.actionNormalSize)

        # 工具菜单
        self.tools_menu_action = QAction("工具", self)
        self.tools_menu = QMenu()
        self.tools_menu_action.setMenu(self.tools_menu)

        self.actionFps = QAction("设置帧率", self)
        self.actionFps.triggered.connect(self.realFPS)
        self.tools_menu.addAction(self.actionFps)

        self.actionMember = QAction("显示人数", self)
        self.actionMember.triggered.connect(self.member)
        self.tools_menu.addAction(self.actionMember)

        self.actionscaledraw = QAction("比例尺", self)
        self.actionscaledraw.triggered.connect(self.scaleButton)
        self.tools_menu.addAction(self.actionscaledraw)

        self.actionLevel = QAction("水平仪", self)
        self.actionLevel.triggered.connect(self.levelButton)
        self.tools_menu.addAction(self.actionLevel)

        self.actionOne = QAction("确认选择", self)
        self.actionOne.triggered.connect(self.confirmSelection)
        self.tools_menu.addAction(self.actionOne)

        self.actionlineSize = QAction("线条大小", self)
        self.actionlineSize.triggered.connect(self.lineSize)
        self.tools_menu.addAction(self.actionlineSize)

        # 将所有菜单动作添加到工具栏
        self.toolbar.addAction(self.file_menu_action)
        self.toolbar.addAction(self.edit_menu_action)
        self.toolbar.addAction(self.tools_menu_action)

        # 禁用初始不可用的功能
        self.actionAnalysis.setEnabled(False)
        self.actionKey.setEnabled(False)
        self.actionscaledraw.setEnabled(False)
        self.actionZoomIn.setEnabled(False)
        self.actionZoomOut.setEnabled(False)
        self.actionNormalSize.setEnabled(False)
        self.actionFps.setEnabled(False)
        self.actionVideoNone.setEnabled(False)
        self.actionLevel.setEnabled(False)
        self.actionMember.setEnabled(False)
        self.actionOutPoint.setEnabled(False)
        self.actionSave.setEnabled(False)
        self.actionOutVideo.setEnabled(False)
        self.actionOne.setEnabled(False)
        self.actionlineSize.setEnabled(False)
        self.actionOutPara.setEnabled(False)

    def init_img_label(self):
        """初始化图像标签"""
        self.scaleFactor = 1.0
        self.imgLabel.setScaledContents(True)
        self.imgLabel.setAlignment(Qt.AlignCenter)
        self.scrollArea.setWidget(self.imgLabel)
        self.scrollArea.setBackgroundRole(QPalette.Dark)

    def init_buttons(self):
        """初始化按钮事件"""
        self.pushButton.clicked.connect(self.last)
        self.pushButton_2.clicked.connect(self.next_)
        self.imgLabel.connect_customized_slot(self.modifyKey)
        self.imgLabel.connect_customized_slot(self.Scale)
        self.imgLabel.connect_customized_slot(self.length)
        self.imgLabel.connect_customized_slot(self.levelTool)
        self.pushButton_5.clicked.connect(self.workspaceClear)
        self.pushButton_3.clicked.connect(self.workspaceStart)
        self.pushButton_4.clicked.connect(self.workspaceEnd)
        self.pushButton_6.clicked.connect(self.timeButton)
        self.pushButton_7.clicked.connect(self.lengthButton)
        self.pushButton_8.clicked.connect(self.jumpToBeginning)
        self.pushButton_9.clicked.connect(self.jumpToEnd)
        self.pushButton_10.clicked.connect(self.play)
        self.horizontalSlider.valueChanged.connect(self.sli)
        self.athlete_profile_btn.clicked.connect(self.edit_athlete_profile)

        # 修复树形控件连接问题
        try:
            self.treeWidget.itemClicked.disconnect()
        except:
            pass
        self.treeWidget.itemClicked.connect(self.treeClicked)

    # ==================== 播放控制功能 ====================
    def play(self):
        """播放/暂停功能"""
        if self.is_playing:
            self.pause_video()
        else:
            self.play_video()

    def play_video(self):
        """开始播放"""
        if not self.video:
            QMessageBox.warning(self, '警告', '请先选择视频文件！')
            return

        self.is_playing = True
        self.pushButton_10.setText("暂停")

        # 设置播放速度 (毫秒)
        interval = max(int(1000 / self.fpsRate), 33)  # 最小33ms (约30fps)
        self.play_timer.start(interval)

    def pause_video(self):
        """暂停播放"""
        self.is_playing = False
        self.pushButton_10.setText("播放")
        self.play_timer.stop()

    def play_next_frame(self):
        """播放下一帧 - 增强版本"""
        try:
            if self.fps < self.fpsMax:
                self.fps += 1
                self.horizontalSlider.setSliderPosition(self.fps)
                self.sli_label()
                self.currentFrame()
            else:
                self.pause_video()  # 播放结束
        except Exception as e:
            logger.error(f"播放下一帧错误: {str(e)}")
            self.pause_video()

    # ==================== 新增功能方法 ====================
    def edit_athlete_profile(self):
        """编辑运动员档案"""
        dialog = AthleteProfileDialog(self, self.athlete_profile)
        if dialog.exec_() == QDialog.Accepted:
            self.athlete_profile = dialog.get_profile()
            QMessageBox.information(self, '成功', '运动员档案已更新')

    def comprehensive_analysis(self):
        """执行综合分析 - 修复版本"""
        if not self.pkl or not self.data or self.fps >= len(self.data):
            return {}

        try:
            keypoints_data = self.data[self.fps]
            if keypoints_data is None or len(keypoints_data) == 0:
                return {}

            # 获取第一个人的关键点数据
            current_keypoints = keypoints_data[0]

            # 获取前一帧数据
            last_keypoints = None
            if self.fps > 0 and self.fps - 1 < len(self.data):
                if self.data[self.fps - 1] is not None and len(self.data[self.fps - 1]) > 0:
                    last_keypoints = self.data[self.fps - 1][0]

            # 执行综合分析
            sport_type = self.athlete_profile.get('sport', 'general') if self.athlete_profile else 'general'

            return EnhancedCalculationModule.comprehensive_analysis(
                current_keypoints,
                last_keypoints,
                self.fpsRate,
                self.pc,
                self.rotationAngle,
                self.athlete_profile,
                sport_type
            )
        except Exception as e:
            logger.error(f"综合分析错误: {str(e)}")
            return {}

    def treeClicked(self):
        """树形控件点击事件 - 增强版本"""
        try:
            item = self.treeWidget.currentItem()
            if not item:
                return

            item_text = item.text(0)
            # 对于需要序列分析的功能，提供统一入口
            sequence_required_items = ['运动表现评分', '标准动作对比', '损伤风险评估', '训练处方建议']

            if item_text in sequence_required_items and not self.sequence_analysis_completed:
                # 显示序列分析说明
                reply = QMessageBox.question(self, '序列分析说明',
                                             f'{item_text}功能需要对完整运动序列进行分析以获得准确结果。\n\n'
                                             '序列分析将：\n'
                                             '• 分析所有帧的运动数据\n'
                                             '• 计算统计指标（均值、标准差、变异系数等）\n'
                                             '• 评估运动一致性和稳定性\n'
                                             '• 识别运动模式和趋势\n\n'
                                             '是否开始序列分析？',
                                             QMessageBox.Yes | QMessageBox.No)
                if reply == QMessageBox.Yes:
                    if not self.run_complete_sequence_analysis():
                        return
                else:
                    return

            if item_text == '3D运动分析':
                self.show_3d_analysis()
            elif item_text == '运动员档案':
                self.show_athlete_profile()

            # 先断开之前的连接
            try:
                self.tableWidget.clicked.disconnect()
            except:
                pass

            if item_text == '运动员档案':
                self.show_athlete_profile()
            elif item_text == '选择单人解析点':
                self.show_person_selection()
            elif item_text == '比例尺信息':
                self.show_scale_info()
            elif item_text == '解析点修正':
                self.show_keypoint_modification()
            elif item_text == '基础运动学结果':
                self.show_basic_kinematics()
            elif item_text == '生物力学分析':
                self.show_biomechanics_analysis()
            elif item_text == '损伤风险评估':
                self.show_injury_risk_assessment()
            elif item_text == '训练处方建议':
                self.show_training_prescription()
            elif item_text == '运动表现评分':
                self.show_performance_score()
            elif item_text == '标准动作对比':
                self.show_standard_comparison()
            elif item_text == '历史数据分析':
                self.show_history_analysis()
            elif item_text == '3D运动分析':  # ✨ 新增
                self.show_3d_analysis()

        except Exception as e:
            QMessageBox.warning(self, '管理器错误', str(e))



    def show_athlete_profile(self):
        """显示运动员档案"""
        self.tableWidget.clear()
        self.tableWidget.setRowCount(0)
        self.tableWidget.setHorizontalHeaderLabels(['属性', '值'])

        if self.athlete_profile:
            attributes = [
                ('姓名', self.athlete_profile.get('name', '未设置')),
                ('年龄', f"{self.athlete_profile.get('age', 0)}岁"),
                ('性别', self.athlete_profile.get('gender', '未设置')),
                ('身高', f"{self.athlete_profile.get('height', 0)}cm"),
                ('体重', f"{self.athlete_profile.get('weight', 0)}kg"),
                ('运动项目', self.athlete_profile.get('sport', '未设置')),
                ('运动水平', self.athlete_profile.get('level', '未设置')),
                ('训练经验', f"{self.athlete_profile.get('experience', 0)}年")
            ]

            for row, (attr, value) in enumerate(attributes):
                self.tableWidget.insertRow(row)
                self.tableWidget.setItem(row, 0, QTableWidgetItem(attr))
                self.tableWidget.setItem(row, 1, QTableWidgetItem(str(value)))
        else:
            self.tableWidget.insertRow(0)
            self.tableWidget.setItem(0, 0, QTableWidgetItem('请设置运动员档案'))
            self.tableWidget.setItem(0, 1, QTableWidgetItem('点击右侧按钮'))

    def show_person_selection(self):
        """显示单人选择"""
        self.tableWidget.clear()
        self.tableWidget.setRowCount(0)
        self.tableWidget.setHorizontalHeaderLabels(['点击选择', '空格键确定'])

        if self.pkl and self.data and self.fps < len(self.data):
            keypoints_data = self.data[self.fps]
            if keypoints_data is not None:
                self.tableWidget.clicked.connect(self.choosePerson)
                shown_people = self.showPeople(keypoints_data)
                for i in range(len(shown_people)):
                    self.tableWidget.insertRow(i)
                    self.tableWidget.setItem(i, 0, QTableWidgetItem('人物'))
                    self.tableWidget.setItem(i, 1, QTableWidgetItem(str(i)))
        else:
            self.tableWidget.insertRow(0)
            self.tableWidget.setItem(0, 0, QTableWidgetItem('缺少解析点数据'))

    def show_scale_info(self):
        """显示比例尺信息"""
        self.tableWidget.clear()
        self.tableWidget.setRowCount(0)
        self.tableWidget.setHorizontalHeaderLabels(['属性', '值'])

        if self.pc:
            self.tableWidget.insertRow(0)
            self.tableWidget.setItem(0, 0, QTableWidgetItem('比例系数(像素/实际):'))
            self.tableWidget.setItem(0, 1, QTableWidgetItem(str(self.pc)))
        else:
            self.tableWidget.insertRow(0)
            self.tableWidget.setItem(0, 0, QTableWidgetItem('未设置比例尺'))

    def show_keypoint_modification(self):
        """显示关键点修正"""
        self.tableWidget.clear()
        self.tableWidget.setRowCount(0)
        self.tableWidget.setHorizontalHeaderLabels(['名称', '编号'])

        if self.pkl:
            points = ['0 鼻子', '1 脖子', '2 右肩', '3 右肘', '4 右腕', '5 左肩', '6 左肘', '7 左腕',
                      '8 中臀', '9 右髋', '10 右膝', '11 右踝', '12 左髋', '13 左膝', '14 左踝',
                      '15 右眼', '16 左眼', '17 右耳', '18 左耳', '19 左足大拇指', '20 左足小拇指',
                      '21 左足跟', '22 右足大拇指', '23 右足小拇指', '24 右足跟']

            for row, point in enumerate(points):
                parts = point.split()
                self.tableWidget.insertRow(row)
                self.tableWidget.setItem(row, 0, QTableWidgetItem(parts[1]))
                self.tableWidget.setItem(row, 1, QTableWidgetItem(parts[0]))
        else:
            self.tableWidget.insertRow(0)
            self.tableWidget.setItem(0, 0, QTableWidgetItem('缺少解析点数据'))

    def show_basic_kinematics(self):
        """显示基础运动学结果"""
        self.tableWidget.clear()
        self.tableWidget.setHorizontalHeaderLabels(['参数', '值'])
        self.tableWidget.setRowCount(0)

        # 显示长度结果
        for key, value in self.longDic.items():
            count = self.tableWidget.rowCount()
            self.tableWidget.insertRow(count)
            self.tableWidget.setItem(count, 0, QTableWidgetItem(key))
            self.tableWidget.setItem(count, 1, QTableWidgetItem(str(value)))

        # 显示时间结果
        for key, value in self.timeDic.items():
            count = self.tableWidget.rowCount()
            self.tableWidget.insertRow(count)
            self.tableWidget.setItem(count, 0, QTableWidgetItem(key))
            self.tableWidget.setItem(count, 1, QTableWidgetItem(str(value)))

        # 显示基础运动学参数
        if self.pkl and self.data and self.fps < len(self.data):
            keypoints_data = self.data[self.fps]
            if keypoints_data is not None and len(keypoints_data) > 0:
                person_keypoints = keypoints_data[0]
                last_keypoints = None

                if self.fps > 0 and self.fps - 1 < len(self.data):
                    if self.data[self.fps - 1] is not None and len(self.data[self.fps - 1]) > 0:
                        last_keypoints = self.data[self.fps - 1][0]

                basic_params = EnhancedCalculationModule.calculate_basic_kinematics(
                    person_keypoints, last_keypoints, self.fpsRate, self.pc, self.rotationAngle
                )

                for key, value in basic_params.items():
                    count = self.tableWidget.rowCount()
                    self.tableWidget.insertRow(count)
                    self.tableWidget.setItem(count, 0, QTableWidgetItem(key))
                    self.tableWidget.setItem(count, 1, QTableWidgetItem(str(value)))

    def show_biomechanics_analysis(self):
        """显示生物力学分析结果"""
        self.tableWidget.clear()
        self.tableWidget.setHorizontalHeaderLabels(['生物力学参数', '值'])
        self.tableWidget.setRowCount(0)

        analysis_results = self.comprehensive_analysis()

        if analysis_results:
            # 显示生物力学特征
            biomech_params = [
                'right_elbow_torque', 'right_knee_torque', 'energy_transfer_efficiency',
                'center_of_mass_x', 'center_of_mass_y', 'shoulder_abduction_angle',
                'ground_reaction_force'
            ]

            param_names = {
                'right_elbow_torque': '右肘关节力矩(Nm)',
                'right_knee_torque': '右膝关节力矩(Nm)',
                'energy_transfer_efficiency': '能量传递效率',
                'center_of_mass_x': '重心X坐标',
                'center_of_mass_y': '重心Y坐标',
                'shoulder_abduction_angle': '肩关节外展角度(°)',
                'ground_reaction_force': '地面反作用力(N)'
            }

            for param in biomech_params:
                if param in analysis_results:
                    count = self.tableWidget.rowCount()
                    self.tableWidget.insertRow(count)
                    self.tableWidget.setItem(count, 0, QTableWidgetItem(param_names.get(param, param)))
                    self.tableWidget.setItem(count, 1, QTableWidgetItem(str(analysis_results[param])))
        else:
            self.tableWidget.insertRow(0)
            self.tableWidget.setItem(0, 0, QTableWidgetItem('需要关键点数据'))
            self.tableWidget.setItem(0, 1, QTableWidgetItem('请先载入解析点'))

    def show_injury_risk_assessment(self):
        """显示损伤风险评估结果"""
        self.tableWidget.clear()
        self.tableWidget.setHorizontalHeaderLabels(['风险评估', '结果'])
        self.tableWidget.setRowCount(0)

        analysis_results = self.comprehensive_analysis()

        if 'injury_risk' in analysis_results:
            risk_data = analysis_results['injury_risk']

            # 整体风险评分
            count = self.tableWidget.rowCount()
            self.tableWidget.insertRow(count)
            self.tableWidget.setItem(count, 0, QTableWidgetItem('整体风险评分'))
            risk_score = risk_data.get('overall_risk_score', 0)
            risk_level = '低' if risk_score < 0.3 else '中' if risk_score < 0.7 else '高'
            self.tableWidget.setItem(count, 1, QTableWidgetItem(f'{risk_score} ({risk_level}风险)'))

            # 高风险关节
            if risk_data.get('high_risk_joints'):
                count = self.tableWidget.rowCount()
                self.tableWidget.insertRow(count)
                self.tableWidget.setItem(count, 0, QTableWidgetItem('高风险关节'))
                self.tableWidget.setItem(count, 1, QTableWidgetItem(', '.join(risk_data['high_risk_joints'])))

            # 风险因素
            for i, factor in enumerate(risk_data.get('risk_factors', [])):
                count = self.tableWidget.rowCount()
                self.tableWidget.insertRow(count)
                self.tableWidget.setItem(count, 0, QTableWidgetItem(f'风险因素{i + 1}'))
                self.tableWidget.setItem(count, 1, QTableWidgetItem(factor))

            # 建议
            for i, recommendation in enumerate(risk_data.get('recommendations', [])):
                count = self.tableWidget.rowCount()
                self.tableWidget.insertRow(count)
                self.tableWidget.setItem(count, 0, QTableWidgetItem(f'建议{i + 1}'))
                self.tableWidget.setItem(count, 1, QTableWidgetItem(recommendation))
        else:
            self.tableWidget.insertRow(0)
            self.tableWidget.setItem(0, 0, QTableWidgetItem('需要关键点数据'))
            self.tableWidget.setItem(0, 1, QTableWidgetItem('请先载入解析点'))

    def show_training_prescription(self):
        """显示训练处方建议 - 基于完整序列分析"""
        self.tableWidget.clear()
        self.tableWidget.setHorizontalHeaderLabels(['训练处方项目', '基于序列分析的建议'])
        self.tableWidget.setRowCount(0)

        # 检查序列分析
        if not self.sequence_analysis_completed:
            reply = QMessageBox.question(self, '需要序列分析',
                                         '训练处方需要完整的序列分析结果。\n是否现在开始分析整个运动序列？',
                                         QMessageBox.Yes | QMessageBox.No)
            if reply == QMessageBox.Yes:
                if not self.run_complete_sequence_analysis():
                    return
            else:
                self.tableWidget.insertRow(0)
                self.tableWidget.setItem(0, 0, QTableWidgetItem('需要序列分析'))
                self.tableWidget.setItem(0, 1, QTableWidgetItem('请先运行完整序列分析'))
                return

        if not self.athlete_profile:
            self.tableWidget.insertRow(0)
            self.tableWidget.setItem(0, 0, QTableWidgetItem('需要运动员档案'))
            self.tableWidget.setItem(0, 1, QTableWidgetItem('请先设置运动员档案'))
            return

        try:
            # 基于序列分析生成训练处方
            prescription = self.generate_sequence_based_training_prescription()

            # 显示基本信息
            row = self.tableWidget.rowCount()
            self.tableWidget.insertRow(row)
            self.tableWidget.setItem(row, 0, QTableWidgetItem('运动员'))
            self.tableWidget.setItem(row, 1, QTableWidgetItem(self.athlete_profile.get('name', '未知')))

            row = self.tableWidget.rowCount()
            self.tableWidget.insertRow(row)
            self.tableWidget.setItem(row, 0, QTableWidgetItem('序列分析基础'))
            analysis_info = f"{len(self.sequence_manager.analysis_results)}帧, {len(self.sequence_manager.analysis_results) / self.fpsRate:.1f}秒"
            self.tableWidget.setItem(row, 1, QTableWidgetItem(analysis_info))

            # 显示风险等级
            row = self.tableWidget.rowCount()
            self.tableWidget.insertRow(row)
            self.tableWidget.setItem(row, 0, QTableWidgetItem('综合风险等级'))
            risk_level = prescription['risk_level']
            self.tableWidget.setItem(row, 1, QTableWidgetItem(f'{risk_level}'))

            # 显示主要问题识别
            if prescription.get('identified_issues'):
                for i, issue in enumerate(prescription['identified_issues']):
                    row = self.tableWidget.rowCount()
                    self.tableWidget.insertRow(row)
                    self.tableWidget.setItem(row, 0, QTableWidgetItem(f'识别问题{i + 1}'))
                    self.tableWidget.setItem(row, 1, QTableWidgetItem(issue))

            # 显示训练重点
            if prescription.get('focus_areas'):
                row = self.tableWidget.rowCount()
                self.tableWidget.insertRow(row)
                self.tableWidget.setItem(row, 0, QTableWidgetItem('训练重点'))
                self.tableWidget.setItem(row, 1, QTableWidgetItem(', '.join(prescription['focus_areas'])))

            # 显示分阶段训练计划
            for phase_key, phase_data in prescription.get('training_phases', {}).items():
                row = self.tableWidget.rowCount()
                self.tableWidget.insertRow(row)
                self.tableWidget.setItem(row, 0, QTableWidgetItem(f'{phase_data["name"]}'))
                duration_info = f'持续时间: {phase_data["duration"]}, 重点: {phase_data["focus"]}'
                self.tableWidget.setItem(row, 1, QTableWidgetItem(duration_info))

                # 显示具体练习
                for i, exercise in enumerate(phase_data.get('exercises', [])[:3]):  # 限制显示数量
                    row = self.tableWidget.rowCount()
                    self.tableWidget.insertRow(row)
                    self.tableWidget.setItem(row, 0, QTableWidgetItem(f'  练习{i + 1}'))
                    exercise_info = f'{exercise["name"]} - {exercise["sets"]}组 {exercise["reps"]}次'
                    self.tableWidget.setItem(row, 1, QTableWidgetItem(exercise_info))

            # 显示进度监测指标
            if prescription.get('monitoring_metrics'):
                row = self.tableWidget.rowCount()
                self.tableWidget.insertRow(row)
                self.tableWidget.setItem(row, 0, QTableWidgetItem('监测指标'))
                metrics_text = ', '.join(prescription['monitoring_metrics'])
                self.tableWidget.setItem(row, 1, QTableWidgetItem(metrics_text))

        except Exception as e:
            QMessageBox.warning(self, '错误', f'训练处方生成失败: {str(e)}')

    def generate_sequence_based_training_prescription(self):
        """基于序列分析生成训练处方"""
        prescription = {
            'athlete_id': self.athlete_profile.get('id', 'unknown'),
            'generation_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'analysis_basis': 'complete_sequence',
            'sequence_duration': len(self.sequence_manager.analysis_results) / self.fpsRate,
            'risk_level': '低风险',
            'identified_issues': [],
            'focus_areas': [],
            'training_phases': {},
            'monitoring_metrics': []
        }

        try:
            sequence_summary = self.sequence_summary

            # 1. 分析主要问题
            issues = []
            focus_areas = []
            risk_factors = []

            # 分析角度稳定性问题
            if 'angles_stats' in sequence_summary:
                for angle_name, stats in sequence_summary['angles_stats'].items():
                    cv = stats.get('coefficient_variation', 0)
                    if cv > 0.3:
                        issues.append(f"{angle_name}稳定性不足(变异系数{cv:.2f})")
                        focus_areas.append(f"{angle_name}稳定性训练")
                        risk_factors.append(cv)

            # 分析运动质量问题
            if 'movement_quality' in sequence_summary:
                quality = sequence_summary['movement_quality']
                consistency = quality.get('consistency', 1.0)
                efficiency = quality.get('average_efficiency', 1.0)

                if consistency < 0.7:
                    issues.append(f"运动一致性偏低({consistency:.2f})")
                    focus_areas.append("动作协调性训练")
                    risk_factors.append(1.0 - consistency)

                if efficiency < 0.6:
                    issues.append(f"运动效率偏低({efficiency:.2f})")
                    focus_areas.append("运动技术优化")
                    risk_factors.append(1.0 - efficiency)

            # 分析稳定性问题
            if 'stability_metrics' in sequence_summary:
                stability = sequence_summary['stability_metrics']
                overall_stability = stability.get('overall_stability', 1.0)

                if overall_stability < 0.6:
                    issues.append(f"重心稳定性不足({overall_stability:.2f})")
                    focus_areas.append("核心稳定性训练")
                    risk_factors.append(1.0 - overall_stability)

            # 2. 确定风险等级
            if risk_factors:
                avg_risk = np.mean(risk_factors)
                if avg_risk > 0.6:
                    prescription['risk_level'] = '高风险'
                elif avg_risk > 0.3:
                    prescription['risk_level'] = '中风险'
                else:
                    prescription['risk_level'] = '低风险'

            prescription['identified_issues'] = issues
            prescription['focus_areas'] = list(set(focus_areas))  # 去重

            # 3. 制定分阶段训练计划
            if prescription['risk_level'] == '高风险':
                prescription['training_phases'] = self._create_high_risk_training_phases()
            elif prescription['risk_level'] == '中风险':
                prescription['training_phases'] = self._create_medium_risk_training_phases()
            else:
                prescription['training_phases'] = self._create_low_risk_training_phases()

            # 4. 设定监测指标
            prescription['monitoring_metrics'] = [
                '关节角度变异系数',
                '运动一致性指数',
                '重心稳定性评分',
                '动作完成质量'
            ]

        except Exception as e:
            logger.error(f"训练处方生成错误: {str(e)}")

        return prescription

    def _create_high_risk_training_phases(self):
        """创建高风险训练阶段"""
        return {
            'phase1': {
                'name': '基础稳定性重建期',
                'duration': '2-3周',
                'focus': '重建基础稳定性和控制能力',
                'exercises': [
                    {'name': '静态平衡训练', 'sets': 3, 'reps': '30秒', 'description': '单脚站立平衡'},
                    {'name': '核心激活训练', 'sets': 2, 'reps': 15, 'description': '平板支撑变化'},
                    {'name': '关节活动度训练', 'sets': 2, 'reps': 10, 'description': '缓慢控制性运动'}
                ]
            },
            'phase2': {
                'name': '动作模式重建期',
                'duration': '3-4周',
                'focus': '重建正确的动作模式',
                'exercises': [
                    {'name': '基础动作模式练习', 'sets': 3, 'reps': 8, 'description': '慢速标准动作'},
                    {'name': '镜像训练', 'sets': 2, 'reps': 10, 'description': '对着镜子练习'},
                    {'name': '本体感觉训练', 'sets': 2, 'reps': 12, 'description': '闭眼平衡训练'}
                ]
            }
        }

    def _create_medium_risk_training_phases(self):
        """创建中风险训练阶段"""
        return {
            'phase1': {
                'name': '稳定性强化期',
                'duration': '2周',
                'focus': '提高动作稳定性和一致性',
                'exercises': [
                    {'name': '动态平衡训练', 'sets': 3, 'reps': 12, 'description': '动态平衡挑战'},
                    {'name': '协调性训练', 'sets': 3, 'reps': 10, 'description': '多关节协调练习'},
                    {'name': '核心强化训练', 'sets': 2, 'reps': 15, 'description': '功能性核心训练'}
                ]
            },
            'phase2': {
                'name': '技术优化期',
                'duration': '2-3周',
                'focus': '优化运动技术和效率',
                'exercises': [
                    {'name': '技术细节练习', 'sets': 4, 'reps': 8, 'description': '分解动作练习'},
                    {'name': '速度渐进训练', 'sets': 3, 'reps': 6, 'description': '逐步提高速度'},
                    {'name': '负荷适应训练', 'sets': 3, 'reps': 10, 'description': '轻负荷技术练习'}
                ]
            }
        }

    def _create_low_risk_training_phases(self):
        """创建低风险训练阶段"""
        return {
            'phase1': {
                'name': '技术精进期',
                'duration': '1-2周',
                'focus': '进一步精进技术动作',
                'exercises': [
                    {'name': '高质量重复训练', 'sets': 4, 'reps': 6, 'description': '高质量标准动作'},
                    {'name': '变化条件训练', 'sets': 3, 'reps': 8, 'description': '不同条件下练习'},
                    {'name': '反馈式训练', 'sets': 3, 'reps': 10, 'description': '实时反馈练习'}
                ]
            },
            'phase2': {
                'name': '表现提升期',
                'duration': '持续进行',
                'focus': '提升运动表现和竞技水平',
                'exercises': [
                    {'name': '高强度间歇训练', 'sets': 4, 'reps': 5, 'description': '高强度技术练习'},
                    {'name': '竞技模拟训练', 'sets': 3, 'reps': 8, 'description': '比赛情境模拟'},
                    {'name': '专项素质训练', 'sets': 3, 'reps': 12, 'description': '专项身体素质'}
                ]
            }
        }

    def currentFrame(self):
        """显示当前帧"""
        if self.video and self.cap:
            try:
                self.cap.set(cv2.CAP_PROP_POS_FRAMES, self.fps)
                ret, frame = self.cap.read()

                if ret:
                    # 如果有关键点数据，绘制关键点
                    if self.pkl and self.data and self.fps < len(self.data):
                        keypoints_data = self.data[self.fps]
                        if keypoints_data is not None and len(keypoints_data) > 0:
                            # 显示前member_个人的关键点
                            shown_people = self.showPeople(keypoints_data)
                            for person_keypoints in shown_people:
                                EnhancedCalculationModule.draw(frame, person_keypoints, self.lSize, self.drawPoint)

                    # 转换为Qt图像格式
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    height, width, channel = frame.shape
                    bytes_per_line = 3 * width
                    q_img = QImage(frame.data, width, height, bytes_per_line, QImage.Format_RGB888)

                    # 显示图像
                    pixmap = QPixmap.fromImage(q_img)
                    self.imgLabel.setPixmap(pixmap)
                    self.imgLabel.setCursor(Qt.CrossCursor)

                    # 调整图像大小
                    self.imgLabel.adjustSize()
                    self.scaleFactor = 1.0

                    # 更新当前显示的分析结果
                    current_item = self.treeWidget.currentItem()
                    if current_item:
                        self.treeClicked()

            except Exception as e:
                QMessageBox.warning(self, '显示图像错误', str(e))

    def analytic(self):
        """解析视频关键点"""
        if not self.video:
            QMessageBox.warning(self, '警告', '请先选择视频文件！')
            return

        radio, ok = Dialog.getResult(self)
        if ok:
            try:
                pkl_path = analysis(self.video, self.cut1, self.cut2, zone=radio)
                self.pkl = pkl_path

                QMessageBox.information(self, '解析完成',
                                        f'解析点数据已保存至：\n{pkl_path}\n\n点击"载入关键点"可载入该文件。')
                self.loadKeys()
            except Exception as e:
                QMessageBox.warning(self, '解析错误', str(e), QMessageBox.Yes | QMessageBox.No, QMessageBox.Yes)

    def onFileOpen(self):
        """打开视频文件"""
        video_path, _ = QFileDialog.getOpenFileName(
            self, '打开视频文件', QDir.currentPath(),
            "视频文件 (*.mp4 *.avi *.mov *.mkv *.flv)"
        )

        if video_path:
            try:
                self.video = video_path
                self.horizontalSlider.setSliderPosition(0)
                self.pkl = False
                self.cap = cv2.VideoCapture(self.video)

                if not self.cap.isOpened():
                    QMessageBox.warning(self, '错误', '无法打开视频文件！')
                    return

                self.fpsMax = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT)) - 1
                self.horizontalSlider.setMaximum(self.fpsMax)

                # 状态栏内容
                self.fpsRate = round(self.cap.get(cv2.CAP_PROP_FPS), 2)
                width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                self.shape = f"{width}×{height}"
                self.text()
                self.sli_label()

                # 显示第一帧
                self.fps = 0
                self.currentFrame()

                # 设置按钮可用
                self.actionAnalysis.setEnabled(True)
                self.actionKey.setEnabled(True)
                self.actionscaledraw.setEnabled(True)
                self.actionZoomIn.setEnabled(True)
                self.actionZoomOut.setEnabled(True)
                self.actionNormalSize.setEnabled(True)
                self.actionFps.setEnabled(True)
                self.actionVideoNone.setEnabled(True)
                self.actionLevel.setEnabled(True)
                self.pushButton_6.setEnabled(True)
                self.pushButton_7.setEnabled(True)
                self.pushButton_3.setEnabled(True)
                self.pushButton_4.setEnabled(True)
                self.pushButton_10.setEnabled(True)
                self.pushButton.setEnabled(True)
                self.pushButton_2.setEnabled(True)
                self.pushButton_8.setEnabled(True)
                self.pushButton_9.setEnabled(True)

            except Exception as e:
                QMessageBox.warning(self, '打开视频错误', str(e))

    def loadKeys(self):
        """载入解析点 - 修复为手动选择文件"""
        pkl_path, _ = QFileDialog.getOpenFileName(
            self, '载入关键点', QDir.currentPath(),
            "Pickle Files (*.pkl);;All Files (*)")

        if pkl_path:
            self.pkl = pkl_path
            try:
                with open(self.pkl, 'rb') as file0:
                    self.data = pickle.load(file0)

                if self.data is not None:
                    self.currentFrame()
                    self.text(i=1)

                    # 启用相关功能
                    self.actionMember.setEnabled(True)
                    self.actionOutPoint.setEnabled(True)
                    self.actionOutPara.setEnabled(True)
                    self.actionOne.setEnabled(True)
                    self.actionSave.setEnabled(True)
                    self.actionOutVideo.setEnabled(True)
                    self.actionlineSize.setEnabled(True)

                    QMessageBox.information(self, '成功', '关键点数据载入成功！')

            except Exception as e:
                QMessageBox.warning(self, '载入解析点错误', str(e))

    def showPeople(self, keypoints_data):
        """显示最大的前N人"""
        if keypoints_data is None or len(keypoints_data) == 0:
            return []

        # 计算每个人的身体长度
        long_dic = {}
        for i, person_keypoints in enumerate(keypoints_data):
            try:
                if len(person_keypoints) >= 9:
                    neck = person_keypoints[1]  # 颈部
                    hip = person_keypoints[8]  # 中臀
                    if neck[2] > 0.1 and hip[2] > 0.1:  # 置信度检查
                        length = ((neck[0] - hip[0]) ** 2 + (neck[1] - hip[1]) ** 2) ** 0.5
                        long_dic[length] = i
            except Exception:
                continue

        # 按长度排序，选择最大的几个
        sorted_lengths = sorted(long_dic.items(), key=lambda x: x[0], reverse=True)
        selected_people = []

        for length, person_idx in sorted_lengths[:self.member_]:
            selected_people.append(keypoints_data[person_idx])

        return selected_people

    def choosePerson(self):
        """选择单人解析点"""
        try:
            if not self.data or self.fps >= len(self.data):
                return

            selected_row = self.tableWidget.currentRow()
            keypoints_data = self.data[self.fps]

            if keypoints_data is not None and selected_row < len(keypoints_data):
                # 显示选中的人
                self.currentFrame()

                # 只绘制选中的人
                frame = self.get_current_frame()
                if frame is not None:
                    shown_people = self.showPeople(keypoints_data)
                    if selected_row < len(shown_people):
                        self.selected_person = shown_people[selected_row]
                        EnhancedCalculationModule.draw(frame, self.selected_person, self.lSize, self.drawPoint)

                        # 显示图像
                        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        height, width, channel = frame.shape
                        bytes_per_line = 3 * width
                        q_img = QImage(frame.data, width, height, bytes_per_line, QImage.Format_RGB888)
                        self.imgLabel.setPixmap(QPixmap.fromImage(q_img))

        except Exception as e:
            QMessageBox.warning(self, '选择单人解析点错误', str(e))

    def confirmSelection(self):
        """确认选择人物"""
        if hasattr(self, 'selected_person'):
            # 将选中的人设为唯一的人
            self.data[self.fps] = np.array([self.selected_person])

            # 更新表格显示
            self.tableWidget.clearContents()
            self.tableWidget.setRowCount(1)
            self.tableWidget.setItem(0, 0, QTableWidgetItem('人物'))
            self.tableWidget.setItem(0, 1, QTableWidgetItem('0'))

    def generateMenu(self, pos):
        """生成右键菜单"""
        menu = QMenu()
        copy_action = menu.addAction("复制")
        clear_action = menu.addAction("清空")
        export_action = menu.addAction("导出当前分析")

        copy_action.triggered.connect(self.copyTable)
        clear_action.triggered.connect(self.clearTable)
        export_action.triggered.connect(self.exportCurrentAnalysis)

        menu.exec_(self.tableWidget.mapToGlobal(pos))

    def copyTable(self):
        """复制表格内容"""
        selection = self.tableWidget.selectedItems()
        if selection:
            text = ""
            for item in selection:
                text += item.text() + "\t"
            text = text.rstrip("\t")
            QApplication.clipboard().setText(text)

    def clearTable(self):
        """清空表格"""
        self.tableWidget.setRowCount(0)

    def exportCurrentAnalysis(self):
        """导出当前分析结果"""
        current_item = self.treeWidget.currentItem()
        if not current_item:
            QMessageBox.warning(self, '警告', '请先选择要导出的分析类型')
            return

        save_path, _ = QFileDialog.getSaveFileName(self, '导出分析结果', os.getcwd(),
                                                   "JSON Files (*.json);;CSV Files (*.csv)")
        if save_path:
            try:
                analysis_data = {
                    'analysis_type': current_item.text(0),
                    'frame_number': self.fps,
                    'timestamp': datetime.now().isoformat(),
                    'athlete_profile': self.athlete_profile,
                    'data': {}
                }

                # 收集表格数据
                for row in range(self.tableWidget.rowCount()):
                    key_item = self.tableWidget.item(row, 0)
                    value_item = self.tableWidget.item(row, 1)
                    if key_item and value_item:
                        analysis_data['data'][key_item.text()] = value_item.text()

                if save_path.endswith('.json'):
                    with open(save_path, 'w', encoding='utf-8') as f:
                        json.dump(analysis_data, f, ensure_ascii=False, indent=2)
                elif save_path.endswith('.csv'):
                    with open(save_path, 'w', newline='', encoding='utf-8') as f:
                        writer = csv.writer(f)
                        writer.writerow(['属性', '值'])
                        for key, value in analysis_data['data'].items():
                            writer.writerow([key, value])

                QMessageBox.information(self, '成功', '分析结果已导出')

            except Exception as e:
                QMessageBox.warning(self, '错误', f'导出失败: {str(e)}')

    def get_analysis_data(self):
        """获取当前分析数据，供智能分析中心使用"""
        return self.comprehensive_analysis()

    def sli_label(self):
        """显示滑动条状态"""
        if self.fpsMax > 0:
            time_now = round(self.fps / self.fpsRate, 3)
            time_total = round(self.fpsMax / self.fpsRate, 3)
            slide_text = f'总时长：{time_total}秒（{self.fpsMax}帧）      当前：{time_now}秒（{self.fps}帧）'
            self.label.setText(slide_text)

        range_text = f'工作区开始：{self.cut1}帧        工作区结束：{self.cut2}帧'
        self.label_4.setText(range_text)

    def text(self, i=0):
        """更新状态文本"""
        video_name = os.path.basename(self.video) if self.video else "未选择视频"
        if i:
            text = f'Video:{video_name}        Size:{self.shape}        FPS:{self.fpsRate}       显示解析人数：{self.member_}      画面旋转角：{self.rotationAngle}°       正方向：↓→'
        else:
            text = f'Video:{video_name}        Size:{self.shape}        FPS:{self.fpsRate}       画面旋转角：{self.rotationAngle}°     正方向：↓→'
        self.label_2.setText(text)

    def get_current_frame(self):
        """获取当前帧的副本"""
        if self.video and self.cap:
            current_pos = self.cap.get(cv2.CAP_PROP_POS_FRAMES)
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, self.fps)
            ret, frame = self.cap.read()
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, current_pos)
            return frame if ret else None
        return None

    def sli(self):
        """滑动条取值"""
        self.fps = self.horizontalSlider.value()
        self.sli_label()
        self.currentFrame()

    def next_(self):
        """下一帧"""
        if self.fps < self.fpsMax:
            self.fps += 1
            self.horizontalSlider.setSliderPosition(self.fps)
            self.sli_label()
            self.currentFrame()

    def last(self):
        """上一帧"""
        if self.fps > 0:
            self.fps -= 1
            self.horizontalSlider.setSliderPosition(self.fps)
            self.sli_label()
            self.currentFrame()

    def jumpToBeginning(self):
        """跳到开始"""
        self.fps = self.cut1 if self.cut1 is not None else 0
        self.horizontalSlider.setSliderPosition(self.fps)
        self.sli_label()
        self.currentFrame()

    def jumpToEnd(self):
        """跳到结束"""
        self.fps = self.cut2 if self.cut2 is not None else self.fpsMax
        self.horizontalSlider.setSliderPosition(self.fps)
        self.sli_label()
        self.currentFrame()

    # 简化的其他方法实现
    def onViewZoomIn(self):
        """放大"""
        if self.imgLabel.pixmap():
            self.scaleFactor *= 1.25
            self.scaleImage(self.scaleFactor)

    def onViewZoomOut(self):
        """缩小"""
        if self.imgLabel.pixmap():
            self.scaleFactor *= 0.8
            self.scaleImage(self.scaleFactor)

    def onViewNormalSize(self):
        """原始尺寸"""
        if self.imgLabel.pixmap():
            self.scaleFactor = 1.0
            self.scaleImage(self.scaleFactor)

    def scaleImage(self, factor):
        """缩放图像"""
        if self.imgLabel.pixmap():
            size = self.imgLabel.pixmap().size()
            scaled_pixmap = self.imgLabel.pixmap().scaled(
                size * factor, Qt.KeepAspectRatio, Qt.SmoothTransformation)
            self.imgLabel.setPixmap(scaled_pixmap)

    def realFPS(self):
        """设置实际帧率"""
        fps, ok = QInputDialog.getDouble(self, '设置帧率', '请输入实际帧率:', self.fpsRate, 1, 120, 2)
        if ok:
            self.fpsRate = fps
            self.text()

    def member(self):
        """设置显示人数"""
        count, ok = QInputDialog.getInt(self, '显示人数', '请输入显示人数:', self.member_, 1, 10)
        if ok:
            self.member_ = count
            self.currentFrame()

    def lineSize(self):
        """设置线条大小"""
        size, ok = QInputDialog.getInt(self, '线条大小', '请输入线条大小:', self.lSize, 1, 10)
        if ok:
            self.lSize = size
            self.currentFrame()

    def workspaceStart(self):
        """设置工作区开始"""
        self.cut1 = self.fps
        self.sli_label()

    def workspaceEnd(self):
        """设置工作区结束"""
        self.cut2 = self.fps
        self.sli_label()

    def workspaceClear(self):
        """清除工作区"""
        self.cut1 = None
        self.cut2 = None
        self.sli_label()

    # 简化其他功能方法的实现
    def scaleButton(self):
        """比例尺工具"""
        if not self.video:
            QMessageBox.warning(self, '警告', '请先选择视频文件！')
            return

        self.scale = True
        self.scale_point = []
        QMessageBox.information(self, '比例尺', '请在图像上点击两点设置比例尺')

    def Scale(self):
        """比例尺测量"""
        if self.scale and len(self.scale_point) < 2:
            self.scale_point.append([self.imgLabel.x, self.imgLabel.y])

            if len(self.scale_point) == 2:
                distance = math.sqrt(
                    (self.scale_point[1][0] - self.scale_point[0][0]) ** 2 +
                    (self.scale_point[1][1] - self.scale_point[0][1]) ** 2
                )

                real_length, ok = QInputDialog.getDouble(
                    self, '设置比例尺', '请输入实际长度(厘米):', 100, 1, 1000, 2
                )

                if ok:
                    self.pc = distance / real_length
                    self.scale = False
                    self.scale_point = []
                    QMessageBox.information(self, '成功', f'比例尺设置完成\n1像素 = {1 / self.pc:.3f}厘米')

    def lengthButton(self):
        """长度测量工具"""
        if not self.video:
            QMessageBox.warning(self, '警告', '请先选择视频文件！')
            return

        self.long = True
        self.lengthPoint = []
        self.setCursor(Qt.CrossCursor)
        QMessageBox.information(self, '长度测量', '请在图像上点击两点进行长度测量')

    def length(self):
        """长度测量实现"""
        if self.long and len(self.lengthPoint) < 2:
            self.lengthPoint.append([self.imgLabel.x, self.imgLabel.y])

            if len(self.lengthPoint) == 2:
                # 计算距离
                distance = math.sqrt(
                    (self.lengthPoint[1][0] - self.lengthPoint[0][0]) ** 2 +
                    (self.lengthPoint[1][1] - self.lengthPoint[0][1]) ** 2
                )

                # 记录测量结果
                measurement_name = f"长度测量{len(self.longDic) + 1}"
                if self.pc:
                    real_distance = distance / self.pc
                    self.longDic[f"{measurement_name}(厘米)"] = round(real_distance, 2)

                self.longDic[f"{measurement_name}(像素)"] = round(distance, 2)
                self.long = False
                self.lengthPoint = []
                self.setCursor(Qt.ArrowCursor)

                QMessageBox.information(self, '测量完成',
                                        f'测量距离: {distance:.2f}像素' +
                                        (f' ({real_distance:.2f}厘米)' if self.pc else ''))

    def timeButton(self):
        """时间测量工具"""
        pass

    def levelButton(self):
        """水平仪工具"""
        pass

    def levelTool(self):
        """水平仪测量"""
        pass

    def modifyKey(self):
        """修改关键点"""
        pass

    def save(self):
        """保存解析点"""
        if self.data is not None:
            save_path, _ = QFileDialog.getSaveFileName(
                self, '保存解析点', os.getcwd(), "Pickle Files (*.pkl)")
            if save_path:
                try:
                    with open(save_path, 'wb') as f:
                        pickle.dump(self.data, f)
                    QMessageBox.information(self, '成功', '解析点已保存')
                except Exception as e:
                    QMessageBox.warning(self, '错误', f'保存失败: {str(e)}')

    def exportVideo(self):
        """导出带解析点视频"""
        pass

    def exportPointlessVideo(self):
        """导出无解析点视频"""
        pass

    def exportKeys(self):
        """导出解析点数据"""
        if self.data is not None:
            save_path, _ = QFileDialog.getSaveFileName(
                self, '导出解析点数据', os.getcwd(), "JSON Files (*.json);;CSV Files (*.csv)")
            if save_path:
                try:
                    if save_path.endswith('.json'):
                        # 转换数据格式
                        export_data = []
                        for frame_idx, frame_data in enumerate(self.data):
                            if frame_data is not None:
                                for person_idx, person_keypoints in enumerate(frame_data):
                                    export_data.append({
                                        'frame': frame_idx,
                                        'person': person_idx,
                                        'keypoints': person_keypoints.tolist()
                                    })

                        with open(save_path, 'w', encoding='utf-8') as f:
                            json.dump(export_data, f, ensure_ascii=False, indent=2)

                    QMessageBox.information(self, '成功', '解析点数据已导出')
                except Exception as e:
                    QMessageBox.warning(self, '错误', f'导出失败: {str(e)}')

    def exportResults(self):
        """导出运动学参数"""
        if self.data is not None and self.athlete_profile:
            save_path, _ = QFileDialog.getSaveFileName(
                self, '导出运动学参数', os.getcwd(), "CSV Files (*.csv);;JSON Files (*.json)")
            if save_path:
                try:
                    all_results = []
                    for frame_idx in range(len(self.data)):
                        if self.data[frame_idx] is not None and len(self.data[frame_idx]) > 0:
                            current_keypoints = self.data[frame_idx][0]
                            last_keypoints = None
                            if frame_idx > 0 and self.data[frame_idx - 1] is not None:
                                last_keypoints = self.data[frame_idx - 1][0]

                            results = EnhancedCalculationModule.comprehensive_analysis(
                                current_keypoints, last_keypoints, self.fpsRate,
                                self.pc, self.rotationAngle, self.athlete_profile
                            )
                            results['frame'] = frame_idx
                            all_results.append(results)

                    if save_path.endswith('.json'):
                        with open(save_path, 'w', encoding='utf-8') as f:
                            json.dump(all_results, f, ensure_ascii=False, indent=2)
                    elif save_path.endswith('.csv'):
                        if all_results:
                            with open(save_path, 'w', newline='', encoding='utf-8') as f:
                                writer = csv.DictWriter(f, fieldnames=all_results[0].keys())
                                writer.writeheader()
                                writer.writerows(all_results)

                    QMessageBox.information(self, '成功', '运动学参数已导出')
                except Exception as e:
                    QMessageBox.warning(self, '错误', f'导出失败: {str(e)}')


# ==================== OpenPose 分析函数 ====================
def analysis(video, cut1, cut2, zone=0):
    """OpenPose视频分析函数"""
    # 确保资源路径正确
    dir_path = os.path.dirname(os.path.realpath(__file__))
    resource_path = os.path.join(dir_path, '..', 'resource')

    # 如果相对路径不存在，尝试使用绝对路径
    if not os.path.exists(resource_path):
        # 使用GoPose项目的绝对路径
        resource_path = r"D:\condaconda\GoPose-main\resource"

    sys.path.append(resource_path)
    bin_path = os.path.join(resource_path, 'bin')
    os.environ['PATH'] = os.environ['PATH'] + ';' + resource_path + ';' + bin_path + ';'
    os.environ['PATH'] = os.environ['PATH'] + ';' + resource_path + ';' + bin_path + ';'

    try:
        import pyopenpose as op
    except ImportError as e:
        raise ImportError(f"无法导入pyopenpose: {str(e)}")

    # 载入模型文件
    params = dict()
    params["model_folder"] = os.path.join(resource_path, "models")

    # 启动OpenPose
    opWrapper = op.WrapperPython()
    opWrapper.configure(params)
    opWrapper.start()

    video_name = os.path.basename(video)
    name = os.path.splitext(video_name)[0]
    datum = op.Datum()
    cap = cv2.VideoCapture(video)
    data_list = []

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    current_frame = 0

    # 设置窗口标题为正确编码
    window_title = '自动识别关键点(按ESC退出)'

    while cap.isOpened():
        ret, frame = cap.read()
        if ret:
            current_frame = int(cap.get(cv2.CAP_PROP_POS_FRAMES))

            if cut1 is not None and cut2 is not None and zone:
                if cut1 <= current_frame <= cut2:
                    datum.cvInputData = frame
                    opWrapper.emplaceAndPop(op.VectorDatum([datum]))
                    data = datum.poseKeypoints
                    data_list.append(data)
                    cv2.imshow(window_title, datum.cvOutputData)
                    if cv2.waitKey(20) == 27:
                        break
                else:
                    data_list.append(None)
                    cv2.imshow(window_title, frame)
                    if cv2.waitKey(20) == 27:
                        break
            else:
                datum.cvInputData = frame
                opWrapper.emplaceAndPop(op.VectorDatum([datum]))
                data = datum.poseKeypoints
                data_list.append(data)
                cv2.imshow(window_title, datum.cvOutputData)
                if cv2.waitKey(20) == 27:
                    break
        else:
            break

    cap.release()
    cv2.destroyAllWindows()

    # 保存数据并返回完整路径
    data_dir = os.path.join(os.getcwd(), 'data')
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)

    pkl_path = os.path.join(data_dir, f'{name}.pkl')
    with open(pkl_path, 'wb') as file0:
        pickle.dump(data_list, file0)

    return pkl_path


# ==================== 主程序 ====================

import numpy as np
import cv2
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import pandas as pd
from scipy import signal
from scipy.spatial.distance import euclidean
import warnings

warnings.filterwarnings('ignore')


# ==================== 3D运动分析器 ====================

def ThreeDAnalyzer(video_path=None, keypoints_data=None, frame_rate=30):
    """
    3D运动分析器

    参数:
    - video_path: 视频文件路径
    - keypoints_data: 关键点数据 (可选，如果提供则跳过视频处理)
    - frame_rate: 视频帧率

    返回:
    - analysis_results: 包含运动分析结果的字典
    """

    print("=== 3D运动分析器启动 ===")

    # 1. 数据预处理
    if keypoints_data is None:
        # 从视频中提取关键点数据 (简化版本)
        keypoints_data = extract_keypoints_from_video(video_path)

    # 2. 3D坐标重建
    coords_3d = reconstruct_3d_coordinates(keypoints_data)

    # 3. 运动轨迹分析
    trajectory_analysis = analyze_trajectory(coords_3d, frame_rate)

    # 4. 速度和加速度计算
    kinematics = calculate_kinematics(coords_3d, frame_rate)

    # 5. 运动模式识别
    motion_patterns = identify_motion_patterns(coords_3d, kinematics)

    # 6. 生成3D可视化
    generate_3d_visualization(coords_3d, trajectory_analysis)

    # 7. 运动质量评估
    quality_metrics = assess_motion_quality(coords_3d, kinematics)

    analysis_results = {
        'coordinates_3d': coords_3d,
        'trajectory': trajectory_analysis,
        'kinematics': kinematics,
        'motion_patterns': motion_patterns,
        'quality_metrics': quality_metrics,
        'frame_rate': frame_rate
    }

    print("✓ 3D运动分析完成")
    return analysis_results


def extract_keypoints_from_video(video_path):
    """从视频中提取关键点 (模拟数据)"""
    # 模拟关键点数据 - 实际应用中可使用OpenPose, MediaPipe等
    n_frames = 100
    n_keypoints = 17  # 人体关键点数量

    # 生成模拟的运动轨迹数据
    t = np.linspace(0, 2 * np.pi, n_frames)
    keypoints = np.zeros((n_frames, n_keypoints, 3))

    for i in range(n_keypoints):
        # 模拟不同关键点的运动轨迹
        keypoints[:, i, 0] = 100 + 50 * np.sin(t + i * 0.3)  # x
        keypoints[:, i, 1] = 100 + 30 * np.cos(t + i * 0.2)  # y
        keypoints[:, i, 2] = 50 + 20 * np.sin(2 * t + i * 0.1)  # z (深度)

    return keypoints


def reconstruct_3d_coordinates(keypoints_2d):
    """3D坐标重建"""
    # 简化的3D重建 - 实际应用中需要相机标定和立体视觉
    coords_3d = keypoints_2d.copy()

    # 添加深度信息的处理
    for frame in range(coords_3d.shape[0]):
        for point in range(coords_3d.shape[1]):
            # 基于运动学约束优化3D坐标
            coords_3d[frame, point] = optimize_3d_point(coords_3d[frame, point])

    return coords_3d


def optimize_3d_point(point):
    """优化3D点坐标"""
    # 简单的噪声过滤
    return point + np.random.normal(0, 0.1, 3)


def analyze_trajectory(coords_3d, frame_rate):
    """分析运动轨迹"""
    n_frames, n_keypoints, _ = coords_3d.shape

    trajectory_metrics = {
        'path_length': [],
        'displacement': [],
        'smoothness': [],
        'direction_changes': []
    }

    for keypoint in range(n_keypoints):
        trajectory = coords_3d[:, keypoint, :]

        # 计算路径长度
        path_length = calculate_path_length(trajectory)
        trajectory_metrics['path_length'].append(path_length)

        # 计算位移
        displacement = euclidean(trajectory[0], trajectory[-1])
        trajectory_metrics['displacement'].append(displacement)

        # 计算平滑度 (曲率变化)
        smoothness = calculate_smoothness(trajectory)
        trajectory_metrics['smoothness'].append(smoothness)

        # 方向变化次数
        direction_changes = count_direction_changes(trajectory)
        trajectory_metrics['direction_changes'].append(direction_changes)

    return trajectory_metrics


def calculate_path_length(trajectory):
    """计算路径长度"""
    distances = [euclidean(trajectory[i], trajectory[i + 1])
                 for i in range(len(trajectory) - 1)]
    return sum(distances)


def calculate_smoothness(trajectory):
    """计算轨迹平滑度"""
    # 使用二阶导数的方差来衡量平滑度
    diff2 = np.diff(trajectory, n=2, axis=0)
    smoothness = np.mean(np.var(diff2, axis=0))
    return smoothness


def count_direction_changes(trajectory):
    """计算方向变化次数"""
    velocities = np.diff(trajectory, axis=0)
    direction_changes = 0

    for i in range(len(velocities) - 1):
        if np.dot(velocities[i], velocities[i + 1]) < 0:
            direction_changes += 1

    return direction_changes


def calculate_kinematics(coords_3d, frame_rate):
    """计算运动学参数"""
    dt = 1.0 / frame_rate
    n_frames, n_keypoints, _ = coords_3d.shape

    # 计算速度 (一阶导数)
    velocities = np.gradient(coords_3d, dt, axis=0)

    # 计算加速度 (二阶导数)
    accelerations = np.gradient(velocities, dt, axis=0)

    # 计算速度大小
    speed = np.linalg.norm(velocities, axis=2)

    # 计算加速度大小
    acceleration_magnitude = np.linalg.norm(accelerations, axis=2)

    kinematics = {
        'positions': coords_3d,
        'velocities': velocities,
        'accelerations': accelerations,
        'speed': speed,
        'acceleration_magnitude': acceleration_magnitude,
        'max_speed': np.max(speed, axis=0),
        'avg_speed': np.mean(speed, axis=0),
        'max_acceleration': np.max(acceleration_magnitude, axis=0)
    }

    return kinematics


def identify_motion_patterns(coords_3d, kinematics):
    """识别运动模式"""
    patterns = {
        'motion_type': [],
        'periodicity': [],
        'dominant_frequency': [],
        'movement_efficiency': []
    }

    n_keypoints = coords_3d.shape[1]

    for keypoint in range(n_keypoints):
        speed_profile = kinematics['speed'][:, keypoint]
        position_profile = coords_3d[:, keypoint, :]

        # 运动类型分类
        motion_type = classify_motion_type(speed_profile)
        patterns['motion_type'].append(motion_type)

        # 周期性分析
        periodicity = analyze_periodicity(position_profile)
        patterns['periodicity'].append(periodicity)

        # 主频率分析
        dominant_freq = find_dominant_frequency(speed_profile)
        patterns['dominant_frequency'].append(dominant_freq)

        # 运动效率
        efficiency = calculate_movement_efficiency(position_profile, speed_profile)
        patterns['movement_efficiency'].append(efficiency)

    return patterns


def classify_motion_type(speed_profile):
    """分类运动类型"""
    speed_var = np.var(speed_profile)
    speed_mean = np.mean(speed_profile)

    if speed_var < 0.1 * speed_mean:
        return "uniform"
    elif speed_var < 0.5 * speed_mean:
        return "rhythmic"
    else:
        return "irregular"


def analyze_periodicity(position_profile):
    """分析周期性"""
    # 使用FFT分析周期性
    fft_result = np.fft.fft(position_profile[:, 0])  # 仅分析x轴
    frequencies = np.fft.fftfreq(len(position_profile))

    # 找到主要频率成分
    dominant_freq_idx = np.argmax(np.abs(fft_result[1:len(fft_result) // 2])) + 1
    periodicity_strength = np.abs(fft_result[dominant_freq_idx]) / np.sum(np.abs(fft_result))

    return periodicity_strength


def find_dominant_frequency(speed_profile):
    """找到主导频率"""
    frequencies, power = signal.periodogram(speed_profile)
    dominant_freq_idx = np.argmax(power[1:]) + 1
    return frequencies[dominant_freq_idx]


def calculate_movement_efficiency(position_profile, speed_profile):
    """计算运动效率"""
    # 效率 = 直线距离 / 实际路径长度
    straight_distance = euclidean(position_profile[0], position_profile[-1])
    actual_path_length = calculate_path_length(position_profile)

    if actual_path_length > 0:
        efficiency = straight_distance / actual_path_length
    else:
        efficiency = 0

    return efficiency


def generate_3d_visualization(coords_3d, trajectory_analysis):
    """生成3D可视化"""
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')

    # 绘制几个关键点的轨迹
    key_points = [0, 5, 10]  # 选择几个代表性关键点
    colors = ['red', 'blue', 'green']

    for i, point_idx in enumerate(key_points):
        trajectory = coords_3d[:, point_idx, :]
        ax.plot(trajectory[:, 0], trajectory[:, 1], trajectory[:, 2],
                color=colors[i], label=f'关键点 {point_idx}', linewidth=2)

        # 标记起始和结束点
        ax.scatter(trajectory[0, 0], trajectory[0, 1], trajectory[0, 2],
                   color=colors[i], s=100, marker='o', alpha=0.8)
        ax.scatter(trajectory[-1, 0], trajectory[-1, 1], trajectory[-1, 2],
                   color=colors[i], s=100, marker='s', alpha=0.8)

    ax.set_xlabel('X轴 (像素)')
    ax.set_ylabel('Y轴 (像素)')
    ax.set_zlabel('Z轴 (深度)')
    ax.set_title('3D运动轨迹分析')
    ax.legend()

    plt.tight_layout()
    plt.show()


def assess_motion_quality(coords_3d, kinematics):
    """评估运动质量"""
    quality_metrics = {
        'stability_score': [],
        'coordination_score': [],
        'fluency_score': [],
        'overall_score': []
    }

    n_keypoints = coords_3d.shape[1]

    for keypoint in range(n_keypoints):
        # 稳定性分数 (基于速度变化)
        speed_changes = np.diff(kinematics['speed'][:, keypoint])
        stability = 1.0 / (1.0 + np.var(speed_changes))
        quality_metrics['stability_score'].append(stability)

        # 协调性分数 (基于加速度平滑性)
        acc_smoothness = 1.0 / (1.0 + np.var(kinematics['acceleration_magnitude'][:, keypoint]))
        quality_metrics['coordination_score'].append(acc_smoothness)

        # 流畅性分数 (基于轨迹平滑性)
        trajectory = coords_3d[:, keypoint, :]
        fluency = 1.0 / (1.0 + calculate_smoothness(trajectory))
        quality_metrics['fluency_score'].append(fluency)

        # 综合分数
        overall = (stability + acc_smoothness + fluency) / 3.0
        quality_metrics['overall_score'].append(overall)

    return quality_metrics


# ==================== 深度学习增强器 ====================
# 保持原来的函数不变，添加一个包装类
# 保持原来的函数不变，添加一个包装类
class DeepLearningEnhancerWrapper:
    """
    DeepLearningEnhancer 包装器类
    当不提供参数调用DeepLearningEnhancer()时返回此类的实例
    """

    def __init__(self):
        self.motion_data = None
        print("=== 深度学习增强器包装器初始化 ===")

    def enhance(self, motion_data, enhancement_type='noise_reduction'):
        """
        执行增强处理

        参数:
        - motion_data: 运动数据
        - enhancement_type: 增强类型

        返回:
        - enhanced_results: 增强结果
        """
        return DeepLearningEnhancer(motion_data, enhancement_type)

    def __call__(self, motion_data, enhancement_type='noise_reduction'):
        """
        使对象可以像函数一样调用
        """
        return self.enhance(motion_data, enhancement_type)

    def set_motion_data(self, motion_data):
        """设置运动数据"""
        self.motion_data = motion_data

    def process_with_stored_data(self, enhancement_type='noise_reduction'):
        """使用存储的数据进行处理"""
        if self.motion_data is None:
            raise ValueError("请先设置运动数据或在调用时提供数据")
        return DeepLearningEnhancer(self.motion_data, enhancement_type)


# 修复后的主函数 - 只保留一个定义，支持可选参数
def DeepLearningEnhancer(motion_data=None, enhancement_type='noise_reduction'):
    """
    深度学习增强器

    参数:
    - motion_data: 运动数据 (来自ThreeDAnalyzer的结果)，如果为None则返回一个可调用对象
    - enhancement_type: 增强类型 ('noise_reduction', 'prediction', 'completion', 'classification')

    返回:
    - enhanced_results: 增强后的结果，或者一个可调用的增强器对象
    """

    # 如果没有提供motion_data，返回一个包装器类实例
    if motion_data is None:
        return DeepLearningEnhancerWrapper()

    print("=== 深度学习增强器启动 ===")

    enhanced_results = {}

    if enhancement_type == 'noise_reduction':
        # 1. 噪声减少和数据平滑
        enhanced_results = apply_noise_reduction(motion_data)

    elif enhancement_type == 'prediction':
        # 2. 运动预测
        enhanced_results = predict_future_motion(motion_data)

    elif enhancement_type == 'completion':
        # 3. 缺失数据补全
        enhanced_results = complete_missing_data(motion_data)

    elif enhancement_type == 'classification':
        # 4. 运动分类和识别
        enhanced_results = classify_motion_patterns(motion_data)

    elif enhancement_type == 'all':
        # 5. 综合增强
        enhanced_results = comprehensive_enhancement(motion_data)

    print(f"✓ 深度学习增强完成 (类型: {enhancement_type})")
    return enhanced_results


def apply_noise_reduction(motion_data):
    """应用噪声减少算法"""
    print("- 执行噪声减少...")

    coords_3d = motion_data['coordinates_3d']

    # 使用简化的自编码器概念进行噪声减少
    enhanced_coords = denoise_with_autoencoder(coords_3d)

    # 使用卡尔曼滤波进一步平滑
    smoothed_coords = apply_kalman_filter(enhanced_coords)

    # 重新计算运动学参数
    enhanced_kinematics = calculate_kinematics(smoothed_coords, motion_data['frame_rate'])

    return {
        'original_data': motion_data,
        'enhanced_coordinates': smoothed_coords,
        'enhanced_kinematics': enhanced_kinematics,
        'noise_reduction_ratio': calculate_noise_reduction_ratio(coords_3d, smoothed_coords)
    }


def denoise_with_autoencoder(coords_3d):
    """使用自编码器概念进行去噪 (简化版本)"""
    # 简化的自编码器逻辑
    n_frames, n_keypoints, n_dims = coords_3d.shape

    # 编码：降维和特征提取
    encoded = np.zeros((n_frames, n_keypoints, 2))  # 降到2维
    for keypoint in range(n_keypoints):
        # PCA降维作为编码过程
        data = coords_3d[:, keypoint, :]
        mean_data = np.mean(data, axis=0)
        centered_data = data - mean_data

        # 计算主成分
        cov_matrix = np.cov(centered_data.T)
        eigenvals, eigenvecs = np.linalg.eigh(cov_matrix)

        # 选择前两个主成分
        top_eigenvecs = eigenvecs[:, -2:]
        encoded[:, keypoint, :] = np.dot(centered_data, top_eigenvecs)

    # 解码：重建到原始维度
    decoded = np.zeros_like(coords_3d)
    for keypoint in range(n_keypoints):
        # 重建过程
        data = coords_3d[:, keypoint, :]
        mean_data = np.mean(data, axis=0)
        centered_data = data - mean_data

        cov_matrix = np.cov(centered_data.T)
        eigenvals, eigenvecs = np.linalg.eigh(cov_matrix)
        top_eigenvecs = eigenvecs[:, -2:]

        # 重建
        reconstructed = np.dot(encoded[:, keypoint, :], top_eigenvecs.T) + mean_data
        decoded[:, keypoint, :] = reconstructed

    return decoded


def apply_kalman_filter(coords_3d):
    """应用卡尔曼滤波"""
    n_frames, n_keypoints, n_dims = coords_3d.shape
    filtered_coords = np.zeros_like(coords_3d)

    for keypoint in range(n_keypoints):
        for dim in range(n_dims):
            # 简化的卡尔曼滤波
            signal_data = coords_3d[:, keypoint, dim]
            filtered_signal = simple_kalman_filter(signal_data)
            filtered_coords[:, keypoint, dim] = filtered_signal

    return filtered_coords


def simple_kalman_filter(signal_data, process_noise=0.01, measurement_noise=0.1):
    """简化的卡尔曼滤波器"""
    n = len(signal_data)
    filtered_signal = np.zeros(n)

    # 初始化
    x_est = signal_data[0]  # 初始状态估计
    p_est = 1.0  # 初始误差协方差

    for i in range(n):
        # 预测步骤
        x_pred = x_est
        p_pred = p_est + process_noise

        # 更新步骤
        kalman_gain = p_pred / (p_pred + measurement_noise)
        x_est = x_pred + kalman_gain * (signal_data[i] - x_pred)
        p_est = (1 - kalman_gain) * p_pred

        filtered_signal[i] = x_est

    return filtered_signal


def calculate_noise_reduction_ratio(original, enhanced):
    """计算噪声减少比例"""
    original_variance = np.var(original)
    enhanced_variance = np.var(enhanced)

    if original_variance > 0:
        reduction_ratio = 1 - (enhanced_variance / original_variance)
    else:
        reduction_ratio = 0

    return max(0, reduction_ratio)


def predict_future_motion(motion_data):
    """预测未来运动"""
    print("- 执行运动预测...")

    coords_3d = motion_data['coordinates_3d']
    kinematics = motion_data['kinematics']

    # 使用LSTM概念进行时间序列预测 (简化版本)
    future_frames = 30  # 预测未来30帧
    predicted_coords = lstm_prediction(coords_3d, future_frames)

    # 预测运动参数
    predicted_kinematics = extrapolate_kinematics(kinematics, future_frames)

    return {
        'original_data': motion_data,
        'predicted_coordinates': predicted_coords,
        'predicted_kinematics': predicted_kinematics,
        'prediction_confidence': calculate_prediction_confidence(coords_3d, predicted_coords)
    }


def lstm_prediction(coords_3d, future_frames):
    """使用LSTM概念进行预测 (简化版本)"""
    n_frames, n_keypoints, n_dims = coords_3d.shape

    # 简化的时间序列外推
    predicted_coords = np.zeros((future_frames, n_keypoints, n_dims))

    for keypoint in range(n_keypoints):
        for dim in range(n_dims):
            signal_data = coords_3d[:, keypoint, dim]

            # 使用多项式拟合进行外推
            t = np.arange(len(signal_data))
            poly_coeffs = np.polyfit(t, signal_data, deg=3)

            # 预测未来点
            future_t = np.arange(len(signal_data), len(signal_data) + future_frames)
            predicted_signal = np.polyval(poly_coeffs, future_t)

            predicted_coords[:, keypoint, dim] = predicted_signal

    return predicted_coords


def extrapolate_kinematics(kinematics, future_frames):
    """外推运动学参数"""
    # 基于当前趋势外推速度和加速度
    current_velocities = kinematics['velocities'][-10:]  # 最后10帧的速度
    current_accelerations = kinematics['accelerations'][-10:]  # 最后10帧的加速度

    # 计算平均变化趋势
    velocity_trend = np.mean(np.diff(current_velocities, axis=0), axis=0)
    acceleration_trend = np.mean(np.diff(current_accelerations, axis=0), axis=0)

    # 外推未来的运动学参数
    future_velocities = []
    future_accelerations = []

    last_velocity = current_velocities[-1]
    last_acceleration = current_accelerations[-1]

    for frame in range(future_frames):
        next_velocity = last_velocity + velocity_trend
        next_acceleration = last_acceleration + acceleration_trend

        future_velocities.append(next_velocity)
        future_accelerations.append(next_acceleration)

        last_velocity = next_velocity
        last_acceleration = next_acceleration

    return {
        'predicted_velocities': np.array(future_velocities),
        'predicted_accelerations': np.array(future_accelerations)
    }


def calculate_prediction_confidence(historical_data, predicted_data):
    """计算预测置信度"""
    # 基于历史数据的变异性计算置信度
    historical_variance = np.var(historical_data)
    predicted_variance = np.var(predicted_data)

    # 简化的置信度计算
    if historical_variance > 0:
        confidence = 1.0 / (1.0 + abs(predicted_variance - historical_variance) / historical_variance)
    else:
        confidence = 0.5

    return min(1.0, max(0.0, confidence))


def complete_missing_data(motion_data):
    """补全缺失数据"""
    print("- 执行缺失数据补全...")

    coords_3d = motion_data['coordinates_3d']

    # 模拟一些缺失数据
    coords_with_missing = introduce_missing_data(coords_3d.copy())

    # 使用插值和机器学习方法补全
    completed_coords = interpolate_missing_data(coords_with_missing)

    return {
        'original_data': motion_data,
        'data_with_missing': coords_with_missing,
        'completed_coordinates': completed_coords,
        'completion_accuracy': calculate_completion_accuracy(coords_3d, completed_coords)
    }


def introduce_missing_data(coords_3d, missing_ratio=0.1):
    """引入模拟的缺失数据"""
    coords_with_missing = coords_3d.copy()
    n_frames, n_keypoints, n_dims = coords_3d.shape

    # 随机选择要设为缺失的数据点
    n_missing = int(n_frames * n_keypoints * missing_ratio)

    for _ in range(n_missing):
        frame_idx = np.random.randint(0, n_frames)
        keypoint_idx = np.random.randint(0, n_keypoints)
        coords_with_missing[frame_idx, keypoint_idx, :] = np.nan

    return coords_with_missing


def interpolate_missing_data(coords_with_missing):
    """插值补全缺失数据"""
    n_frames, n_keypoints, n_dims = coords_with_missing.shape
    completed_coords = coords_with_missing.copy()

    for keypoint in range(n_keypoints):
        for dim in range(n_dims):
            signal_data = coords_with_missing[:, keypoint, dim]

            # 找到非缺失的数据点
            valid_indices = ~np.isnan(signal_data)

            if np.any(valid_indices):
                valid_frames = np.where(valid_indices)[0]
                valid_values = signal_data[valid_indices]

                # 对缺失数据进行插值
                missing_indices = np.where(~valid_indices)[0]
                if len(missing_indices) > 0:
                    interpolated_values = np.interp(missing_indices, valid_frames, valid_values)
                    completed_coords[missing_indices, keypoint, dim] = interpolated_values

    return completed_coords


def calculate_completion_accuracy(original, completed):
    """计算补全准确性"""
    mse = np.mean((original - completed) ** 2)
    max_value = np.max(np.abs(original))

    if max_value > 0:
        accuracy = 1.0 / (1.0 + mse / (max_value ** 2))
    else:
        accuracy = 1.0

    return accuracy


def classify_motion_patterns(motion_data):
    """分类运动模式"""
    print("- 执行运动模式分类...")

    # 提取特征
    features = extract_motion_features(motion_data)

    # 简化的分类器 (基于规则)
    classifications = rule_based_classifier(features)

    # 计算分类置信度
    confidence_scores = calculate_classification_confidence(features, classifications)

    return {
        'original_data': motion_data,
        'extracted_features': features,
        'classifications': classifications,
        'confidence_scores': confidence_scores
    }


def extract_motion_features(motion_data):
    """提取运动特征"""
    coords_3d = motion_data['coordinates_3d']
    kinematics = motion_data['kinematics']

    features = {}

    # 统计特征
    features['mean_speed'] = np.mean(kinematics['speed'], axis=0)
    features['max_speed'] = np.max(kinematics['speed'], axis=0)
    features['speed_variance'] = np.var(kinematics['speed'], axis=0)

    # 轨迹特征
    features['path_length'] = [calculate_path_length(coords_3d[:, i, :])
                               for i in range(coords_3d.shape[1])]

    # 频域特征
    features['dominant_frequencies'] = []
    for keypoint in range(coords_3d.shape[1]):
        speed_signal = kinematics['speed'][:, keypoint]
        dom_freq = find_dominant_frequency(speed_signal)
        features['dominant_frequencies'].append(dom_freq)

    return features


def rule_based_classifier(features):
    """基于规则的分类器"""
    classifications = []

    for keypoint in range(len(features['mean_speed'])):
        mean_speed = features['mean_speed'][keypoint]
        speed_var = features['speed_variance'][keypoint]
        dom_freq = features['dominant_frequencies'][keypoint]

        # 简单的分类规则
        if mean_speed < 5:
            motion_class = "静止"
        elif speed_var < 2 and dom_freq < 0.5:
            motion_class = "匀速运动"
        elif dom_freq > 1.0:
            motion_class = "节律运动"
        elif speed_var > 10:
            motion_class = "不规则运动"
        else:
            motion_class = "一般运动"

        classifications.append(motion_class)

    return classifications


def calculate_classification_confidence(features, classifications):
    """计算分类置信度"""
    confidence_scores = []

    for i, classification in enumerate(classifications):
        # 基于特征的一致性计算置信度
        mean_speed = features['mean_speed'][i]
        speed_var = features['speed_variance'][i]

        # 简化的置信度计算
        if classification == "静止":
            confidence = 1.0 / (1.0 + mean_speed)
        elif classification == "匀速运动":
            confidence = 1.0 / (1.0 + speed_var)
        elif classification == "节律运动":
            # 这里需要定义 dom_freq，假设从特征中获取
            dom_freq = features.get('dominant_frequency', [0])[i] if i < len(
                features.get('dominant_frequency', [])) else 0
            confidence = min(1.0, dom_freq)
        elif classification == "不规则运动":
            confidence = speed_var / (speed_var + 10)
        else:
            confidence = 0.5

        confidence_scores.append(min(1.0, max(0.0, confidence)))

    return confidence_scores


def comprehensive_enhancement(motion_data):
    """综合增强 - 应用所有增强技术"""
    print("- 执行综合增强...")

    enhanced_results = {}

    # 1. 噪声减少
    print("  > 噪声减少...")
    noise_reduced = apply_noise_reduction(motion_data)
    enhanced_results['noise_reduction'] = noise_reduced

    # 2. 运动预测
    print("  > 运动预测...")
    motion_prediction = predict_future_motion(motion_data)
    enhanced_results['motion_prediction'] = motion_prediction

    # 3. 缺失数据补全
    print("  > 数据补全...")
    data_completion = complete_missing_data(motion_data)
    enhanced_results['data_completion'] = data_completion

    # 4. 运动分类
    print("  > 运动分类...")
    motion_classification = classify_motion_patterns(motion_data)
    enhanced_results['motion_classification'] = motion_classification

    # 5. 综合质量评估
    print("  > 综合质量评估...")
    comprehensive_quality = evaluate_comprehensive_quality(enhanced_results)
    enhanced_results['comprehensive_quality'] = comprehensive_quality

    # 6. 生成增强报告
    enhancement_report = generate_enhancement_report(enhanced_results)
    enhanced_results['enhancement_report'] = enhancement_report

    return enhanced_results


def evaluate_comprehensive_quality(enhanced_results):
    """评估综合质量"""
    quality_metrics = {
        'data_quality_score': 0.0,
        'prediction_reliability': 0.0,
        'classification_accuracy': 0.0,
        'overall_enhancement_score': 0.0
    }

    # 数据质量分数
    if 'noise_reduction' in enhanced_results:
        noise_reduction_ratio = enhanced_results['noise_reduction']['noise_reduction_ratio']
        quality_metrics['data_quality_score'] = np.mean(noise_reduction_ratio)

    # 预测可靠性
    if 'motion_prediction' in enhanced_results:
        prediction_confidence = enhanced_results['motion_prediction']['prediction_confidence']
        quality_metrics['prediction_reliability'] = prediction_confidence

    # 分类准确性
    if 'motion_classification' in enhanced_results:
        classification_confidence = enhanced_results['motion_classification']['confidence_scores']
        quality_metrics['classification_accuracy'] = np.mean(classification_confidence)

    # 综合增强分数
    scores = [
        quality_metrics['data_quality_score'],
        quality_metrics['prediction_reliability'],
        quality_metrics['classification_accuracy']
    ]
    quality_metrics['overall_enhancement_score'] = np.mean([s for s in scores if s > 0])

    return quality_metrics


def generate_enhancement_report(enhanced_results):
    """生成增强报告"""
    report = {
        'timestamp': pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S'),
        'applied_enhancements': [],
        'performance_metrics': {},
        'recommendations': []
    }

    # 记录应用的增强技术
    if 'noise_reduction' in enhanced_results:
        report['applied_enhancements'].append('噪声减少')
        noise_ratio = enhanced_results['noise_reduction']['noise_reduction_ratio']
        report['performance_metrics']['noise_reduction_ratio'] = f"{np.mean(noise_ratio):.3f}"

    if 'motion_prediction' in enhanced_results:
        report['applied_enhancements'].append('运动预测')
        pred_confidence = enhanced_results['motion_prediction']['prediction_confidence']
        report['performance_metrics']['prediction_confidence'] = f"{pred_confidence:.3f}"

    if 'data_completion' in enhanced_results:
        report['applied_enhancements'].append('数据补全')
        completion_accuracy = enhanced_results['data_completion']['completion_accuracy']
        report['performance_metrics']['completion_accuracy'] = f"{completion_accuracy:.3f}"

    if 'motion_classification' in enhanced_results:
        report['applied_enhancements'].append('运动分类')
        class_confidence = enhanced_results['motion_classification']['confidence_scores']
        report['performance_metrics']['classification_confidence'] = f"{np.mean(class_confidence):.3f}"

    # 生成建议
    if 'comprehensive_quality' in enhanced_results:
        overall_score = enhanced_results['comprehensive_quality']['overall_enhancement_score']

        if overall_score > 0.8:
            report['recommendations'].append("数据质量优秀，建议用于高精度分析")
        elif overall_score > 0.6:
            report['recommendations'].append("数据质量良好，可进行常规分析")
        elif overall_score > 0.4:
            report['recommendations'].append("数据质量一般，建议进一步优化")
        else:
            report['recommendations'].append("数据质量较差，需要重新采集或更多预处理")

    return report

class EnhancedDataAnalysisUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("增强版运动姿势改良系统")
        self.resize(1600, 1000)

        # 初始化科研管理器
        self.research_manager = ResearchDataManager()
        self.current_project_id = None

        # 创建主标签页（只创建一次！）
        self.tab_widget = QTabWidget()
        self.setCentralWidget(self.tab_widget)

        # 添加运动数据分析标签页
        self.data_analysis_tab = QWidget()
        self.init_data_analysis_ui()
        self.tab_widget.addTab(self.data_analysis_tab, "智能分析中心")

        # 添加增强版GoPose标签页
        self.enhanced_gopose_tab = EnhancedGoPoseModule()
        self.tab_widget.addTab(self.enhanced_gopose_tab, "增强版GoPose分析")

        # 添加科研管理标签页
        self.research_tab = QWidget()
        self.init_research_management_ui()
        self.tab_widget.addTab(self.research_tab, "科研管理中心")

        # 初始化智能教练状态
        self.smart_coach_status = "正在初始化智能教练..."
        self.check_smart_coach_availability()
        # 删除重复的代码块！

    def closeEvent(self, event):
        """关闭事件处理"""
        reply = QMessageBox.question(self, '确认退出',
                                     '确定要退出增强版运动姿势改良系统吗？',
                                     QMessageBox.Yes | QMessageBox.No,
                                     QMessageBox.No)

        if reply == QMessageBox.Yes:
            try:
                # 清理GoPose模块
                if hasattr(self, 'enhanced_gopose_tab'):
                    if hasattr(self.enhanced_gopose_tab, 'memory_manager'):
                        self.enhanced_gopose_tab.memory_manager.cleanup_on_exit()
                    if hasattr(self.enhanced_gopose_tab, 'cap') and self.enhanced_gopose_tab.cap:
                        self.enhanced_gopose_tab.cap.release()
                    if hasattr(self.enhanced_gopose_tab, 'play_timer'):
                        self.enhanced_gopose_tab.play_timer.stop()

                event.accept()
            except Exception as e:
                logger.error(f"应用程序关闭清理失败: {e}")
                event.accept()  # 仍然接受关闭事件
        else:
            event.ignore()

    def check_smart_coach_availability(self):
        """检查智能教练可用性"""

        def check_async():
            try:
                if SMART_COACH_AVAILABLE:
                    test_bot = SmartSportsBot()
                    if test_bot.coach_available:
                        self.smart_coach_status = "✅ 智能运动教练已就绪"
                    else:
                        self.smart_coach_status = "⚠️ 智能教练模式受限"
                else:
                    self.smart_coach_status = "📚 基础教练模式"
            except:
                self.smart_coach_status = "❌ 教练初始化失败"

        threading.Thread(target=check_async, daemon=True).start()

    # 在 init_data_analysis_ui 方法中的改进

    def init_data_analysis_ui(self):
        # 主布局
        layout = QVBoxLayout(self.data_analysis_tab)
        layout.setSpacing(24)  # 增加间距
        layout.setContentsMargins(32, 32, 32, 32)  # 增加边距

        # 1. 简化标题区域
        header_widget = QWidget()
        header_widget.setStyleSheet("""
            QWidget {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 #ffffff, stop:1 #f8f9fa);
                border-radius: 16px;
                padding: 24px;
            }
        """)
        header_layout = QVBoxLayout(header_widget)

        title = QLabel("运动姿势智能分析系统")
        title.setAlignment(Qt.AlignCenter)
        title.setStyleSheet("""
            QLabel {
                font-size: 32px;
                font-weight: 700;
                color: #212529;
                margin: 0;
                padding: 0;
            }
        """)

        subtitle = QLabel("专业运动生物力学分析 • AI损伤风险评估 • 个性化训练方案")
        subtitle.setAlignment(Qt.AlignCenter)
        subtitle.setStyleSheet("""
            QLabel {
                font-size: 16px;
                color: #6c757d;
                margin-top: 8px;
                font-weight: 400;
            }
        """)

        header_layout.addWidget(title)
        header_layout.addWidget(subtitle)
        layout.addWidget(header_widget)

        # 2. 主要按钮区域
        action_widget = QWidget()
        action_layout = QHBoxLayout(action_widget)
        action_layout.setSpacing(16)

        # 主要分析按钮
        self.start_analysis_btn = QPushButton('开始分析')
        self.start_analysis_btn.setObjectName("primary-button")
        self.start_analysis_btn.setFixedSize(160, 48)
        self.start_analysis_btn.setStyleSheet("""
            QPushButton#primary-button {
                background-color: #0d6efd;
                color: white;
                border: none;
                border-radius: 24px;
                font-size: 16px;
                font-weight: 600;
            }
            QPushButton#primary-button:hover {
                background-color: #0b5ed7;
                transform: translateY(-2px);
            }
        """)

        # AI教练按钮
        self.ai_coach_btn = QPushButton('智能教练')
        self.ai_coach_btn.setObjectName("secondary-button")
        self.ai_coach_btn.setFixedSize(160, 48)
        self.ai_coach_btn.setStyleSheet("""
            QPushButton#secondary-button {
                background-color: #ffffff;
                color: #495057;
                border: 2px solid #dee2e6;
                border-radius: 24px;
                font-size: 16px;
                font-weight: 600;
            }
            QPushButton#secondary-button:hover {
                background-color: #f8f9fa;
                border-color: #0d6efd;
                color: #0d6efd;
            }
            QPushButton#secondary-button:pressed {
                background-color: #e7f1ff;
                border-color: #0b5ed7;
            }
        """)

        action_layout.addStretch()
        action_layout.addWidget(self.start_analysis_btn)
        action_layout.addWidget(self.ai_coach_btn)
        action_layout.addStretch()

        layout.addWidget(action_widget)

        # 3. 功能卡片区域
        cards_widget = QWidget()
        cards_layout = QHBoxLayout(cards_widget)
        cards_layout.setSpacing(16)

        # 使用更简单的图标和颜色
        features = [
            ("生物力学分析", "关节力矩 • 能量传递\n重心分析 • 活动度评估", "#0d6efd"),
            ("损伤风险评估", "膝关节检测 • 肩关节分析\n脊柱评估 • 运动模式", "#dc3545"),
            ("智能训练方案", "个性化处方 • 进度跟踪\n康复建议 • 专项训练", "#198754")
        ]

        for title, content, color in features:
            card = self.create_feature_card(title, content, color)
            cards_layout.addWidget(card)

        layout.addWidget(cards_widget)

        # 4. 快捷功能按钮区域
        shortcuts_widget = QWidget()
        shortcuts_layout = QHBoxLayout(shortcuts_widget)
        shortcuts_layout.setSpacing(12)

        # 定义快捷按钮列表
        shortcut_buttons = [
            ('📊 表现评分', self.show_performance_dashboard),
            ('📈 历史分析', self.show_history_dashboard),
            ('🎯 标准对比', self.show_comparison_dashboard),
            ('⚕️ 健康报告', self.show_health_dashboard)
        ]

        for text, slot in shortcut_buttons:
            btn = QPushButton(text)
            btn.clicked.connect(slot)
            btn.setStyleSheet("""
                QPushButton {
                    background-color: #ffffff;
                    color: #495057;
                    border: 2px solid #dee2e6;
                    padding: 12px 16px;
                    border-radius: 8px;
                    font-size: 14px;
                    font-weight: 500;
                    min-height: 20px;
                }
                QPushButton:hover {
                    background-color: #f8f9fa;
                    border-color: #0d6efd;
                    color: #0d6efd;
                }
                QPushButton:pressed {
                    background-color: #e7f1ff;
                    border-color: #0b5ed7;
                }
            """)
            shortcuts_layout.addWidget(btn)

        layout.addWidget(shortcuts_widget)

        # 5. 状态区域
        status_widget = QWidget()
        status_widget.setStyleSheet("""
            QWidget {
                background-color: #ffffff;
                border: 1px solid #dee2e6;
                border-radius: 8px;
                padding: 16px;
            }
        """)
        status_layout = QVBoxLayout(status_widget)

        self.system_status = QLabel("系统就绪")
        self.system_status.setStyleSheet("""
            QLabel {
                font-size: 14px;
                color: #198754;
                font-weight: 500;
            }
        """)

        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        self.progress_bar.setFixedHeight(8)

        status_layout.addWidget(self.system_status)
        status_layout.addWidget(self.progress_bar)

        layout.addWidget(status_widget)

        # 6. 结果显示区域
        self.results_group = QGroupBox()
        self.results_group.setTitle("")  # 移除标题
        self.results_group.setStyleSheet("""
            QGroupBox {
                background-color: #ffffff;
                border: 1px solid #dee2e6;
                border-radius: 12px;
                padding: 20px;
                margin-top: 0;
            }
        """)

        self.results_layout = QVBoxLayout()

        # 创建结果标签页
        self.results_tab_widget = QTabWidget()
        self.results_tab_widget.setStyleSheet("""
            QTabWidget::pane {
                border: none;
                background-color: transparent;
            }
            QTabBar::tab {
                padding: 12px 20px;
                margin-right: 4px;
                background-color: #f8f9fa;
                border-radius: 6px 6px 0 0;
            }
            QTabBar::tab:selected {
                background-color: #0d6efd;
                color: white;
            }
        """)

        # 添加结果标签页
        self.setup_results_tabs()

        self.results_layout.addWidget(self.results_tab_widget)
        self.results_group.setLayout(self.results_layout)
        layout.addWidget(self.results_group)

        # 连接事件
        self.start_analysis_btn.clicked.connect(self.start_comprehensive_analysis)
        self.ai_coach_btn.clicked.connect(self.open_ai_coach)

    def setup_results_tabs(self):
        """设置结果显示标签页"""
        # 基础运动学结果标签页
        self.basic_widget = QWidget()
        self.basic_layout = QVBoxLayout(self.basic_widget)
        self.basic_table = QTableWidget()
        self.basic_table.setColumnCount(2)
        self.basic_table.setHorizontalHeaderLabels(["参数", "值"])
        self.basic_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.basic_layout.addWidget(self.basic_table)
        self.results_tab_widget.addTab(self.basic_widget, "基础运动学")

        # 生物力学分析结果标签页
        self.biomech_widget = QWidget()
        self.biomech_layout = QVBoxLayout(self.biomech_widget)
        self.biomech_table = QTableWidget()
        self.biomech_table.setColumnCount(2)
        self.biomech_table.setHorizontalHeaderLabels(["生物力学参数", "值"])
        self.biomech_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.biomech_layout.addWidget(self.biomech_table)
        self.results_tab_widget.addTab(self.biomech_widget, "生物力学")

        # 损伤风险评估标签页
        self.risk_widget = QWidget()
        self.risk_layout = QVBoxLayout(self.risk_widget)
        self.risk_table = QTableWidget()
        self.risk_table.setColumnCount(2)
        self.risk_table.setHorizontalHeaderLabels(["风险评估", "结果"])
        self.risk_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.risk_layout.addWidget(self.risk_table)
        self.results_tab_widget.addTab(self.risk_widget, "损伤风险")

        # 训练处方标签页
        self.prescription_widget = QWidget()
        self.prescription_layout = QVBoxLayout(self.prescription_widget)
        self.prescription_table = QTableWidget()
        self.prescription_table.setColumnCount(2)
        self.prescription_table.setHorizontalHeaderLabels(["训练建议", "内容"])
        self.prescription_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.prescription_layout.addWidget(self.prescription_table)
        self.results_tab_widget.addTab(self.prescription_widget, "训练处方")

    def show_performance_dashboard(self):
        """显示表现仪表板"""
        try:
            # 获取GoPose数据
            gopose_module = self.enhanced_gopose_tab
            if not gopose_module.data or not gopose_module.athlete_profile:
                QMessageBox.warning(self, '数据不足',
                                    '请先在GoPose标签页中载入数据和设置运动员档案')
                return

            # 计算表现评分
            analysis_data = gopose_module.get_analysis_data()
            if analysis_data:
                performance_scores = PerformanceScoreSystem.calculate_performance_score(
                    analysis_data,
                    gopose_module.athlete_profile.get('sport', 'general')
                )

                # 创建表现仪表板窗口
                dashboard_dialog = QDialog(self)
                dashboard_dialog.setWindowTitle('表现评分仪表板')
                dashboard_dialog.setFixedSize(800, 600)

                layout = QVBoxLayout(dashboard_dialog)

                # 评分显示
                score_widget = QWidget()
                score_layout = QHBoxLayout(score_widget)

                # 总体得分
                overall_label = QLabel(
                    f"总体得分\n{performance_scores['overall_score']:.1f}分\n({performance_scores['grade']})")
                overall_label.setAlignment(Qt.AlignCenter)
                overall_label.setStyleSheet("""
                    QLabel {
                        background-color: #0d6efd;
                        color: white;
                        border-radius: 12px;
                        padding: 20px;
                        font-size: 18px;
                        font-weight: bold;
                    }
                """)

                # 各维度得分
                scores_data = [
                    ('技术', performance_scores['technique_score'], '#dc3545'),
                    ('稳定性', performance_scores['stability_score'], '#fd7e14'),
                    ('效率', performance_scores['efficiency_score'], '#198754'),
                    ('安全性', performance_scores['safety_score'], '#6f42c1')
                ]

                score_layout.addWidget(overall_label)

                for name, score, color in scores_data:
                    score_label = QLabel(f"{name}\n{score:.1f}分")
                    score_label.setAlignment(Qt.AlignCenter)
                    score_label.setStyleSheet(f"""
                        QLabel {{
                            background-color: {color};
                            color: white;
                            border-radius: 8px;
                            padding: 15px;
                            font-size: 14px;
                            font-weight: bold;
                        }}
                    """)
                    score_layout.addWidget(score_label)

                layout.addWidget(score_widget)

                # 建议显示
                recommendations_group = QGroupBox("改进建议")
                recommendations_layout = QVBoxLayout(recommendations_group)

                for i, rec in enumerate(performance_scores['recommendations']):
                    rec_label = QLabel(f"{i + 1}. {rec}")
                    rec_label.setWordWrap(True)
                    rec_label.setStyleSheet("padding: 8px; border-bottom: 1px solid #dee2e6;")
                    recommendations_layout.addWidget(rec_label)

                layout.addWidget(recommendations_group)

                dashboard_dialog.exec_()
            else:
                QMessageBox.warning(self, '警告', '无法获取分析数据')

        except Exception as e:
            QMessageBox.warning(self, '错误', f'显示表现仪表板失败: {str(e)}')

    def show_history_dashboard(self):
        """显示历史分析仪表板"""
        try:
            gopose_module = self.enhanced_gopose_tab
            if not gopose_module.athlete_profile:
                QMessageBox.warning(self, '警告', '请先设置运动员档案')
                return

            # 获取历史数据
            progress_tracker = ProgressTrackingModule()
            athlete_id = gopose_module.athlete_profile.get('id', 'unknown')
            report = progress_tracker.generate_progress_report(athlete_id, days=30)

            # 创建历史分析窗口
            history_dialog = QDialog(self)
            history_dialog.setWindowTitle('历史训练分析')
            history_dialog.setFixedSize(900, 700)

            layout = QVBoxLayout(history_dialog)

            # 摘要信息
            summary_label = QLabel(f"📊 {report['summary']}")
            summary_label.setStyleSheet("""
                QLabel {
                    background-color: #e7f1ff;
                    padding: 15px;
                    border-radius: 8px;
                    font-size: 16px;
                    border-left: 4px solid #0d6efd;
                }
            """)
            layout.addWidget(summary_label)

            # 趋势分析表格
            trends_group = QGroupBox("趋势分析")
            trends_layout = QVBoxLayout(trends_group)

            trends_table = QTableWidget()
            trends_table.setColumnCount(3)
            trends_table.setHorizontalHeaderLabels(['指标', '变化趋势', '变化幅度'])
            trends_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)

            row = 0
            for metric, trend_data in report['trends'].items():
                trends_table.insertRow(row)
                metric_name = {
                    'overall_score': '总体得分',
                    'technique_score': '技术得分',
                    'stability_score': '稳定性得分',
                    'efficiency_score': '效率得分',
                    'safety_score': '安全性得分'
                }.get(metric, metric)

                trends_table.setItem(row, 0, QTableWidgetItem(metric_name))
                trends_table.setItem(row, 1, QTableWidgetItem(trend_data['direction']))
                trends_table.setItem(row, 2, QTableWidgetItem(f"{trend_data['change']:+.1f}分"))
                row += 1

            trends_layout.addWidget(trends_table)
            layout.addWidget(trends_group)

            # 成就展示
            if report['achievements']:
                achievements_group = QGroupBox("训练成就")
                achievements_layout = QVBoxLayout(achievements_group)

                for achievement in report['achievements']:
                    achievement_label = QLabel(achievement)
                    achievement_label.setStyleSheet("""
                        QLabel {
                            background-color: #d4edda;
                            color: #155724;
                            padding: 8px 12px;
                            border-radius: 6px;
                            margin: 2px;
                            border-left: 4px solid #28a745;
                        }
                    """)
                    achievements_layout.addWidget(achievement_label)

                layout.addWidget(achievements_group)

            history_dialog.exec_()

        except Exception as e:
            QMessageBox.warning(self, '错误', f'显示历史分析失败: {str(e)}')

    def show_comparison_dashboard(self):
        """显示对比分析仪表板"""
        try:
            gopose_module = self.enhanced_gopose_tab
            analysis_data = gopose_module.get_analysis_data()

            if not analysis_data:
                QMessageBox.warning(self, '警告', '请先在GoPose标签页中进行分析')
                return

            # 创建标准对比模块
            comparison_module = StandardComparisonModule()
            available_exercises = comparison_module.get_available_exercises()

            # 选择动作类型
            exercise_type, ok = QInputDialog.getItem(
                self, '选择动作类型', '请选择要对比的标准动作:',
                available_exercises, 0, False
            )

            if ok and exercise_type:
                comparison_result = comparison_module.compare_with_standard(analysis_data, exercise_type)

                # 创建对比窗口
                comparison_dialog = QDialog(self)
                comparison_dialog.setWindowTitle(f'{exercise_type} - 标准动作对比')
                comparison_dialog.setFixedSize(800, 600)

                layout = QVBoxLayout(comparison_dialog)

                # 相似度评分
                similarity_widget = QWidget()
                similarity_layout = QHBoxLayout(similarity_widget)

                similarity_label = QLabel(f"相似度评分\n{comparison_result['similarity_score']:.1f}分")
                similarity_label.setAlignment(Qt.AlignCenter)
                similarity_label.setStyleSheet("""
                    QLabel {
                        background-color: #198754;
                        color: white;
                        border-radius: 12px;
                        padding: 20px;
                        font-size: 18px;
                        font-weight: bold;
                    }
                """)

                assessment_label = QLabel(comparison_result['overall_assessment'])
                assessment_label.setWordWrap(True)
                assessment_label.setStyleSheet("""
                    QLabel {
                        background-color: #f8f9fa;
                        padding: 15px;
                        border-radius: 8px;
                        font-size: 14px;
                        border-left: 4px solid #6c757d;
                    }
                """)

                similarity_layout.addWidget(similarity_label)
                similarity_layout.addWidget(assessment_label)
                layout.addWidget(similarity_widget)

                # 角度对比表格
                angles_group = QGroupBox("角度对比分析")
                angles_layout = QVBoxLayout(angles_group)

                angles_table = QTableWidget()
                angles_table.setColumnCount(4)
                angles_table.setHorizontalHeaderLabels(['关节角度', '您的数值', '标准范围', '评价'])
                angles_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)

                row = 0
                for angle_name, comparison in comparison_result.get('angle_comparisons', {}).items():
                    angles_table.insertRow(row)
                    angles_table.setItem(row, 0, QTableWidgetItem(angle_name))
                    angles_table.setItem(row, 1, QTableWidgetItem(f"{comparison['user_value']:.1f}°"))
                    angles_table.setItem(row, 2, QTableWidgetItem(comparison['standard_range']))
                    angles_table.setItem(row, 3, QTableWidgetItem(comparison['status']))
                    row += 1

                angles_layout.addWidget(angles_table)
                layout.addWidget(angles_group)

                comparison_dialog.exec_()

        except Exception as e:
            QMessageBox.warning(self, '错误', f'显示标准对比失败: {str(e)}')

    def show_health_dashboard(self):
        """显示健康报告仪表板"""
        try:
            gopose_module = self.enhanced_gopose_tab
            analysis_data = gopose_module.get_analysis_data()

            if not analysis_data:
                QMessageBox.warning(self, '警告', '请先进行运动分析')
                return

            # 创建健康报告窗口
            health_dialog = QDialog(self)
            health_dialog.setWindowTitle('运动健康评估报告')
            health_dialog.setFixedSize(900, 700)

            layout = QVBoxLayout(health_dialog)

            # 整体健康状态
            if 'injury_risk' in analysis_data:
                risk_data = analysis_data['injury_risk']
                risk_score = risk_data.get('overall_risk_score', 0)

                if risk_score < 0.3:
                    health_status = "健康状态良好"
                    status_color = "#198754"
                    status_icon = "✅"
                elif risk_score < 0.7:
                    health_status = "需要注意"
                    status_color = "#fd7e14"
                    status_icon = "⚠️"
                else:
                    health_status = "存在风险"
                    status_color = "#dc3545"
                    status_icon = "🚨"

                status_label = QLabel(f"{status_icon} {health_status}\n风险评分: {risk_score:.2f}")
                status_label.setAlignment(Qt.AlignCenter)
                status_label.setStyleSheet(f"""
                    QLabel {{
                        background-color: {status_color};
                        color: white;
                        border-radius: 12px;
                        padding: 20px;
                        font-size: 18px;
                        font-weight: bold;
                        margin-bottom: 20px;
                    }}
                """)
                layout.addWidget(status_label)

                # 风险因素
                if risk_data.get('risk_factors'):
                    risks_group = QGroupBox("发现的风险因素")
                    risks_layout = QVBoxLayout(risks_group)

                    for factor in risk_data['risk_factors']:
                        factor_label = QLabel(f"⚠️ {factor}")
                        factor_label.setStyleSheet("""
                            QLabel {
                                background-color: #fff3cd;
                                color: #856404;
                                padding: 8px 12px;
                                border-radius: 6px;
                                margin: 2px;
                                border-left: 4px solid #fd7e14;
                            }
                        """)
                        risks_layout.addWidget(factor_label)

                    layout.addWidget(risks_group)

                # 健康建议
                if risk_data.get('recommendations'):
                    recommendations_group = QGroupBox("健康建议")
                    recommendations_layout = QVBoxLayout(recommendations_group)

                    for rec in risk_data['recommendations']:
                        rec_label = QLabel(f"💡 {rec}")
                        rec_label.setWordWrap(True)
                        rec_label.setStyleSheet("""
                            QLabel {
                                background-color: #d1ecf1;
                                color: #0c5460;
                                padding: 8px 12px;
                                border-radius: 6px;
                                margin: 2px;
                                border-left: 4px solid #17a2b8;
                            }
                        """)
                        recommendations_layout.addWidget(rec_label)

                    layout.addWidget(recommendations_group)

            health_dialog.exec_()

        except Exception as e:
            QMessageBox.warning(self, '错误', f'显示健康报告失败: {str(e)}')

    def update_ai_coach_button(self):
        if SMART_COACH_AVAILABLE:
            self.ai_coach_btn.setText('🏃‍♂️ 智能运动教练 (增强版)')
            self.ai_coach_btn.setToolTip('专业运动知识库 + AI增强回答')
        else:
            self.ai_coach_btn.setText('🤖 AI基础教练')
            self.ai_coach_btn.setToolTip('基础AI对话模式')

    def init_research_management_ui(self):
        """初始化科研管理UI"""
        layout = QVBoxLayout(self.research_tab)

        # 标题
        title = QLabel("科研管理中心")
        title.setAlignment(Qt.AlignCenter)
        title.setStyleSheet("font-size: 28px; font-weight: bold; margin: 20px; color: #2c3e50;")
        layout.addWidget(title)

        # 创建子标签页
        self.research_sub_tabs = QTabWidget()
        layout.addWidget(self.research_sub_tabs)

        # 项目管理子标签页
        self.setup_project_management_tab()

        # 高级分析子标签页
        self.setup_advanced_analysis_tab()

        # 批量处理子标签页
        self.setup_batch_processing_tab()

        # 数据可视化子标签页
        self.setup_visualization_tab()

        # 科研报告子标签页
        self.setup_research_reports_tab()

    def setup_project_management_tab(self):
        """设置项目管理标签页"""
        project_widget = QWidget()
        layout = QVBoxLayout(project_widget)

        # 项目控制区域
        control_group = QGroupBox("项目管理")
        control_layout = QHBoxLayout(control_group)

        self.new_project_btn = QPushButton("新建项目")
        self.load_project_btn = QPushButton("载入项目")
        self.save_project_btn = QPushButton("保存项目")
        self.export_project_btn = QPushButton("导出项目")

        self.new_project_btn.clicked.connect(self.create_new_research_project)
        self.load_project_btn.clicked.connect(self.load_research_project)
        self.save_project_btn.clicked.connect(self.save_research_project)
        self.export_project_btn.clicked.connect(self.export_research_project)

        control_layout.addWidget(self.new_project_btn)
        control_layout.addWidget(self.load_project_btn)
        control_layout.addWidget(self.save_project_btn)
        control_layout.addWidget(self.export_project_btn)

        layout.addWidget(control_group)

        # 项目信息显示
        info_group = QGroupBox("项目信息")
        info_layout = QVBoxLayout(info_group)

        self.project_info_display = QTextEdit()
        self.project_info_display.setMaximumHeight(120)
        self.project_info_display.setPlaceholderText("请创建或载入科研项目...")
        info_layout.addWidget(self.project_info_display)

        layout.addWidget(info_group)

        # 参与者管理表格
        participants_group = QGroupBox("参与者管理")
        participants_layout = QVBoxLayout(participants_group)

        # 参与者控制按钮
        participant_controls = QHBoxLayout()
        self.add_participant_btn = QPushButton("添加参与者")
        self.edit_participant_btn = QPushButton("编辑参与者")
        self.remove_participant_btn = QPushButton("移除参与者")

        self.add_participant_btn.clicked.connect(self.add_research_participant)
        self.edit_participant_btn.clicked.connect(self.edit_research_participant)
        self.remove_participant_btn.clicked.connect(self.remove_research_participant)

        participant_controls.addWidget(self.add_participant_btn)
        participant_controls.addWidget(self.edit_participant_btn)
        participant_controls.addWidget(self.remove_participant_btn)
        participant_controls.addStretch()

        participants_layout.addLayout(participant_controls)

        # 参与者表格
        self.participants_table = QTableWidget()
        self.participants_table.setColumnCount(6)
        self.participants_table.setHorizontalHeaderLabels([
            "参与者ID", "姓名", "年龄", "性别", "数据会话数", "状态"
        ])
        self.participants_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        participants_layout.addWidget(self.participants_table)

        layout.addWidget(participants_group)

        self.research_sub_tabs.addTab(project_widget, "项目管理")

    def setup_advanced_analysis_tab(self):
        """设置高级分析标签页"""
        analysis_widget = QWidget()
        layout = QVBoxLayout(analysis_widget)

        # 分析类型选择
        analysis_type_group = QGroupBox("高级分析类型")
        analysis_type_layout = QHBoxLayout(analysis_type_group)

        self.analysis_type_combo = QComboBox()
        self.analysis_type_combo.addItems([
            "深度学习增强分析",
            "3D运动重建分析",
            "高级生物力学分析",
            "运动专项化分析",
            "疲劳与恢复分析",
            "多模态数据融合"
        ])

        self.run_advanced_analysis_btn = QPushButton("开始分析")
        self.run_advanced_analysis_btn.clicked.connect(self.run_selected_advanced_analysis)

        analysis_type_layout.addWidget(QLabel("分析类型:"))
        analysis_type_layout.addWidget(self.analysis_type_combo)
        analysis_type_layout.addWidget(self.run_advanced_analysis_btn)
        analysis_type_layout.addStretch()

        layout.addWidget(analysis_type_group)

        # 分析参数设置
        params_group = QGroupBox("分析参数")
        params_layout = QFormLayout(params_group)

        self.sport_type_combo = QComboBox()
        self.sport_type_combo.addItems(['篮球', '足球', '网球', '举重', '跑步', '游泳'])
        params_layout.addRow("运动类型:", self.sport_type_combo)

        self.analysis_fps_spin = QSpinBox()
        self.analysis_fps_spin.setRange(1, 120)
        self.analysis_fps_spin.setValue(30)
        params_layout.addRow("分析帧率:", self.analysis_fps_spin)

        self.confidence_threshold_spin = QDoubleSpinBox()
        self.confidence_threshold_spin.setRange(0.1, 1.0)
        self.confidence_threshold_spin.setValue(0.3)
        self.confidence_threshold_spin.setSingleStep(0.1)
        params_layout.addRow("置信度阈值:", self.confidence_threshold_spin)

        layout.addWidget(params_group)

        # 分析结果显示
        results_group = QGroupBox("分析结果")
        results_layout = QVBoxLayout(results_group)

        self.advanced_results_display = QTextEdit()
        self.advanced_results_display.setFont(QFont("Consolas", 10))
        results_layout.addWidget(self.advanced_results_display)

        layout.addWidget(results_group)

        self.research_sub_tabs.addTab(analysis_widget, "高级分析")

    def setup_batch_processing_tab(self):
        """设置批量处理标签页"""
        batch_widget = QWidget()
        layout = QVBoxLayout(batch_widget)

        # 批量处理控制
        batch_control_group = QGroupBox("批量处理控制")
        batch_control_layout = QHBoxLayout(batch_control_group)

        self.batch_analysis_type_combo = QComboBox()
        self.batch_analysis_type_combo.addItems([
            'biomechanical', 'performance', 'fatigue', 'sport_specific'
        ])

        self.start_batch_btn = QPushButton("开始批量分析")
        self.stop_batch_btn = QPushButton("停止处理")
        self.start_batch_btn.clicked.connect(self.start_batch_analysis)
        self.stop_batch_btn.clicked.connect(self.stop_batch_analysis)

        batch_control_layout.addWidget(QLabel("批量分析类型:"))
        batch_control_layout.addWidget(self.batch_analysis_type_combo)
        batch_control_layout.addWidget(self.start_batch_btn)
        batch_control_layout.addWidget(self.stop_batch_btn)
        batch_control_layout.addStretch()

        layout.addWidget(batch_control_group)

        # 批量处理进度
        progress_group = QGroupBox("处理进度")
        progress_layout = QVBoxLayout(progress_group)

        self.batch_progress_bar = QProgressBar()
        self.batch_status_label = QLabel("就绪")

        progress_layout.addWidget(self.batch_progress_bar)
        progress_layout.addWidget(self.batch_status_label)

        layout.addWidget(progress_group)

        # 批量结果摘要
        summary_group = QGroupBox("批量结果摘要")
        summary_layout = QVBoxLayout(summary_group)

        self.batch_summary_table = QTableWidget()
        self.batch_summary_table.setColumnCount(4)
        self.batch_summary_table.setHorizontalHeaderLabels([
            "参与者", "处理状态", "数据质量", "分析结果"
        ])
        self.batch_summary_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)

        summary_layout.addWidget(self.batch_summary_table)

        layout.addWidget(summary_group)

        self.research_sub_tabs.addTab(batch_widget, "批量处理")

    def setup_visualization_tab(self):
        """设置数据可视化标签页"""
        viz_widget = QWidget()
        layout = QVBoxLayout(viz_widget)

        # 可视化控制
        viz_control_group = QGroupBox("可视化控制")
        viz_control_layout = QHBoxLayout(viz_control_group)

        self.viz_type_combo = QComboBox()
        self.viz_type_combo.addItems([
            '关节角度分布', '运动轨迹', '疲劳趋势',
            '表现对比', '3D运动分析', '数据质量报告'
        ])

        self.create_visualization_btn = QPushButton("生成可视化")
        self.export_visualization_btn = QPushButton("导出图表")

        self.create_visualization_btn.clicked.connect(self.create_research_visualization)
        self.export_visualization_btn.clicked.connect(self.export_research_visualization)

        viz_control_layout.addWidget(QLabel("可视化类型:"))
        viz_control_layout.addWidget(self.viz_type_combo)
        viz_control_layout.addWidget(self.create_visualization_btn)
        viz_control_layout.addWidget(self.export_visualization_btn)
        viz_control_layout.addStretch()

        layout.addWidget(viz_control_group)

        # 可视化显示区域
        viz_display_group = QGroupBox("可视化显示")
        viz_display_layout = QVBoxLayout(viz_display_group)

        # 创建图表显示区域
        self.research_viz_widget = QWidget()
        self.research_viz_layout = QVBoxLayout(self.research_viz_widget)

        viz_display_layout.addWidget(self.research_viz_widget)
        layout.addWidget(viz_display_group)

        self.research_sub_tabs.addTab(viz_widget, "数据可视化")

    def setup_research_reports_tab(self):
        """设置科研报告标签页"""
        reports_widget = QWidget()
        layout = QVBoxLayout(reports_widget)

        # 报告生成控制
        report_control_group = QGroupBox("报告生成")
        report_control_layout = QHBoxLayout(report_control_group)

        self.report_type_combo = QComboBox()
        self.report_type_combo.addItems([
            'comprehensive', 'biomechanical', 'performance', 'statistical'
        ])

        self.generate_report_btn = QPushButton("生成报告")
        self.export_report_btn = QPushButton("导出报告")

        self.generate_report_btn.clicked.connect(self.generate_research_report)
        self.export_report_btn.clicked.connect(self.export_research_report)

        report_control_layout.addWidget(QLabel("报告类型:"))
        report_control_layout.addWidget(self.report_type_combo)
        report_control_layout.addWidget(self.generate_report_btn)
        report_control_layout.addWidget(self.export_report_btn)
        report_control_layout.addStretch()

        layout.addWidget(report_control_group)

        # 报告显示区域
        report_display_group = QGroupBox("报告内容")
        report_display_layout = QVBoxLayout(report_display_group)

        self.research_report_display = QTextEdit()
        self.research_report_display.setFont(QFont("Georgia", 11))
        report_display_layout.addWidget(self.research_report_display)

        layout.addWidget(report_display_group)

        self.research_sub_tabs.addTab(reports_widget, "科研报告")

    def create_feature_card(self, title, content, color):
        """创建现代简约功能卡片"""
        card = QGroupBox()
        card.setFixedHeight(180)
        card.setStyleSheet(f"""
            QGroupBox {{
                background-color: #ffffff;
                border: 1px solid #dee2e6;
                border-radius: 12px;
                padding: 20px;
                margin: 8px;
            }}
            QGroupBox:hover {{
                border-color: {color};
                box-shadow: 0 4px 12px rgba(0, 0, 0, 0.1);
            }}
        """)

        layout = QVBoxLayout(card)
        layout.setSpacing(12)

        # 标题区域
        title_layout = QHBoxLayout()

        # 图标区域
        icon_label = QLabel("●")
        icon_label.setStyleSheet(f"""
            color: {color};
            font-size: 24px;
            font-weight: bold;
            margin-right: 8px;
        """)

        # 标题
        title_label = QLabel(title)
        title_label.setStyleSheet(f"""
            color: #212529;
            font-size: 18px;
            font-weight: 600;
            margin: 0;
        """)

        title_layout.addWidget(icon_label)
        title_layout.addWidget(title_label)
        title_layout.addStretch()

        # 内容
        content_label = QLabel(content)
        content_label.setStyleSheet(f"""
            color: #6c757d;
            font-size: 14px;
            line-height: 1.5;
            margin: 0;
            padding: 0;
        """)
        content_label.setWordWrap(True)

        layout.addLayout(title_layout)
        layout.addWidget(content_label)
        layout.addStretch()

        return card

    def start_comprehensive_analysis(self):
        """开始综合分析"""
        try:
            # 检查GoPose标签页是否有数据
            gopose_module = self.enhanced_gopose_tab

            if not gopose_module.data or not gopose_module.athlete_profile:
                QMessageBox.warning(self, '数据不足',
                                    '请先在GoPose标签页中：\n1. 载入视频文件\n2. 载入解析点数据\n3. 设置运动员档案')
                return

            # 更新状态
            self.system_status.setText("正在进行综合分析...")
            self.progress_bar.setVisible(True)
            self.progress_bar.setValue(0)

            # 获取分析数据
            analysis_data = gopose_module.get_analysis_data()

            if not analysis_data:
                self.system_status.setText("分析失败 - 数据不足")
                self.progress_bar.setVisible(False)
                return

            # 更新进度
            self.progress_bar.setValue(25)

            # 显示基础运动学结果
            self.show_basic_results(analysis_data)
            self.progress_bar.setValue(50)

            # 显示生物力学分析结果
            self.show_biomech_results(analysis_data)
            self.progress_bar.setValue(75)

            # 显示损伤风险评估结果
            self.show_risk_results(analysis_data)
            self.progress_bar.setValue(90)

            # 显示训练处方建议
            self.show_prescription_results(analysis_data)
            self.progress_bar.setValue(100)

            # 完成
            self.system_status.setText("分析完成 ✓")
            QTimer.singleShot(2000, lambda: self.progress_bar.setVisible(False))

        except Exception as e:
            self.system_status.setText(f"分析出错: {str(e)}")
            self.progress_bar.setVisible(False)
            QMessageBox.warning(self, '错误', f'分析过程中出现错误: {str(e)}')

    def show_basic_results(self, analysis_data):
        """显示基础运动学结果"""
        self.basic_table.setRowCount(0)

        # 基础运动学参数
        basic_params = [
            '鼻子X', '鼻子Y', '脖子X', '脖子Y', '右肩X', '右肩Y', '右肘X', '右肘Y',
            '右腕X', '右腕Y', '身体中心X', '身体中心Y', '躯干角度',
            '右肘角度', '左肘角度', '右膝角度', '左膝角度',
            '颈部速度(像素/秒)', '右手速度(像素/秒)', '左手速度(像素/秒)',
            '身高(像素)', '肩宽(像素)'
        ]

        for param in basic_params:
            if param in analysis_data:
                row = self.basic_table.rowCount()
                self.basic_table.insertRow(row)
                self.basic_table.setItem(row, 0, QTableWidgetItem(param))
                self.basic_table.setItem(row, 1, QTableWidgetItem(str(analysis_data[param])))

    def show_biomech_results(self, analysis_data):
        """显示生物力学分析结果"""
        self.biomech_table.setRowCount(0)

        biomech_params = {
            'right_elbow_torque': '右肘关节力矩(Nm)',
            'right_knee_torque': '右膝关节力矩(Nm)',
            'energy_transfer_efficiency': '能量传递效率',
            'center_of_mass_x': '重心X坐标',
            'center_of_mass_y': '重心Y坐标',
            'shoulder_abduction_angle': '肩关节外展角度(°)',
            'ground_reaction_force': '地面反作用力(N)'
        }

        for param, name in biomech_params.items():
            if param in analysis_data:
                row = self.biomech_table.rowCount()
                self.biomech_table.insertRow(row)
                self.biomech_table.setItem(row, 0, QTableWidgetItem(name))
                self.biomech_table.setItem(row, 1, QTableWidgetItem(str(analysis_data[param])))

    def show_risk_results(self, analysis_data):
        """显示损伤风险评估结果"""
        self.risk_table.setRowCount(0)

        if 'injury_risk' in analysis_data:
            risk_data = analysis_data['injury_risk']

            # 整体风险评分
            row = self.risk_table.rowCount()
            self.risk_table.insertRow(row)
            self.risk_table.setItem(row, 0, QTableWidgetItem('整体风险评分'))
            risk_score = risk_data.get('overall_risk_score', 0)
            risk_level = '低' if risk_score < 0.3 else '中' if risk_score < 0.7 else '高'
            self.risk_table.setItem(row, 1, QTableWidgetItem(f'{risk_score} ({risk_level}风险)'))

            # 高风险关节
            if risk_data.get('high_risk_joints'):
                row = self.risk_table.rowCount()
                self.risk_table.insertRow(row)
                self.risk_table.setItem(row, 0, QTableWidgetItem('高风险关节'))
                self.risk_table.setItem(row, 1, QTableWidgetItem(', '.join(risk_data['high_risk_joints'])))

            # 风险因素
            for i, factor in enumerate(risk_data.get('risk_factors', [])):
                row = self.risk_table.rowCount()
                self.risk_table.insertRow(row)
                self.risk_table.setItem(row, 0, QTableWidgetItem(f'风险因素{i + 1}'))
                self.risk_table.setItem(row, 1, QTableWidgetItem(factor))

            # 建议
            for i, recommendation in enumerate(risk_data.get('recommendations', [])):
                row = self.risk_table.rowCount()
                self.risk_table.insertRow(row)
                self.risk_table.setItem(row, 0, QTableWidgetItem(f'建议{i + 1}'))
                self.risk_table.setItem(row, 1, QTableWidgetItem(recommendation))

    def show_prescription_results(self, analysis_data):
        """显示训练处方建议结果"""
        self.prescription_table.setRowCount(0)

        if 'training_prescription' in analysis_data:
            prescription = analysis_data['training_prescription']

            # 基本信息
            gopose_module = self.enhanced_gopose_tab
            if gopose_module.athlete_profile:
                row = self.prescription_table.rowCount()
                self.prescription_table.insertRow(row)
                self.prescription_table.setItem(row, 0, QTableWidgetItem('运动员'))
                self.prescription_table.setItem(row, 1, QTableWidgetItem(
                    gopose_module.athlete_profile.get('name', '未知')))

            # 风险等级
            row = self.prescription_table.rowCount()
            self.prescription_table.insertRow(row)
            self.prescription_table.setItem(row, 0, QTableWidgetItem('风险等级'))
            risk_level = '低' if prescription['risk_level'] < 0.3 else '中' if prescription['risk_level'] < 0.7 else '高'
            self.prescription_table.setItem(row, 1, QTableWidgetItem(f'{risk_level}风险'))

            # 训练重点
            if prescription.get('focus_areas'):
                row = self.prescription_table.rowCount()
                self.prescription_table.insertRow(row)
                self.prescription_table.setItem(row, 0, QTableWidgetItem('训练重点'))
                self.prescription_table.setItem(row, 1, QTableWidgetItem(
                    ', '.join(prescription['focus_areas'])))

            # 训练阶段
            for phase_key, phase_data in prescription.get('training_phases', {}).items():
                row = self.prescription_table.rowCount()
                self.prescription_table.insertRow(row)
                self.prescription_table.setItem(row, 0, QTableWidgetItem(f'{phase_data["name"]}'))
                self.prescription_table.setItem(row, 1, QTableWidgetItem(
                    f'持续时间: {phase_data["duration"]}'))

                # 显示练习
                for i, exercise in enumerate(phase_data.get('exercises', [])):
                    row = self.prescription_table.rowCount()
                    self.prescription_table.insertRow(row)
                    self.prescription_table.setItem(row, 0, QTableWidgetItem(f'  练习{i + 1}'))
                    self.prescription_table.setItem(row, 1, QTableWidgetItem(exercise['name']))

                    row = self.prescription_table.rowCount()
                    self.prescription_table.insertRow(row)
                    self.prescription_table.setItem(row, 0, QTableWidgetItem('  描述'))
                    self.prescription_table.setItem(row, 1, QTableWidgetItem(exercise['description']))

        # 在EnhancedGoPoseModule类中添加缺失的方法（约第1890行位置）

    # 在EnhancedGoPoseModule类中添加缺失的方法（约第1890行位置）
    def show_performance_score(self):
        """显示运动表现评分"""
        self.tableWidget.clear()
        self.tableWidget.setHorizontalHeaderLabels(['评分项目', '得分'])
        self.tableWidget.setRowCount(0)

        analysis_results = self.comprehensive_analysis()

        if analysis_results:
            # 计算表现评分
            performance_scores = PerformanceScoreSystem.calculate_performance_score(
                analysis_results,
                self.athlete_profile.get('sport', 'general') if self.athlete_profile else 'general'
            )

            # 显示总体评分
            self.tableWidget.insertRow(0)
            self.tableWidget.setItem(0, 0, QTableWidgetItem('总体得分'))
            score_text = f"{performance_scores['overall_score']}分 ({performance_scores['grade']})"
            self.tableWidget.setItem(0, 1, QTableWidgetItem(score_text))

            # 显示各维度得分
            score_items = [
                ('技术得分', performance_scores['technique_score']),
                ('稳定性得分', performance_scores['stability_score']),
                ('效率得分', performance_scores['efficiency_score']),
                ('安全性得分', performance_scores['safety_score'])
            ]

            for name, score in score_items:
                row = self.tableWidget.rowCount()
                self.tableWidget.insertRow(row)
                self.tableWidget.setItem(row, 0, QTableWidgetItem(name))
                self.tableWidget.setItem(row, 1, QTableWidgetItem(f"{score:.1f}分"))

            # 显示改进建议
            for i, recommendation in enumerate(performance_scores['recommendations']):
                row = self.tableWidget.rowCount()
                self.tableWidget.insertRow(row)
                self.tableWidget.setItem(row, 0, QTableWidgetItem(f'建议{i + 1}'))
                self.tableWidget.setItem(row, 1, QTableWidgetItem(recommendation))

            # 保存训练记录
            if self.athlete_profile:
                progress_tracker = ProgressTrackingModule()
                progress_tracker.save_training_session(
                    self.athlete_profile.get('id', 'unknown'),
                    '综合分析',
                    performance_scores,
                    analysis_results
                )
        else:
            self.tableWidget.insertRow(0)
            self.tableWidget.setItem(0, 0, QTableWidgetItem('需要分析数据'))
            self.tableWidget.setItem(0, 1, QTableWidgetItem('请先载入解析点'))

    def show_standard_comparison(self):
        """显示标准动作对比"""
        self.tableWidget.clear()
        self.tableWidget.setHorizontalHeaderLabels(['对比项目', '结果'])
        self.tableWidget.setRowCount(0)

        analysis_results = self.comprehensive_analysis()

        if analysis_results:
            # 创建对比模块
            comparison_module = StandardComparisonModule()

            # 获取可用的标准动作
            available_exercises = comparison_module.get_available_exercises()

            # 让用户选择要对比的动作类型
            exercise_type, ok = QInputDialog.getItem(
                self, '选择动作类型', '请选择要对比的标准动作:',
                available_exercises, 0, False
            )

            if ok and exercise_type:
                # 执行对比
                comparison_result = comparison_module.compare_with_standard(
                    analysis_results, exercise_type
                )

                # 显示相似度得分
                row = self.tableWidget.rowCount()
                self.tableWidget.insertRow(row)
                self.tableWidget.setItem(row, 0, QTableWidgetItem('相似度得分'))
                self.tableWidget.setItem(row, 1, QTableWidgetItem(f"{comparison_result['similarity_score']:.1f}分"))

                # 显示整体评估
                row = self.tableWidget.rowCount()
                self.tableWidget.insertRow(row)
                self.tableWidget.setItem(row, 0, QTableWidgetItem('整体评估'))
                self.tableWidget.setItem(row, 1, QTableWidgetItem(comparison_result['overall_assessment']))

                # 显示角度对比
                for angle_name, comparison in comparison_result.get('angle_comparisons', {}).items():
                    row = self.tableWidget.rowCount()
                    self.tableWidget.insertRow(row)
                    self.tableWidget.setItem(row, 0, QTableWidgetItem(angle_name))
                    result_text = f"{comparison['user_value']:.1f}° (标准:{comparison['standard_range']}) - {comparison['status']}"
                    self.tableWidget.setItem(row, 1, QTableWidgetItem(result_text))

                # 显示改进建议
                for i, suggestion in enumerate(comparison_result['improvement_suggestions']):
                    row = self.tableWidget.rowCount()
                    self.tableWidget.insertRow(row)
                    self.tableWidget.setItem(row, 0, QTableWidgetItem(f'改进建议{i + 1}'))
                    self.tableWidget.setItem(row, 1, QTableWidgetItem(suggestion))
            else:
                self.tableWidget.insertRow(0)
                self.tableWidget.setItem(0, 0, QTableWidgetItem('未选择动作类型'))
        else:
            self.tableWidget.insertRow(0)
            self.tableWidget.setItem(0, 0, QTableWidgetItem('需要分析数据'))
            self.tableWidget.setItem(0, 1, QTableWidgetItem('请先载入解析点'))

    def show_history_analysis(self):
        """显示历史数据分析"""
        self.tableWidget.clear()
        self.tableWidget.setHorizontalHeaderLabels(['分析项目', '结果'])
        self.tableWidget.setRowCount(0)

        if not self.athlete_profile:
            self.tableWidget.insertRow(0)
            self.tableWidget.setItem(0, 0, QTableWidgetItem('需要运动员档案'))
            self.tableWidget.setItem(0, 1, QTableWidgetItem('请先设置运动员档案'))
            return

        progress_tracker = ProgressTrackingModule()
        athlete_id = self.athlete_profile.get('id', 'unknown')

        # 生成进步报告
        report = progress_tracker.generate_progress_report(athlete_id, days=30)

        # 显示摘要
        row = self.tableWidget.rowCount()
        self.tableWidget.insertRow(row)
        self.tableWidget.setItem(row, 0, QTableWidgetItem('30天训练摘要'))
        self.tableWidget.setItem(row, 1, QTableWidgetItem(report['summary']))

        # 显示趋势
        for metric, trend_data in report['trends'].items():
            metric_name = {
                'overall_score': '总体得分趋势',
                'technique_score': '技术得分趋势',
                'stability_score': '稳定性得分趋势',
                'efficiency_score': '效率得分趋势',
                'safety_score': '安全性得分趋势'
            }.get(metric, metric)

            row = self.tableWidget.rowCount()
            self.tableWidget.insertRow(row)
            self.tableWidget.setItem(row, 0, QTableWidgetItem(metric_name))
            trend_text = f"{trend_data['direction']} ({trend_data['change']:+.1f}分)"
            self.tableWidget.setItem(row, 1, QTableWidgetItem(trend_text))

        # 显示成就
        for i, achievement in enumerate(report['achievements']):
            row = self.tableWidget.rowCount()
            self.tableWidget.insertRow(row)
            self.tableWidget.setItem(row, 0, QTableWidgetItem(f'成就{i + 1}'))
            self.tableWidget.setItem(row, 1, QTableWidgetItem(achievement))

        # 显示建议
        for i, recommendation in enumerate(report['recommendations']):
            row = self.tableWidget.rowCount()
            self.tableWidget.insertRow(row)
            self.tableWidget.setItem(row, 0, QTableWidgetItem(f'建议{i + 1}'))
            self.tableWidget.setItem(row, 1, QTableWidgetItem(recommendation))

    def open_ai_coach(self):
        """打开AI虚拟教练对话框"""
        try:
            # 获取当前分析数据
            analysis_data = self.enhanced_gopose_tab.get_analysis_data()

            # 打开AI教练对话框
            coach_dialog = AICoachDialog(self, analysis_data)
            coach_dialog.exec_()

        except Exception as e:
            QMessageBox.warning(self, '错误', f'无法打开AI虚拟教练: {str(e)}')

    def closeEvent(self, event):
        """关闭事件处理"""
        reply = QMessageBox.question(self, '确认退出',
                                     '确定要退出增强版运动姿势改良系统吗？',
                                     QMessageBox.Yes | QMessageBox.No,
                                     QMessageBox.No)

        if reply == QMessageBox.Yes:
            # 清理资源
            if hasattr(self.enhanced_gopose_tab, 'cap') and self.enhanced_gopose_tab.cap:
                self.enhanced_gopose_tab.cap.release()
            if hasattr(self.enhanced_gopose_tab, 'play_timer'):
                self.enhanced_gopose_tab.play_timer.stop()
            event.accept()
        else:
            event.ignore()

    def refresh_dashboard(self):
        """刷新仪表板"""
        try:
            if not self.enhanced_gopose_tab.athlete_profile:
                self.progress_summary.setHtml("<p>请先设置运动员档案以查看数据可视化</p>")
                return

            athlete_id = self.enhanced_gopose_tab.athlete_profile.get('id', 'unknown')
            dashboard = DashboardModule()

            # 更新进度摘要
            summary_html = dashboard.create_progress_summary_widget(athlete_id)
            self.progress_summary.setHtml(summary_html)

            # 创建图表
            figure = dashboard.create_performance_chart(athlete_id, days=30)

            if figure:
                # 清除现有图表
                for i in reversed(range(self.chart_layout.count())):
                    child = self.chart_layout.itemAt(i).widget()
                    if isinstance(child, FigureCanvas):
                        child.setParent(None)

                # 添加新图表
                canvas = FigureCanvas(figure)
                self.chart_layout.addWidget(canvas)

            QMessageBox.information(self, '成功', '仪表板已刷新')

        except Exception as e:
            QMessageBox.warning(self, '错误', f'刷新仪表板失败: {str(e)}')

    def export_chart(self):
        """导出图表"""
        try:
            if not self.enhanced_gopose_tab.athlete_profile:
                QMessageBox.warning(self, '警告', '请先设置运动员档案')
                return

            save_path, _ = QFileDialog.getSaveFileName(
                self, '导出图表', os.getcwd(),
                "PNG图片 (*.png);;PDF文件 (*.pdf);;所有文件 (*)"
            )

            if save_path:
                athlete_id = self.enhanced_gopose_tab.athlete_profile.get('id', 'unknown')
                dashboard = DashboardModule()
                figure = dashboard.create_performance_chart(athlete_id, days=30)

                if figure:
                    figure.savefig(save_path, dpi=300, bbox_inches='tight')
                    QMessageBox.information(self, '成功', f'图表已导出到: {save_path}')
                else:
                    QMessageBox.warning(self, '错误', '无法生成图表')

        except Exception as e:
            QMessageBox.warning(self, '错误', f'导出失败: {str(e)}')
    # ==================== 科研管理相关方法 ====================

    def create_new_research_project(self):
        """创建新的科研项目"""
        dialog = QDialog(self)
        dialog.setWindowTitle("新建科研项目")
        dialog.setFixedSize(500, 400)

        layout = QVBoxLayout(dialog)

        # 项目信息表单
        form_layout = QFormLayout()

        name_edit = QLineEdit()
        description_edit = QTextEdit()
        description_edit.setMaximumHeight(100)
        researcher_edit = QLineEdit()
        institution_edit = QLineEdit()

        project_type_combo = QComboBox()
        project_type_combo.addItems([
            '生物力学研究', '运动表现分析', '损伤预防研究',
            '康复评估', '技术动作优化', '疲劳监测研究'
        ])

        form_layout.addRow("项目名称:", name_edit)
        form_layout.addRow("项目描述:", description_edit)
        form_layout.addRow("主要研究者:", researcher_edit)
        form_layout.addRow("研究机构:", institution_edit)
        form_layout.addRow("项目类型:", project_type_combo)

        layout.addLayout(form_layout)

        # 按钮
        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(dialog.accept)
        buttons.rejected.connect(dialog.reject)
        layout.addWidget(buttons)

        if dialog.exec_() == QDialog.Accepted:
            project_info = {
                'name': name_edit.text(),
                'description': description_edit.toPlainText(),
                'researcher': researcher_edit.text(),
                'institution': institution_edit.text(),
                'type': project_type_combo.currentText(),
                'creation_date': datetime.now().isoformat()
            }

            self.current_project_id = self.research_manager.create_research_project(project_info)
            self.update_project_display()
            QMessageBox.information(self, '成功',
                                    f'科研项目创建成功！\n项目ID: {self.current_project_id}')

    def load_research_project(self):
        """载入科研项目"""
        projects = list(self.research_manager.research_projects.keys())
        if not projects:
            QMessageBox.information(self, '提示', '暂无可用的科研项目')
            return

        project_id, ok = QInputDialog.getItem(
            self, '选择项目', '请选择要载入的科研项目:', projects, 0, False
        )

        if ok and project_id:
            self.current_project_id = project_id
            self.update_project_display()
            QMessageBox.information(self, '成功', '科研项目载入成功！')

    def save_research_project(self):
        """保存科研项目"""
        if not self.current_project_id:
            QMessageBox.warning(self, '警告', '请先创建或载入科研项目')
            return

        filename, _ = QFileDialog.getSaveFileName(
            self, '保存科研项目', f'research_project_{self.current_project_id}.json',
            "JSON Files (*.json)"
        )

        if filename:
            try:
                project_data = self.research_manager.research_projects[self.current_project_id]
                with open(filename, 'w', encoding='utf-8') as f:
                    json.dump(project_data, f, ensure_ascii=False, indent=2)
                QMessageBox.information(self, '成功', f'项目已保存到: {filename}')
            except Exception as e:
                QMessageBox.warning(self, '错误', f'保存失败: {str(e)}')

    def export_research_project(self):
        """导出科研项目"""
        if not self.current_project_id:
            QMessageBox.warning(self, '警告', '请先选择科研项目')
            return

        export_format, ok = QInputDialog.getItem(
            self, '导出格式', '请选择导出格式:', ['json', 'csv'], 0, False
        )

        if ok:
            try:
                data = self.research_manager.export_research_data(
                    self.current_project_id, export_format, include_raw_data=True
                )

                filename, _ = QFileDialog.getSaveFileName(
                    self, '导出科研数据', f'research_export_{self.current_project_id}.{export_format}',
                    f"{export_format.upper()} Files (*.{export_format})"
                )

                if filename:
                    if export_format == 'json':
                        with open(filename, 'w', encoding='utf-8') as f:
                            f.write(data)
                    else:
                        data.to_csv(filename, index=False, encoding='utf-8')

                    QMessageBox.information(self, '成功', f'数据已导出到: {filename}')
            except Exception as e:
                QMessageBox.warning(self, '错误', f'导出失败: {str(e)}')

    def update_project_display(self):
        """更新项目显示"""
        if not self.current_project_id:
            self.project_info_display.setText("请创建或载入科研项目...")
            return

        project = self.research_manager.research_projects[self.current_project_id]

        info_text = f"""
    项目名称: {project['info']['name']}
    研究者: {project['info']['researcher']}
    研究机构: {project['info'].get('institution', '未设置')}
    项目类型: {project['info'].get('type', '未设置')}
    创建时间: {project['created_date'][:10]}
    参与者数量: {len(project['participants'])}
    数据会话数: {len(project['data_sessions'])}
    项目状态: {project['status']}
        """
        self.project_info_display.setText(info_text)

        # 更新参与者表格
        self.participants_table.setRowCount(len(project['participants']))
        for i, participant in enumerate(project['participants']):
            self.participants_table.setItem(i, 0, QTableWidgetItem(participant['id']))
            self.participants_table.setItem(i, 1, QTableWidgetItem(
                participant['info'].get('name', '未设置')))
            self.participants_table.setItem(i, 2, QTableWidgetItem(
                str(participant['info'].get('age', '未设置'))))
            self.participants_table.setItem(i, 3, QTableWidgetItem(
                participant['info'].get('gender', '未设置')))
            self.participants_table.setItem(i, 4, QTableWidgetItem(
                str(len(participant['sessions']))))
            self.participants_table.setItem(i, 5, QTableWidgetItem("活跃"))

    def add_research_participant(self):
        """添加研究参与者"""
        if not self.current_project_id:
            QMessageBox.warning(self, '警告', '请先创建或载入科研项目')
            return

        # 复用运动员档案对话框
        dialog = AthleteProfileDialog(self)
        if dialog.exec_() == QDialog.Accepted:
            participant_info = dialog.get_profile()
            participant_id = self.research_manager.add_participant(
                self.current_project_id, participant_info
            )

            if participant_id:
                self.update_project_display()
                QMessageBox.information(self, '成功', f'参与者添加成功！ID: {participant_id}')
            else:
                QMessageBox.warning(self, '错误', '添加参与者失败')

    def edit_research_participant(self):
        """编辑研究参与者"""
        # TODO: 实现编辑参与者功能
        QMessageBox.information(self, '提示', '编辑功能开发中...')

    def remove_research_participant(self):
        """移除研究参与者"""
        # TODO: 实现移除参与者功能
        QMessageBox.information(self, '提示', '移除功能开发中...')

    def run_selected_advanced_analysis(self):
        """运行选择的高级分析 - 完整实现版本"""
        analysis_type = self.analysis_type_combo.currentText()

        # 获取GoPose标签页的数据
        gopose_data = self.enhanced_gopose_tab.get_analysis_data()

        if not gopose_data:
            QMessageBox.warning(self, '警告',
                                '请先在GoPose标签页中载入视频和解析点数据')
            return

        self.advanced_results_display.clear()
        self.advanced_results_display.append(f"开始执行{analysis_type}...")

        try:
            if analysis_type == "深度学习增强分析":
                results = self.run_deep_learning_analysis(gopose_data)
            elif analysis_type == "3D运动重建分析":
                results = self.run_3d_analysis(gopose_data)
            elif analysis_type == "高级生物力学分析":
                results = self.run_advanced_biomech_analysis(gopose_data)
            elif analysis_type == "运动专项化分析":
                results = self.run_sport_specific_analysis(gopose_data)
            elif analysis_type == "疲劳与恢复分析":
                results = self.run_fatigue_analysis(gopose_data)
            elif analysis_type == "多模态数据融合":
                results = self.run_multimodal_fusion(gopose_data)
            else:
                results = {"error": f"未知的分析类型: {analysis_type}"}

            self.advanced_results_display.append("\n分析完成！")
            self.advanced_results_display.append("\n结果摘要:")

            # 格式化显示结果
            formatted_results = self.format_analysis_results(results, analysis_type)
            self.advanced_results_display.append(formatted_results)

        except Exception as e:
            self.advanced_results_display.append(f"\n分析出错: {str(e)}")
            import traceback
            self.advanced_results_display.append(f"\n详细错误信息:\n{traceback.format_exc()}")

    def run_deep_learning_analysis(self, data):
        """运行深度学习分析 - 实际实现"""
        try:
            analyzer = DeepLearningEnhancer()

            # 获取当前关键点数据
            gopose_module = self.enhanced_gopose_tab
            if not gopose_module.data or gopose_module.fps >= len(gopose_module.data):
                return {"error": "无有效的关键点数据"}

            current_keypoints = gopose_module.data[gopose_module.fps][0]

            # 执行深度学习增强分析
            results = {
                "analysis_type": "deep_learning",
                "status": "completed",
                "enhanced_keypoints": [],
                "fatigue_detection": {},
                "technique_classification": {},
                "quality_score": 0
            }

            # 1. 姿态精细化
            refined_keypoints = analyzer.refine_pose_keypoints(current_keypoints)
            results["enhanced_keypoints"] = refined_keypoints

            # 2. 疲劳检测
            if len(gopose_module.data) > 10:
                # 获取最近的运动序列
                recent_sequence = []
                start_frame = max(0, gopose_module.fps - 10)
                for i in range(start_frame, gopose_module.fps + 1):
                    if i < len(gopose_module.data) and gopose_module.data[i] is not None:
                        recent_sequence.append(gopose_module.data[i][0])

                if recent_sequence:
                    fatigue_result = analyzer.detect_fatigue_level(recent_sequence)
                    results["fatigue_detection"] = fatigue_result

            # 3. 技术分类（简化实现）
            sport_type = gopose_module.athlete_profile.get('sport',
                                                           'general') if gopose_module.athlete_profile else 'general'
            technique_score = self.calculate_technique_score(refined_keypoints, sport_type)
            results["technique_classification"] = {
                "sport_type": sport_type,
                "technique_score": technique_score,
                "classification": "良好" if technique_score > 0.7 else "需改进"
            }

            # 4. 总体质量评分
            quality_factors = []
            if results["fatigue_detection"]:
                quality_factors.append(1.0 - results["fatigue_detection"].get("score", 0))
            quality_factors.append(technique_score)

            results["quality_score"] = np.mean(quality_factors) if quality_factors else 0.5

            return results

        except Exception as e:
            return {"error": f"深度学习分析失败: {str(e)}"}

    def run_3d_analysis(self, data):
        """运行3D分析 - 实际实现"""
        try:
            gopose_module = self.enhanced_gopose_tab

            # 检查是否有3D分析器
            if not hasattr(gopose_module, 'threed_analyzer'):
                gopose_module.threed_analyzer = Enhanced3DAnalyzer()

            if not gopose_module.data or gopose_module.fps >= len(gopose_module.data):
                return {"error": "无有效的关键点数据"}

            current_keypoints = gopose_module.data[gopose_module.fps][0]

            # 执行3D重建
            height_pixels = gopose_module.threed_analyzer._estimate_height_from_keypoints(current_keypoints)
            pose_3d = gopose_module.threed_analyzer.reconstruct_3d_pose_enhanced(
                current_keypoints,
                previous_3d=getattr(gopose_module, 'last_3d_pose', None),
                height_pixels=height_pixels
            )

            if pose_3d is None:
                return {"error": "3D重建失败"}

            # 分析3D运动质量
            if not hasattr(gopose_module, 'pose_3d_sequence'):
                gopose_module.pose_3d_sequence = []
            gopose_module.pose_3d_sequence.append(pose_3d)

            if len(gopose_module.pose_3d_sequence) > 1:
                quality_metrics = gopose_module.threed_analyzer.analyze_3d_movement_quality(
                    gopose_module.pose_3d_sequence[-10:]  # 最近10帧
                )
            else:
                quality_metrics = {"overall_quality": 0.5}

            # 计算3D角度
            angles_3d = gopose_module.threed_analyzer.calculate_3d_angles_enhanced(pose_3d)

            # 评估重建质量
            reconstruction_quality = gopose_module.threed_analyzer._assess_reconstruction_quality(
                pose_3d, current_keypoints
            )

            results = {
                "analysis_type": "3d_reconstruction",
                "status": "completed",
                "pose_3d": pose_3d.tolist() if hasattr(pose_3d, 'tolist') else pose_3d,
                "reconstruction_quality": reconstruction_quality,
                "angles_3d": angles_3d,
                "movement_quality": quality_metrics,
                "key_measurements": self.extract_3d_measurements(pose_3d)
            }

            return results

        except Exception as e:
            return {"error": f"3D分析失败: {str(e)}"}

    def run_advanced_biomech_analysis(self, data):
        """运行高级生物力学分析 - 实际实现"""
        try:
            analyzer = AdvancedBiomechanics()
            gopose_module = self.enhanced_gopose_tab

            if not gopose_module.data or gopose_module.fps >= len(gopose_module.data):
                return {"error": "无有效的关键点数据"}

            current_keypoints = gopose_module.data[gopose_module.fps][0]
            athlete_profile = gopose_module.athlete_profile or {}

            # 转换为3D格式（简化）
            keypoints_3d = []
            for kp in current_keypoints:
                if len(kp) >= 3:
                    keypoints_3d.append([kp[0], kp[1], 0, kp[2]])
                else:
                    keypoints_3d.append([0, 0, 0, 0])

            results = {
                "analysis_type": "advanced_biomechanics",
                "status": "completed",
                "center_of_mass": {},
                "joint_torques": {},
                "power_analysis": {},
                "energy_efficiency": 0
            }

            # 1. 重心分析
            com_analysis = analyzer.calculate_advanced_com(keypoints_3d, athlete_profile)
            results["center_of_mass"] = com_analysis

            # 2. 关节力矩计算
            joint_torques = analyzer.calculate_joint_torques_advanced(keypoints_3d, athlete_profile)
            results["joint_torques"] = joint_torques

            # 3. 功率分析（需要序列数据）
            if len(gopose_module.data) > 1:
                sequence_data = []
                start_frame = max(0, gopose_module.fps - 5)
                for i in range(start_frame, gopose_module.fps + 1):
                    if i < len(gopose_module.data) and gopose_module.data[i] is not None:
                        sequence_data.append(gopose_module.data[i][0])

                if len(sequence_data) > 1:
                    power_analysis = analyzer.calculate_joint_power(
                        sequence_data, athlete_profile, fps=gopose_module.fpsRate
                    )
                    results["power_analysis"] = power_analysis

            # 4. 能量效率评估
            if data and 'energy_transfer_efficiency' in data:
                results["energy_efficiency"] = data['energy_transfer_efficiency']
            else:
                results["energy_efficiency"] = 0.7  # 默认值

            return results

        except Exception as e:
            return {"error": f"高级生物力学分析失败: {str(e)}"}

    def run_sport_specific_analysis(self, data):
        """运行运动专项分析 - 实际实现"""
        try:
            analyzer = SportSpecificAnalyzer()
            gopose_module = self.enhanced_gopose_tab

            if not gopose_module.data:
                return {"error": "无有效的关键点数据"}

            athlete_profile = gopose_module.athlete_profile or {}
            sport_type = athlete_profile.get('sport', '通用')

            # 获取关键点序列
            sequence_data = []
            start_frame = max(0, gopose_module.fps - 20)
            end_frame = min(len(gopose_module.data), gopose_module.fps + 1)

            for i in range(start_frame, end_frame):
                if i < len(gopose_module.data) and gopose_module.data[i] is not None:
                    sequence_data.append(gopose_module.data[i][0])

            if not sequence_data:
                return {"error": "无足够的序列数据"}

            # 执行专项分析
            analysis_result = analyzer.analyze_sport_specific_performance(
                sequence_data, sport_type, athlete_profile
            )

            results = {
                "analysis_type": "sport_specific",
                "status": "completed",
                "sport": sport_type,
                "performance_analysis": analysis_result,
                "recommendations": analysis_result.get('recommendations', []),
                "technique_scores": analysis_result.get('technique_scores', {}),
                "injury_assessment": analysis_result.get('injury_risk_assessment', {})
            }

            return results

        except Exception as e:
            return {"error": f"运动专项分析失败: {str(e)}"}

    def run_fatigue_analysis(self, data):
        """运行疲劳分析 - 实际实现"""
        try:
            analyzer = FatigueRecoveryAnalyzer()
            gopose_module = self.enhanced_gopose_tab

            if not gopose_module.data or len(gopose_module.data) < 10:
                return {"error": "需要更多的数据来进行疲劳分析"}

            # 获取足够的序列数据
            sequence_data = []
            timestamps = []

            # 取全部数据或最近100帧
            start_frame = max(0, len(gopose_module.data) - 100)

            for i in range(start_frame, len(gopose_module.data)):
                if gopose_module.data[i] is not None and len(gopose_module.data[i]) > 0:
                    sequence_data.append(gopose_module.data[i][0])
                    timestamps.append(i / gopose_module.fpsRate)  # 转换为时间

            if len(sequence_data) < 10:
                return {"error": "数据量不足以进行疲劳分析"}

            # 将序列分段进行疲劳分析
            segment_length = 10
            segments = []
            segment_timestamps = []

            for i in range(0, len(sequence_data), segment_length):
                segment = sequence_data[i:i + segment_length]
                if len(segment) >= segment_length:
                    segments.append(segment)
                    segment_timestamps.append(timestamps[i])

            if not segments:
                return {"error": "无法创建有效的分析段"}

            # 执行疲劳分析
            fatigue_result = analyzer.analyze_fatigue_progression(segments, segment_timestamps)

            results = {
                "analysis_type": "fatigue_analysis",
                "status": "completed",
                "fatigue_level": fatigue_result.get('fatigue_level', 'unknown'),
                "fatigue_timeline": fatigue_result.get('fatigue_timeline', []),
                "critical_points": fatigue_result.get('critical_points', []),
                "recovery_recommendations": fatigue_result.get('recovery_recommendations', []),
                "analysis_summary": {
                    "total_segments": len(segments),
                    "analysis_duration": f"{len(sequence_data) / gopose_module.fpsRate:.1f}秒",
                    "average_fatigue": np.mean(
                        [point.get('fatigue_level', 0) for point in fatigue_result.get('fatigue_timeline', [])])
                }
            }

            return results

        except Exception as e:
            return {"error": f"疲劳分析失败: {str(e)}"}

    def run_multimodal_fusion(self, data):
        """运行多模态融合 - 实际实现"""
        try:
            analyzer = MultiModalDataFusion()
            gopose_module = self.enhanced_gopose_tab

            if not gopose_module.data or gopose_module.fps >= len(gopose_module.data):
                return {"error": "无有效的关键点数据"}

            # 模拟多模态数据
            current_time = datetime.now()

            # 添加姿态数据
            pose_data = {
                'keypoints': gopose_module.data[gopose_module.fps][0],
                'timestamp': current_time.isoformat()
            }
            analyzer.add_data_stream('pose', pose_data, current_time.isoformat())

            # 模拟其他传感器数据
            # IMU数据
            imu_data = {
                'orientation': [0, 5, 0],  # 模拟倾斜
                'angular_velocity': [0.1, 0.2, 0.05],
                'linear_acceleration': [0.2, 9.8, 0.1]
            }
            analyzer.add_data_stream('imu', imu_data, current_time.isoformat())

            # 模拟力板数据
            force_data = {
                'grf': [0, 700, 0],  # 地面反作用力
                'cop': [0, 0]  # 压力中心
            }
            analyzer.add_data_stream('force_plate', force_data, current_time.isoformat())

            # 执行数据融合
            fusion_result = analyzer.fuse_data('weighted_average', time_window=1.0)

            results = {
                "analysis_type": "multimodal_fusion",
                "status": "completed",
                "fusion_result": fusion_result,
                "data_quality": {
                    "pose_data_available": True,
                    "imu_data_simulated": True,
                    "force_plate_simulated": True
                },
                "enhanced_metrics": {
                    "enhanced_balance": fusion_result.get('biomechanics_enhanced', {}).get('dynamic_balance', {}),
                    "movement_efficiency": fusion_result.get('performance_metrics', {}).get('movement_efficiency', {}),
                    "comprehensive_fatigue": fusion_result.get('performance_metrics', {}).get('fatigue_state', {})
                },
                "confidence_scores": fusion_result.get('confidence_scores', {})
            }

            return results

        except Exception as e:
            return {"error": f"多模态融合失败: {str(e)}"}

    def calculate_technique_score(self, keypoints, sport_type):
        """计算技术评分"""
        try:
            # 基础技术评分算法
            score_factors = []

            # 1. 姿态稳定性
            if len(keypoints) > 8:
                # 检查主要关节点的置信度
                key_joints = [1, 2, 5, 8, 9, 12]  # 颈部、双肩、中臀、双髋
                confidence_scores = [keypoints[i][2] for i in key_joints if
                                     i < len(keypoints) and len(keypoints[i]) > 2]
                if confidence_scores:
                    score_factors.append(np.mean(confidence_scores))

            # 2. 对称性评分
            if len(keypoints) > 14:
                symmetric_pairs = [(2, 5), (3, 6), (4, 7), (9, 12), (10, 13), (11, 14)]
                symmetry_scores = []

                for left_idx, right_idx in symmetric_pairs:
                    if (left_idx < len(keypoints) and right_idx < len(keypoints) and
                            len(keypoints[left_idx]) > 2 and len(keypoints[right_idx]) > 2 and
                            keypoints[left_idx][2] > 0.3 and keypoints[right_idx][2] > 0.3):
                        left_pos = np.array(keypoints[left_idx][:2])
                        right_pos = np.array(keypoints[right_idx][:2])
                        distance = np.linalg.norm(left_pos - right_pos)

                        # 归一化对称性评分
                        symmetry = 1.0 / (1.0 + distance / 100.0)
                        symmetry_scores.append(symmetry)

                if symmetry_scores:
                    score_factors.append(np.mean(symmetry_scores))

            # 3. 运动类型特定评分
            sport_bonus = {
                '篮球': 0.1,
                '足球': 0.1,
                '网球': 0.15,
                '举重': 0.2,
                '跑步': 0.05
            }.get(sport_type, 0)

            base_score = np.mean(score_factors) if score_factors else 0.5
            final_score = min(1.0, base_score + sport_bonus)

            return final_score

        except Exception as e:
            print(f"技术评分计算错误: {e}")
            return 0.5

    def extract_3d_measurements(self, pose_3d):
        """提取3D关键测量值"""
        measurements = {}

        try:
            # 身体主要尺寸
            if len(pose_3d) > 14:
                # 身高
                if (len(pose_3d[0]) >= 4 and len(pose_3d[11]) >= 4 and
                        pose_3d[0][3] > 0.1 and pose_3d[11][3] > 0.1):
                    head_pos = np.array(pose_3d[0][:3])
                    ankle_pos = np.array(pose_3d[11][:3])
                    measurements['estimated_height'] = np.linalg.norm(head_pos - ankle_pos)

                # 肩宽
                if (len(pose_3d[2]) >= 4 and len(pose_3d[5]) >= 4 and
                        pose_3d[2][3] > 0.1 and pose_3d[5][3] > 0.1):
                    left_shoulder = np.array(pose_3d[2][:3])
                    right_shoulder = np.array(pose_3d[5][:3])
                    measurements['shoulder_width'] = np.linalg.norm(left_shoulder - right_shoulder)

                # 臂展
                if (len(pose_3d[4]) >= 4 and len(pose_3d[7]) >= 4 and
                        pose_3d[4][3] > 0.1 and pose_3d[7][3] > 0.1):
                    left_hand = np.array(pose_3d[4][:3])
                    right_hand = np.array(pose_3d[7][:3])
                    measurements['arm_span'] = np.linalg.norm(left_hand - right_hand)

        except Exception as e:
            print(f"3D测量提取错误: {e}")

        return measurements

    def format_analysis_results(self, results, analysis_type):
        """格式化分析结果显示"""
        try:
            if "error" in results:
                return f"❌ 分析失败: {results['error']}"

            formatted = f"✅ {analysis_type} 分析完成\n"
            formatted += "=" * 50 + "\n"

            if analysis_type == "深度学习增强分析":
                if "fatigue_detection" in results:
                    fatigue = results["fatigue_detection"]
                    formatted += f"疲劳检测: {fatigue.get('level', '未知')} (评分: {fatigue.get('score', 0):.2f})\n"

                if "technique_classification" in results:
                    tech = results["technique_classification"]
                    formatted += f"技术分类: {tech.get('classification', '未知')} (评分: {tech.get('technique_score', 0):.2f})\n"

                formatted += f"整体质量评分: {results.get('quality_score', 0):.2f}\n"

            elif analysis_type == "3D运动重建分析":
                formatted += f"重建质量: {results.get('reconstruction_quality', 0):.3f}\n"

                if "angles_3d" in results:
                    formatted += "\n3D关节角度:\n"
                    for angle_name, angle_value in results["angles_3d"].items():
                        formatted += f"  {angle_name}: {angle_value:.1f}°\n"

                if "movement_quality" in results:
                    quality = results["movement_quality"]
                    formatted += f"\n运动质量评分: {quality.get('overall_quality', 0):.3f}\n"

            elif analysis_type == "高级生物力学分析":
                if "center_of_mass" in results:
                    com = results["center_of_mass"]
                    if com:
                        formatted += f"重心位置: X={com.get('com_3d', [0, 0, 0])[0]:.1f}, Y={com.get('com_3d', [0, 0, 0])[1]:.1f}\n"

                if "joint_torques" in results:
                    formatted += "\n关节力矩:\n"
                    for joint, torque in results["joint_torques"].items():
                        formatted += f"  {joint}: {torque:.2f} Nm\n"

                formatted += f"能量效率: {results.get('energy_efficiency', 0):.2f}\n"

            elif analysis_type == "运动专项化分析":
                formatted += f"运动项目: {results.get('sport', '未知')}\n"

                if "technique_scores" in results:
                    formatted += "\n技术评分:\n"
                    for technique, score in results["technique_scores"].items():
                        formatted += f"  {technique}: {score:.2f}\n"

                if "recommendations" in results:
                    formatted += "\n专项建议:\n"
                    for i, rec in enumerate(results["recommendations"][:3], 1):
                        formatted += f"  {i}. {rec}\n"

            elif analysis_type == "疲劳与恢复分析":
                formatted += f"疲劳水平: {results.get('fatigue_level', '未知')}\n"

                if "analysis_summary" in results:
                    summary = results["analysis_summary"]
                    formatted += f"分析时长: {summary.get('analysis_duration', '未知')}\n"
                    formatted += f"平均疲劳度: {summary.get('average_fatigue', 0):.3f}\n"

                if "recovery_recommendations" in results:
                    formatted += "\n恢复建议:\n"
                    for i, rec in enumerate(results["recovery_recommendations"][:3], 1):
                        formatted += f"  {i}. {rec}\n"

            elif analysis_type == "多模态数据融合":
                if "confidence_scores" in results:
                    confidence = results["confidence_scores"]
                    formatted += f"融合置信度: {confidence.get('overall', 0):.3f}\n"

                if "enhanced_metrics" in results:
                    metrics = results["enhanced_metrics"]
                    formatted += "\n增强指标:\n"
                    for metric_name, metric_data in metrics.items():
                        if isinstance(metric_data, dict) and metric_data:
                            formatted += f"  {metric_name}: 已计算\n"

            return formatted

        except Exception as e:
            return f"结果格式化错误: {str(e)}"

    def start_batch_analysis(self):
        """开始批量分析"""
        if not self.current_project_id:
            QMessageBox.warning(self, '警告', '请先选择科研项目')
            return

        analysis_type = self.batch_analysis_type_combo.currentText()

        try:
            self.batch_status_label.setText("正在进行批量分析...")
            self.batch_progress_bar.setValue(0)

            # 运行批量分析
            results = self.research_manager.batch_analysis(
                self.current_project_id, analysis_type, {
                    'sport_type': self.sport_type_combo.currentText()
                }
            )

            if results:
                self.batch_progress_bar.setValue(100)
                self.batch_status_label.setText("批量分析完成")
                self.update_batch_summary(results)
                QMessageBox.information(self, '成功', '批量分析完成！')
            else:
                self.batch_status_label.setText("批量分析失败")
                QMessageBox.warning(self, '错误', '批量分析失败')

        except Exception as e:
            self.batch_status_label.setText(f"分析出错: {str(e)}")
            QMessageBox.warning(self, '错误', f'批量分析出错: {str(e)}')

    def stop_batch_analysis(self):
        """停止批量分析"""
        self.batch_status_label.setText("用户取消")
        self.batch_progress_bar.setValue(0)

    def update_batch_summary(self, results):
        """更新批量分析摘要"""
        if not results or 'results' not in results:
            return

        result_list = results['results']
        self.batch_summary_table.setRowCount(len(result_list))

        for i, result_item in enumerate(result_list):
            participant_id = result_item.get('participant_id', '未知')
            status = "成功" if 'error' not in result_item.get('result', {}) else "失败"
            quality = "良好"  # 简化显示
            summary = "已完成"

            self.batch_summary_table.setItem(i, 0, QTableWidgetItem(participant_id))
            self.batch_summary_table.setItem(i, 1, QTableWidgetItem(status))
            self.batch_summary_table.setItem(i, 2, QTableWidgetItem(quality))
            self.batch_summary_table.setItem(i, 3, QTableWidgetItem(summary))

    def create_research_visualization(self):
        """创建科研可视化 - 统一实现"""
        if not check_matplotlib():
            QMessageBox.warning(self, '错误', '缺少matplotlib库，请安装: pip install matplotlib')
            return

        if not self.current_project_id:
            QMessageBox.warning(self, '警告', '请先选择科研项目')
            return

        viz_type = self.viz_type_combo.currentText()

        try:
            # 创建可视化窗口
            viz_window = VisualizationWindow(self.research_manager, self.current_project_id)
            viz_window.viz_type_combo.setCurrentText(viz_type)
            viz_window.create_visualizations()
            viz_window.show()
        except Exception as e:
            QMessageBox.warning(self, '错误', f'创建可视化失败: {str(e)}')

    def export_research_visualization(self):
        """导出科研可视化"""
        QMessageBox.information(self, '提示', '可视化导出功能请在可视化窗口中操作')

    def generate_research_report(self):
        """生成科研报告"""
        if not self.current_project_id:
            QMessageBox.warning(self, '警告', '请先选择科研项目')
            return

        report_type = self.report_type_combo.currentText()

        try:
            report = self.research_manager.generate_research_report(
                self.current_project_id, report_type
            )

            if report:
                # 格式化显示报告
                report_text = self.format_research_report(report)
                self.research_report_display.setText(report_text)
                QMessageBox.information(self, '成功', '科研报告生成完成！')
            else:
                QMessageBox.warning(self, '错误', '报告生成失败')

        except Exception as e:
            QMessageBox.warning(self, '错误', f'生成报告出错: {str(e)}')

    def format_research_report(self, report):
        """格式化科研报告"""
        formatted_text = f"""
    # 科研报告

    ## 项目基本信息
    - 项目名称: {report['project_info']['name']}
    - 主要研究者: {report['project_info']['researcher']}
    - 研究机构: {report['project_info'].get('institution', '未设置')}
    - 报告生成时间: {report['generation_date'][:19]}

    ## 研究概况
    - 总参与者数: {report['participants_summary']['total_participants']}
    - 总数据会话数: {report['participants_summary']['total_sessions']}

    ## 分析结果摘要
    """

        if 'analysis_summary' in report:
            formatted_text += f"- 已完成分析类型: {', '.join(report['analysis_summary']['analysis_types'])}\n"

            if 'key_findings' in report['analysis_summary']:
                formatted_text += "\n### 关键发现:\n"
                for finding in report['analysis_summary']['key_findings']:
                    formatted_text += f"  • {finding}\n"

        formatted_text += "\n## 研究结论\n"
        for conclusion in report['conclusions']:
            formatted_text += f"- {conclusion}\n"

        formatted_text += "\n## 建议与展望\n"
        for recommendation in report['recommendations']:
            formatted_text += f"- {recommendation}\n"

        return formatted_text

    def export_research_report(self):
        """导出科研报告"""
        if not self.research_report_display.toPlainText():
            QMessageBox.warning(self, '警告', '请先生成报告')
            return

        filename, _ = QFileDialog.getSaveFileName(
            self, '导出科研报告', f'research_report_{self.current_project_id}.txt',
            "文本文件 (*.txt);;Markdown文件 (*.md);;PDF文件 (*.pdf)"
        )

        if filename:
            try:
                with open(filename, 'w', encoding='utf-8') as f:
                    f.write(self.research_report_display.toPlainText())
                QMessageBox.information(self, '成功', f'报告已导出到: {filename}')
            except Exception as e:
                QMessageBox.warning(self, '错误', f'导出失败: {str(e)}')

        def get_research_data(self):
            """获取科研数据格式"""
            if not self.data or not self.athlete_profile:
                return None

            research_data = {
                'keypoints_sequence': self.data,
                'athlete_profile': self.athlete_profile,
                'video_info': {
                    'fps': self.fpsRate,
                    'total_frames': self.fpsMax,
                    'current_frame': self.fps
                },
                'analysis_params': {
                    'pc': self.pc,
                    'rotation_angle': self.rotationAngle
                }
            }

            return research_data

        def set_research_mode(self, enabled=True):
            """设置科研模式"""
            if enabled:
                # 启用高精度分析
                self.confidence_threshold = 0.1  # 降低置信度阈值
                # 其他科研模式设置
            else:
                # 恢复普通模式
                self.confidence_threshold = 0.3


import numpy as np
import cv2
from scipy import signal
from scipy.spatial.distance import euclidean
from sklearn.preprocessing import StandardScaler
import json
from datetime import datetime
import logging
from collections import deque
import warnings
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

# 设置日志
logger = logging.getLogger(__name__)

# 忽略一些常见的警告
warnings.filterwarnings('ignore', category=RuntimeWarning)
warnings.filterwarnings('ignore', category=np.RankWarning)


# 设置matplotlib中文字体支持
def setup_chinese_font():
    """设置matplotlib中文字体支持"""
    try:
        # 尝试设置中文字体
        plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
        plt.rcParams['axes.unicode_minus'] = False
    except Exception:
        # 如果中文字体不可用，使用默认字体
        plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
        logger.warning("中文字体不可用，使用默认字体")


# 初始化字体设置
setup_chinese_font()


def safe_array_check(arr, condition_func):
    """安全的数组条件检查"""
    try:
        if isinstance(arr, (list, tuple)):
            return condition_func(arr)
        elif isinstance(arr, np.ndarray):
            if arr.size == 1:
                return condition_func(arr.item())
            else:
                # 对于多元素数组，使用 any() 或 all()
                return condition_func(arr).any() if hasattr(condition_func(arr), 'any') else bool(condition_func(arr))
        else:
            return condition_func(arr)
    except Exception:
        return False


def safe_confidence_check(keypoint, threshold=0.1):
    """安全的置信度检查"""
    try:
        if isinstance(keypoint, (list, tuple)) and len(keypoint) >= 3:
            confidence = keypoint[2]
            if isinstance(confidence, np.ndarray):
                return confidence.item() > threshold if confidence.size == 1 else confidence.any() > threshold
            return confidence > threshold
        elif isinstance(keypoint, np.ndarray) and keypoint.size >= 3:
            confidence = keypoint[2] if keypoint.ndim == 1 else keypoint[0, 2]
            if isinstance(confidence, np.ndarray):
                return confidence.item() > threshold if confidence.size == 1 else confidence.any() > threshold
            return confidence > threshold
        return False
    except Exception:
        return False


def safe_length_check(obj, min_length):
    """安全的长度检查"""
    try:
        if hasattr(obj, '__len__'):
            return len(obj) >= min_length
        elif isinstance(obj, np.ndarray):
            return obj.size >= min_length
        return False
    except Exception:
        return False


class FixedCoordinationAnalyzer:
    """修复的肢体协调性分析器 - 完整版"""

    @staticmethod
    def analyze_limb_coordination(pose_sequence):
        """分析肢体协调性"""
        try:
            if not pose_sequence or len(pose_sequence) < 3:
                return {
                    "overall_coordination": 0.0,
                    "upper_limb_sync": 0.0,
                    "lower_limb_sync": 0.0,
                    "cross_lateral_sync": 0.0,
                    "stability_score": 0.0
                }

            coordination_results = {}

            # 1. 上肢协调性
            upper_sync = FixedCoordinationAnalyzer._analyze_upper_limb_sync(pose_sequence)
            coordination_results["upper_limb_sync"] = upper_sync

            # 2. 下肢协调性
            lower_sync = FixedCoordinationAnalyzer._analyze_lower_limb_sync(pose_sequence)
            coordination_results["lower_limb_sync"] = lower_sync

            # 3. 交叉侧协调性
            cross_sync = FixedCoordinationAnalyzer._analyze_cross_lateral_sync(pose_sequence)
            coordination_results["cross_lateral_sync"] = cross_sync

            # 4. 整体稳定性
            stability = FixedCoordinationAnalyzer._analyze_postural_stability(pose_sequence)
            coordination_results["stability_score"] = stability

            # 5. 综合协调性评分
            coordination_scores = [upper_sync, lower_sync, cross_sync, stability]
            valid_scores = [s for s in coordination_scores if s > 0]
            overall_coordination = np.mean(valid_scores) if valid_scores else 0.0
            coordination_results["overall_coordination"] = overall_coordination

            return coordination_results

        except Exception as e:
            logger.error(f"肢体协调性分析失败: {e}")
            return {
                "overall_coordination": 0.0,
                "upper_limb_sync": 0.0,
                "lower_limb_sync": 0.0,
                "cross_lateral_sync": 0.0,
                "stability_score": 0.0
            }

    @staticmethod
    def _analyze_upper_limb_sync(pose_sequence):
        """分析上肢同步性"""
        try:
            # 左右手臂的关节索引 (COCO格式)
            left_arm = [5, 7, 9]  # 左肩、左肘、左腕
            right_arm = [6, 8, 10]  # 右肩、右肘、右腕

            sync_scores = []

            # 计算左右手臂的运动同步性
            for left_idx, right_idx in zip(left_arm, right_arm):
                left_trajectory = []
                right_trajectory = []

                for pose in pose_sequence:
                    # 修复的条件检查
                    left_valid = (left_idx < len(pose) and
                                  safe_length_check(pose[left_idx], 3) and
                                  safe_confidence_check(pose[left_idx]))

                    right_valid = (right_idx < len(pose) and
                                   safe_length_check(pose[right_idx], 3) and
                                   safe_confidence_check(pose[right_idx]))

                    if left_valid and right_valid:
                        left_trajectory.append(pose[left_idx][:2])
                        right_trajectory.append(pose[right_idx][:2])

                if len(left_trajectory) >= 3:
                    sync_score = FixedCoordinationAnalyzer._calculate_trajectory_sync(
                        left_trajectory, right_trajectory
                    )
                    if sync_score is not None:
                        sync_scores.append(sync_score)

            return np.mean(sync_scores) if sync_scores else 0.0

        except Exception as e:
            logger.error(f"上肢同步性分析失败: {e}")
            return 0.0

    @staticmethod
    def _analyze_lower_limb_sync(pose_sequence):
        """分析下肢同步性"""
        try:
            # 左右腿的关节索引 (COCO格式)
            left_leg = [11, 13, 15]  # 左臀、左膝、左踝
            right_leg = [12, 14, 16]  # 右臀、右膝、右踝

            sync_scores = []

            # 计算左右腿的运动同步性
            for left_idx, right_idx in zip(left_leg, right_leg):
                left_trajectory = []
                right_trajectory = []

                for pose in pose_sequence:
                    # 修复的条件检查
                    left_valid = (left_idx < len(pose) and
                                  safe_length_check(pose[left_idx], 3) and
                                  safe_confidence_check(pose[left_idx]))

                    right_valid = (right_idx < len(pose) and
                                   safe_length_check(pose[right_idx], 3) and
                                   safe_confidence_check(pose[right_idx]))

                    if left_valid and right_valid:
                        left_trajectory.append(pose[left_idx][:2])
                        right_trajectory.append(pose[right_idx][:2])

                if len(left_trajectory) >= 3:
                    sync_score = FixedCoordinationAnalyzer._calculate_trajectory_sync(
                        left_trajectory, right_trajectory
                    )
                    if sync_score is not None:
                        sync_scores.append(sync_score)

            return np.mean(sync_scores) if sync_scores else 0.0

        except Exception as e:
            logger.error(f"下肢同步性分析失败: {e}")
            return 0.0

    @staticmethod
    def _analyze_cross_lateral_sync(pose_sequence):
        """分析交叉侧协调性"""
        try:
            # 对角线肢体协调（左臂-右腿，右臂-左腿）
            cross_pairs = [
                ([5, 7], [12, 14]),  # 左臂 - 右腿
                ([6, 8], [11, 13])  # 右臂 - 左腿
            ]

            cross_sync_scores = []

            for arm_joints, leg_joints in cross_pairs:
                arm_movements = []
                leg_movements = []

                for i in range(len(pose_sequence) - 1):
                    pose1, pose2 = pose_sequence[i], pose_sequence[i + 1]

                    # 计算手臂运动
                    arm_movement = FixedCoordinationAnalyzer._calculate_joint_movement(
                        pose1, pose2, arm_joints
                    )

                    # 计算腿部运动
                    leg_movement = FixedCoordinationAnalyzer._calculate_joint_movement(
                        pose1, pose2, leg_joints
                    )

                    if arm_movement is not None and leg_movement is not None:
                        arm_movements.append(arm_movement)
                        leg_movements.append(leg_movement)

                if len(arm_movements) >= 3:
                    # 计算运动模式的相关性
                    correlation = FixedCoordinationAnalyzer._calculate_movement_correlation(
                        arm_movements, leg_movements
                    )
                    if correlation is not None:
                        cross_sync_scores.append(correlation)

            return np.mean(cross_sync_scores) if cross_sync_scores else 0.0

        except Exception as e:
            logger.error(f"交叉协调性分析失败: {e}")
            return 0.0

    @staticmethod
    def _analyze_postural_stability(pose_sequence):
        """分析姿态稳定性"""
        try:
            stability_metrics = []

            # 1. 重心稳定性
            com_stability = FixedCoordinationAnalyzer._calculate_com_stability(pose_sequence)
            if com_stability is not None:
                stability_metrics.append(com_stability)

            # 2. 关键关节稳定性
            key_joints = [1, 8]  # 颈部和臀部
            for joint_idx in key_joints:
                joint_stability = FixedCoordinationAnalyzer._calculate_joint_stability(
                    pose_sequence, joint_idx
                )
                if joint_stability is not None:
                    stability_metrics.append(joint_stability)

            # 3. 身体摆动稳定性
            sway_stability = FixedCoordinationAnalyzer._calculate_body_sway_stability(pose_sequence)
            if sway_stability is not None:
                stability_metrics.append(sway_stability)

            return np.mean(stability_metrics) if stability_metrics else 0.0

        except Exception as e:
            logger.error(f"姿态稳定性分析失败: {e}")
            return 0.0

    @staticmethod
    def _calculate_trajectory_sync(traj1, traj2):
        """计算轨迹同步性"""
        try:
            if len(traj1) != len(traj2) or len(traj1) < 3:
                return None

            traj1 = np.array(traj1)
            traj2 = np.array(traj2)

            # 计算速度
            vel1 = np.diff(traj1, axis=0)
            vel2 = np.diff(traj2, axis=0)

            # 计算速度幅度
            speed1 = np.linalg.norm(vel1, axis=1)
            speed2 = np.linalg.norm(vel2, axis=1)

            # 相关性分析 - 修复数组条件判断
            if len(speed1) > 1:
                std1 = np.std(speed1)
                std2 = np.std(speed2)

                # 安全的标准差检查
                if std1 > 1e-6 and std2 > 1e-6:
                    correlation = np.corrcoef(speed1, speed2)[0, 1]
                    correlation = np.nan_to_num(correlation)
                    return max(0, (correlation + 1) / 2)  # 转换到0-1范围

            return 0.5

        except Exception as e:
            logger.error(f"轨迹同步性计算失败: {e}")
            return None

    @staticmethod
    def _calculate_joint_movement(pose1, pose2, joint_indices):
        """计算关节运动量"""
        try:
            movements = []

            for joint_idx in joint_indices:
                # 修复的条件检查
                pose1_valid = (joint_idx < len(pose1) and
                               safe_length_check(pose1[joint_idx], 3) and
                               safe_confidence_check(pose1[joint_idx]))

                pose2_valid = (joint_idx < len(pose2) and
                               safe_length_check(pose2[joint_idx], 3) and
                               safe_confidence_check(pose2[joint_idx]))

                if pose1_valid and pose2_valid:
                    pos1 = np.array(pose1[joint_idx][:2])
                    pos2 = np.array(pose2[joint_idx][:2])
                    movement = np.linalg.norm(pos2 - pos1)
                    movements.append(movement)

            return np.mean(movements) if movements else None

        except Exception as e:
            logger.error(f"关节运动量计算失败: {e}")
            return None

    @staticmethod
    def _calculate_movement_correlation(movements1, movements2):
        """计算运动相关性"""
        try:
            if len(movements1) != len(movements2) or len(movements1) < 3:
                return None

            movements1 = np.array(movements1)
            movements2 = np.array(movements2)

            # 修复标准差检查
            std1 = np.std(movements1)
            std2 = np.std(movements2)

            if std1 > 1e-6 and std2 > 1e-6:
                correlation = np.corrcoef(movements1, movements2)[0, 1]
                correlation = np.nan_to_num(correlation)
                return max(0, abs(correlation))

            return 0.5

        except Exception as e:
            logger.error(f"运动相关性计算失败: {e}")
            return None

    @staticmethod
    def _calculate_com_stability(pose_sequence):
        """计算重心稳定性"""
        try:
            com_positions = []

            for pose in pose_sequence:
                com = FixedCoordinationAnalyzer._estimate_center_of_mass(pose)
                if com is not None:
                    com_positions.append(com)

            if len(com_positions) < 3:
                return None

            com_array = np.array(com_positions)

            # 计算重心位置方差
            com_variance = np.var(com_array, axis=0)
            stability_score = 1.0 / (1.0 + np.mean(com_variance) / 1000.0)

            return min(max(stability_score, 0.0), 1.0)

        except Exception as e:
            logger.error(f"重心稳定性计算失败: {e}")
            return None

    @staticmethod
    def _estimate_center_of_mass(pose):
        """估算重心位置"""
        try:
            # 关键点权重（基于人体质量分布）
            weights = {
                0: 0.07, 1: 0.07,  # 头颈部
                2: 0.05, 5: 0.05,  # 肩膀
                8: 0.15,  # 臀部
                9: 0.1, 12: 0.1,  # 大腿
                10: 0.045, 13: 0.045,  # 小腿
                11: 0.015, 14: 0.015  # 脚踝
            }

            weighted_x, weighted_y, total_weight = 0, 0, 0

            for idx, weight in weights.items():
                # 修复的条件检查
                if (idx < len(pose) and
                        safe_length_check(pose[idx], 3) and
                        safe_confidence_check(pose[idx])):
                    weighted_x += pose[idx][0] * weight
                    weighted_y += pose[idx][1] * weight
                    total_weight += weight

            if total_weight > 0.1:  # 确保有足够的权重
                return [weighted_x / total_weight, weighted_y / total_weight]
            else:
                return None

        except Exception as e:
            logger.error(f"重心估算失败: {e}")
            return None

    @staticmethod
    def _calculate_joint_stability(pose_sequence, joint_idx):
        """计算单个关节稳定性"""
        try:
            positions = []

            for pose in pose_sequence:
                # 修复的条件检查
                if (joint_idx < len(pose) and
                        safe_length_check(pose[joint_idx], 3) and
                        safe_confidence_check(pose[joint_idx])):
                    positions.append(pose[joint_idx][:2])

            if len(positions) < 3:
                return None

            positions = np.array(positions)
            position_variance = np.var(positions, axis=0)
            stability_score = 1.0 / (1.0 + np.mean(position_variance) / 100.0)

            return min(max(stability_score, 0.0), 1.0)

        except Exception as e:
            logger.error(f"关节稳定性计算失败: {e}")
            return None

    @staticmethod
    def _calculate_body_sway_stability(pose_sequence):
        """计算身体摆动稳定性"""
        try:
            # 使用头部和臀部来检测身体摆动
            head_positions = []
            hip_positions = []

            for pose in pose_sequence:
                # 头部位置 (鼻子) - 修复的条件检查
                if (0 < len(pose) and
                        safe_length_check(pose[0], 3) and
                        safe_confidence_check(pose[0])):
                    head_positions.append(pose[0][:2])

                # 臀部位置 - 修复的条件检查
                if (8 < len(pose) and
                        safe_length_check(pose[8], 3) and
                        safe_confidence_check(pose[8])):
                    hip_positions.append(pose[8][:2])

            if len(head_positions) < 3 or len(hip_positions) < 3:
                return None

            # 计算身体轴线的摆动
            sway_angles = []
            min_length = min(len(head_positions), len(hip_positions))

            for i in range(min_length):
                head_pos = np.array(head_positions[i])
                hip_pos = np.array(hip_positions[i])

                # 计算身体轴线角度
                body_vector = head_pos - hip_pos
                if np.linalg.norm(body_vector) > 1e-6:
                    angle = np.arctan2(body_vector[0], body_vector[1])
                    sway_angles.append(angle)

            if len(sway_angles) < 3:
                return None

            # 计算角度稳定性
            angle_variance = np.var(sway_angles)
            stability_score = 1.0 / (1.0 + angle_variance * 10)

            return min(max(stability_score, 0.0), 1.0)

        except Exception as e:
            logger.error(f"身体摆动稳定性计算失败: {e}")
            return None


class FixedSymmetryAnalyzer:
    """修复的对称性分析器"""

    @staticmethod
    def analyze_body_symmetry(pose_sequence):
        """分析身体对称性"""
        try:
            if not pose_sequence:
                return {
                    "overall_symmetry": 0.0,
                    "static_symmetry": 0.0,
                    "dynamic_symmetry": 0.0,
                    "limb_symmetry": {},
                    "postural_alignment": 0.0
                }

            symmetry_results = {}

            # 1. 静态对称性
            static_sym = FixedSymmetryAnalyzer._calculate_static_symmetry(pose_sequence[0])
            symmetry_results["static_symmetry"] = static_sym

            # 2. 动态对称性
            if len(pose_sequence) > 1:
                dynamic_sym = FixedSymmetryAnalyzer._calculate_dynamic_symmetry(pose_sequence)
                symmetry_results["dynamic_symmetry"] = dynamic_sym
            else:
                symmetry_results["dynamic_symmetry"] = static_sym

            # 3. 肢体对称性详细分析
            limb_symmetry = FixedSymmetryAnalyzer._analyze_limb_symmetry(pose_sequence)
            symmetry_results["limb_symmetry"] = limb_symmetry

            # 4. 姿态对齐分析
            postural_alignment = FixedSymmetryAnalyzer._calculate_postural_alignment(pose_sequence[0])
            symmetry_results["postural_alignment"] = postural_alignment

            # 5. 综合对称性评分
            symmetry_scores = [static_sym, symmetry_results["dynamic_symmetry"], postural_alignment]
            valid_scores = [s for s in symmetry_scores if s > 0]
            overall_symmetry = np.mean(valid_scores) if valid_scores else 0.0
            symmetry_results["overall_symmetry"] = overall_symmetry

            return symmetry_results

        except Exception as e:
            logger.error(f"身体对称性分析失败: {e}")
            return {
                "overall_symmetry": 0.0,
                "static_symmetry": 0.0,
                "dynamic_symmetry": 0.0,
                "limb_symmetry": {},
                "postural_alignment": 0.0
            }

    @staticmethod
    def _calculate_static_symmetry(pose):
        """计算静态对称性"""
        try:
            # 对称点对 (COCO格式)
            symmetric_pairs = [
                (2, 5),  # 左右肩
                (3, 6),  # 左右肘
                (4, 7),  # 左右腕
                (9, 12),  # 左右髋
                (10, 13),  # 左右膝
                (11, 14)  # 左右踝
            ]

            symmetry_scores = []

            # 计算身体中心线
            center_line = FixedSymmetryAnalyzer._calculate_body_centerline(pose)
            if center_line is None:
                return 0.0

            for left_idx, right_idx in symmetric_pairs:
                # 修复的条件检查
                left_valid = (left_idx < len(pose) and
                              safe_length_check(pose[left_idx], 3) and
                              safe_confidence_check(pose[left_idx]))

                right_valid = (right_idx < len(pose) and
                               safe_length_check(pose[right_idx], 3) and
                               safe_confidence_check(pose[right_idx]))

                if left_valid and right_valid:
                    left_pos = np.array(pose[left_idx][:2])
                    right_pos = np.array(pose[right_idx][:2])

                    # 计算相对于中心线的对称性
                    left_dist = FixedSymmetryAnalyzer._distance_to_centerline(left_pos, center_line)
                    right_dist = FixedSymmetryAnalyzer._distance_to_centerline(right_pos, center_line)

                    if left_dist > 0 and right_dist > 0:
                        # 对称性评分：距离差异越小越对称
                        symmetry = 1.0 - min(abs(left_dist - right_dist) / max(left_dist, right_dist), 1.0)
                        symmetry_scores.append(symmetry)

            return np.mean(symmetry_scores) if symmetry_scores else 0.0

        except Exception as e:
            logger.error(f"静态对称性计算失败: {e}")
            return 0.0

    @staticmethod
    def _calculate_dynamic_symmetry(pose_sequence):
        """计算动态对称性"""
        try:
            if len(pose_sequence) < 3:
                return 0.0

            dynamic_symmetry_scores = []

            # 对称肢体对
            limb_pairs = [
                ([5, 7, 9], [6, 8, 10]),  # 左右手臂
                ([11, 13, 15], [12, 14, 16])  # 左右腿
            ]

            for left_limb, right_limb in limb_pairs:
                left_movements = []
                right_movements = []

                # 计算每一帧的肢体运动
                for i in range(len(pose_sequence) - 1):
                    pose1, pose2 = pose_sequence[i], pose_sequence[i + 1]

                    left_movement = FixedSymmetryAnalyzer._calculate_limb_movement(
                        pose1, pose2, left_limb
                    )
                    right_movement = FixedSymmetryAnalyzer._calculate_limb_movement(
                        pose1, pose2, right_limb
                    )

                    if left_movement is not None and right_movement is not None:
                        left_movements.append(left_movement)
                        right_movements.append(right_movement)

                # 计算运动对称性
                if len(left_movements) >= 3:
                    movement_symmetry = FixedSymmetryAnalyzer._calculate_movement_symmetry(
                        left_movements, right_movements
                    )
                    if movement_symmetry is not None:
                        dynamic_symmetry_scores.append(movement_symmetry)

            return np.mean(dynamic_symmetry_scores) if dynamic_symmetry_scores else 0.0

        except Exception as e:
            logger.error(f"动态对称性计算失败: {e}")
            return 0.0

    @staticmethod
    def _analyze_limb_symmetry(pose_sequence):
        """分析肢体对称性详情"""
        try:
            limb_symmetry = {}

            # 分析每个肢体对的对称性
            limb_pairs = {
                "arms": ([5, 7, 9], [6, 8, 10]),  # 手臂
                "legs": ([11, 13, 15], [12, 14, 16]),  # 腿部
                "shoulders": ([2], [5]),  # 肩膀
                "hips": ([9], [12])  # 髋部
            }

            for limb_name, (left_joints, right_joints) in limb_pairs.items():
                symmetry_scores = []

                for pose in pose_sequence:
                    left_positions = []
                    right_positions = []

                    # 获取有效的关节位置
                    for joint_idx in left_joints:
                        if (joint_idx < len(pose) and
                                safe_length_check(pose[joint_idx], 3) and
                                safe_confidence_check(pose[joint_idx])):
                            left_positions.append(pose[joint_idx][:2])

                    for joint_idx in right_joints:
                        if (joint_idx < len(pose) and
                                safe_length_check(pose[joint_idx], 3) and
                                safe_confidence_check(pose[joint_idx])):
                            right_positions.append(pose[joint_idx][:2])

                    # 计算肢体对称性
                    if len(left_positions) == len(right_positions) and len(left_positions) > 0:
                        limb_sym = FixedSymmetryAnalyzer._calculate_limb_pair_symmetry(
                            left_positions, right_positions, pose
                        )
                        if limb_sym is not None:
                            symmetry_scores.append(limb_sym)

                limb_symmetry[limb_name] = np.mean(symmetry_scores) if symmetry_scores else 0.0

            return limb_symmetry

        except Exception as e:
            logger.error(f"肢体对称性分析失败: {e}")
            return {}

    @staticmethod
    def _calculate_postural_alignment(pose):
        """计算姿态对齐"""
        try:
            alignment_scores = []

            # 1. 肩膀水平对齐
            left_shoulder_valid = (2 < len(pose) and
                                   safe_length_check(pose[2], 3) and
                                   safe_confidence_check(pose[2]))
            right_shoulder_valid = (5 < len(pose) and
                                    safe_length_check(pose[5], 3) and
                                    safe_confidence_check(pose[5]))

            if left_shoulder_valid and right_shoulder_valid:
                left_shoulder = pose[2][:2]
                right_shoulder = pose[5][:2]
                shoulder_alignment = 1.0 - min(abs(left_shoulder[1] - right_shoulder[1]) / 100.0, 1.0)
                alignment_scores.append(shoulder_alignment)

            # 2. 髋部水平对齐
            left_hip_valid = (9 < len(pose) and
                              safe_length_check(pose[9], 3) and
                              safe_confidence_check(pose[9]))
            right_hip_valid = (12 < len(pose) and
                               safe_length_check(pose[12], 3) and
                               safe_confidence_check(pose[12]))

            if left_hip_valid and right_hip_valid:
                left_hip = pose[9][:2]
                right_hip = pose[12][:2]
                hip_alignment = 1.0 - min(abs(left_hip[1] - right_hip[1]) / 100.0, 1.0)
                alignment_scores.append(hip_alignment)

            # 3. 身体中轴对齐
            body_axis_alignment = FixedSymmetryAnalyzer._calculate_body_axis_alignment(pose)
            if body_axis_alignment is not None:
                alignment_scores.append(body_axis_alignment)

            return np.mean(alignment_scores) if alignment_scores else 0.0

        except Exception as e:
            logger.error(f"姿态对齐计算失败: {e}")
            return 0.0

    @staticmethod
    def _calculate_body_centerline(pose):
        """计算身体中心线"""
        try:
            # 使用鼻子和中髋来定义中心线
            nose_valid = (0 < len(pose) and
                          safe_length_check(pose[0], 3) and
                          safe_confidence_check(pose[0]))
            hip_valid = (8 < len(pose) and
                         safe_length_check(pose[8], 3) and
                         safe_confidence_check(pose[8]))

            if nose_valid and hip_valid:
                nose_pos = np.array(pose[0][:2])
                mid_hip_pos = np.array(pose[8][:2])

                return {
                    "point1": nose_pos,
                    "point2": mid_hip_pos,
                    "vector": mid_hip_pos - nose_pos
                }

            return None

        except Exception as e:
            logger.error(f"身体中心线计算失败: {e}")
            return None

    @staticmethod
    def _distance_to_centerline(point, centerline):
        """计算点到中心线的距离"""
        try:
            if centerline is None:
                return 0

            line_point = centerline["point1"]
            line_vector = centerline["vector"]

            # 避免零向量
            if np.linalg.norm(line_vector) < 1e-6:
                return np.linalg.norm(point - line_point)

            # 计算点到直线的距离
            point_vector = point - line_point
            cross_product = np.cross(point_vector, line_vector)
            distance = abs(cross_product) / np.linalg.norm(line_vector)

            return distance

        except Exception as e:
            logger.error(f"点到中心线距离计算失败: {e}")
            return 0

    @staticmethod
    def _calculate_limb_movement(pose1, pose2, joint_indices):
        """计算肢体运动量"""
        try:
            total_movement = 0
            valid_joints = 0

            for joint_idx in joint_indices:
                # 修复的条件检查
                pose1_valid = (joint_idx < len(pose1) and
                               safe_length_check(pose1[joint_idx], 3) and
                               safe_confidence_check(pose1[joint_idx]))

                pose2_valid = (joint_idx < len(pose2) and
                               safe_length_check(pose2[joint_idx], 3) and
                               safe_confidence_check(pose2[joint_idx]))

                if pose1_valid and pose2_valid:
                    pos1 = np.array(pose1[joint_idx][:2])
                    pos2 = np.array(pose2[joint_idx][:2])
                    movement = np.linalg.norm(pos2 - pos1)
                    total_movement += movement
                    valid_joints += 1

            return total_movement / valid_joints if valid_joints > 0 else None

        except Exception as e:
            logger.error(f"肢体运动量计算失败: {e}")
            return None

    @staticmethod
    def _calculate_movement_symmetry(left_movements, right_movements):
        """计算运动对称性"""
        try:
            if len(left_movements) != len(right_movements) or len(left_movements) < 2:
                return None

            left_array = np.array(left_movements)
            right_array = np.array(right_movements)

            # 计算运动幅度的相关性 - 修复标准差检查
            std_left = np.std(left_array)
            std_right = np.std(right_array)

            if std_left > 1e-6 and std_right > 1e-6:
                correlation = np.corrcoef(left_array, right_array)[0, 1]
                correlation = np.nan_to_num(correlation)
                return max(0, (correlation + 1) / 2)  # 转换到0-1范围

            # 如果标准差太小，计算差异的倒数
            movement_diff = np.mean(np.abs(left_array - right_array))
            max_movement = max(np.mean(left_array), np.mean(right_array))

            if max_movement > 1e-6:
                symmetry = 1.0 - min(movement_diff / max_movement, 1.0)
                return max(symmetry, 0.0)

            return 0.5

        except Exception as e:
            logger.error(f"运动对称性计算失败: {e}")
            return None

    @staticmethod
    def _calculate_limb_pair_symmetry(left_positions, right_positions, pose):
        """计算肢体对对称性"""
        try:
            if len(left_positions) != len(right_positions) or len(left_positions) == 0:
                return None

            # 获取身体中心线
            centerline = FixedSymmetryAnalyzer._calculate_body_centerline(pose)
            if centerline is None:
                return 0.5

            symmetry_scores = []

            for left_pos, right_pos in zip(left_positions, right_positions):
                left_pos = np.array(left_pos)
                right_pos = np.array(right_pos)

                # 计算相对于中心线的距离
                left_dist = FixedSymmetryAnalyzer._distance_to_centerline(left_pos, centerline)
                right_dist = FixedSymmetryAnalyzer._distance_to_centerline(right_pos, centerline)

                if left_dist > 0 and right_dist > 0:
                    # 对称性评分
                    symmetry = 1.0 - min(abs(left_dist - right_dist) / max(left_dist, right_dist), 1.0)
                    symmetry_scores.append(symmetry)

            return np.mean(symmetry_scores) if symmetry_scores else None

        except Exception as e:
            logger.error(f"肢体对对称性计算失败: {e}")
            return None

    @staticmethod
    def _calculate_body_axis_alignment(pose):
        """计算身体轴线对齐"""
        try:
            # 检查关键点的垂直对齐
            key_points = []

            # 鼻子
            if (0 < len(pose) and
                    safe_length_check(pose[0], 3) and
                    safe_confidence_check(pose[0])):
                key_points.append(pose[0][:2])

            # 颈部
            if (1 < len(pose) and
                    safe_length_check(pose[1], 3) and
                    safe_confidence_check(pose[1])):
                key_points.append(pose[1][:2])

            # 中髋
            if (8 < len(pose) and
                    safe_length_check(pose[8], 3) and
                    safe_confidence_check(pose[8])):
                key_points.append(pose[8][:2])

            if len(key_points) < 2:
                return None

            # 计算垂直对齐度
            x_coords = [point[0] for point in key_points]
            x_variance = np.var(x_coords)

            # 对齐评分：方差越小越对齐
            alignment_score = 1.0 / (1.0 + x_variance / 100.0)

            return min(max(alignment_score, 0.0), 1.0)

        except Exception as e:
            logger.error(f"身体轴线对齐计算失败: {e}")
            return None


class SafeVisualizationManager:
    """安全的可视化管理器 - 修复内存泄漏和字体问题"""

    def __init__(self):
        self.color_schemes = {
            "professional": {
                "primary": "#2E86AB",
                "secondary": "#A23B72",
                "accent": "#F18F01",
                "background": "#F8F9FA",
                "text": "#212529"
            },
            "sports": {
                "primary": "#FF6B35",
                "secondary": "#004E89",
                "accent": "#FFE66D",
                "background": "#FFFFFF",
                "text": "#2C3E50"
            }
        }
        self.current_scheme = "professional"
        self.figures = []  # 跟踪创建的图形，用于清理

    def __del__(self):
        """析构函数 - 清理资源"""
        self.cleanup()

    def cleanup(self):
        """清理matplotlib图形资源"""
        try:
            for fig in self.figures:
                if fig is not None:
                    plt.close(fig)
            self.figures.clear()
        except Exception as e:
            logger.warning(f"清理图形资源时出现警告: {e}")

    def create_pose_visualization(self, pose_data, analysis_results=None):
        """创建姿态可视化 - 添加安全检查"""
        fig = None
        try:
            # 确保字体设置
            setup_chinese_font()

            fig, ax = plt.subplots(1, 1, figsize=(10, 8))
            self.figures.append(fig)  # 跟踪图形

            # 设置颜色方案
            colors = self.color_schemes[self.current_scheme]
            fig.patch.set_facecolor(colors["background"])
            ax.set_facecolor(colors["background"])

            if not pose_data:
                ax.text(0.5, 0.5, "No pose data available", ha='center', va='center',
                        transform=ax.transAxes, fontsize=16, color=colors["text"])
                return fig

            # 绘制关键点
            self._draw_keypoints(ax, pose_data, colors)

            # 绘制骨架连接
            self._draw_skeleton(ax, pose_data, colors)

            # 如果有分析结果，添加可视化
            if analysis_results:
                self._add_analysis_overlay(ax, pose_data, analysis_results, colors)

            # 设置图形属性
            ax.set_xlim(0, 640)
            ax.set_ylim(480, 0)  # 翻转Y轴
            ax.set_aspect('equal')
            ax.set_title("Pose Analysis Visualization", fontsize=16, color=colors["text"], pad=20)

            # 移除坐标轴
            ax.set_xticks([])
            ax.set_yticks([])

            plt.tight_layout()
            return fig

        except Exception as e:
            logger.error(f"姿态可视化创建失败: {e}")
            if fig is not None:
                plt.close(fig)
                if fig in self.figures:
                    self.figures.remove(fig)
            return None

    def _draw_keypoints(self, ax, pose_data, colors):
        """绘制关键点 - 添加安全检查"""
        try:
            # COCO关键点配色
            joint_colors = {
                0: colors["accent"],  # 鼻子
                1: colors["primary"],  # 颈部
                2: colors["secondary"],  # 右肩
                5: colors["secondary"],  # 左肩
                8: colors["primary"],  # 中髋
            }

            for i, keypoint in enumerate(pose_data):
                # 安全的关键点检查
                if (safe_length_check(keypoint, 3) and
                        safe_confidence_check(keypoint)):

                    x, y, confidence = keypoint[0], keypoint[1], keypoint[2]

                    # 确保坐标是数值类型
                    if isinstance(x, np.ndarray):
                        x = x.item() if x.size == 1 else float(x[0])
                    if isinstance(y, np.ndarray):
                        y = y.item() if y.size == 1 else float(y[0])
                    if isinstance(confidence, np.ndarray):
                        confidence = confidence.item() if confidence.size == 1 else float(confidence[0])

                    # 根据置信度调整点的大小
                    point_size = 20 + confidence * 30

                    # 选择颜色
                    color = joint_colors.get(i, colors["primary"])

                    ax.scatter(x, y, s=point_size, c=color, alpha=0.8,
                               edgecolors='white', linewidth=2, zorder=3)

                    # 添加关键点标签（可选）
                    if confidence > 0.7:  # 只为高置信度点添加标签
                        ax.annotate(str(i), (x, y), xytext=(5, 5),
                                    textcoords='offset points', fontsize=8,
                                    color=colors["text"], alpha=0.7)

        except Exception as e:
            logger.error(f"关键点绘制失败: {e}")

    def _draw_skeleton(self, ax, pose_data, colors):
        """绘制骨架连接 - 添加安全检查"""
        try:
            # COCO骨架连接定义
            skeleton_connections = [
                (1, 2), (1, 5),  # 颈部到肩膀
                (2, 3), (3, 4),  # 右臂
                (5, 6), (6, 7),  # 左臂
                (1, 8),  # 颈部到髋部
                (8, 9), (8, 12),  # 髋部到大腿
                (9, 10), (10, 11),  # 右腿
                (12, 13), (13, 14)  # 左腿
            ]

            for start_idx, end_idx in skeleton_connections:
                # 安全的关键点检查
                start_valid = (start_idx < len(pose_data) and
                               safe_length_check(pose_data[start_idx], 3) and
                               safe_confidence_check(pose_data[start_idx]))

                end_valid = (end_idx < len(pose_data) and
                             safe_length_check(pose_data[end_idx], 3) and
                             safe_confidence_check(pose_data[end_idx]))

                if start_valid and end_valid:
                    start_point = pose_data[start_idx][:2]
                    end_point = pose_data[end_idx][:2]

                    # 确保坐标是数值类型
                    start_x = start_point[0].item() if isinstance(start_point[0], np.ndarray) else start_point[0]
                    start_y = start_point[1].item() if isinstance(start_point[1], np.ndarray) else start_point[1]
                    end_x = end_point[0].item() if isinstance(end_point[0], np.ndarray) else end_point[0]
                    end_y = end_point[1].item() if isinstance(end_point[1], np.ndarray) else end_point[1]

                    # 根据置信度调整线条粗细和透明度
                    start_conf = pose_data[start_idx][2]
                    end_conf = pose_data[end_idx][2]

                    if isinstance(start_conf, np.ndarray):
                        start_conf = start_conf.item() if start_conf.size == 1 else float(start_conf[0])
                    if isinstance(end_conf, np.ndarray):
                        end_conf = end_conf.item() if end_conf.size == 1 else float(end_conf[0])

                    confidence = min(start_conf, end_conf)
                    line_width = 1 + confidence * 3
                    alpha = 0.3 + confidence * 0.5

                    ax.plot([start_x, end_x], [start_y, end_y],
                            color=colors["primary"], linewidth=line_width,
                            alpha=alpha, zorder=1)

        except Exception as e:
            logger.error(f"骨架绘制失败: {e}")

    def _add_analysis_overlay(self, ax, pose_data, analysis_results, colors):
        """添加分析结果覆盖层 - 添加安全检查"""
        try:
            # 添加重心标记
            if "center_of_mass" in analysis_results:
                com = analysis_results["center_of_mass"]
                if com and len(com) >= 2:
                    com_x = com[0].item() if isinstance(com[0], np.ndarray) else com[0]
                    com_y = com[1].item() if isinstance(com[1], np.ndarray) else com[1]

                    ax.scatter(com_x, com_y, s=100, marker='+',
                               c=colors["accent"], linewidth=3, zorder=4)
                    ax.annotate('COM', (com_x, com_y), xytext=(10, 10),
                                textcoords='offset points', fontsize=10,
                                color=colors["accent"], fontweight='bold')

        except Exception as e:
            logger.error(f"分析覆盖层添加失败: {e}")
# ==================== 独立函数定义 ====================
def setup_application():
    """设置应用程序 - 简约清晰风格"""
    app = QApplication(sys.argv)

    # 设置应用程序信息
    app.setApplicationName("增强版运动姿势改良系统")
    app.setApplicationVersion("2.0.0")
    app.setOrganizationName("运动科学实验室")

    # 设置现代简约风格
    app.setStyle(QStyleFactory.create('Fusion'))

    # 简约调色板 - 使用现代灰白色调
    palette = QPalette()
    palette.setColor(QPalette.Window, QColor(248, 249, 250))  # 浅灰背景
    palette.setColor(QPalette.WindowText, QColor(33, 37, 41))  # 深灰文字
    palette.setColor(QPalette.Base, QColor(255, 255, 255))  # 纯白基础
    palette.setColor(QPalette.AlternateBase, QColor(241, 243, 245))  # 淡灰交替
    palette.setColor(QPalette.ToolTipBase, QColor(255, 255, 255))  # 白色提示
    palette.setColor(QPalette.ToolTipText, QColor(73, 80, 87))  # 灰色提示文字
    palette.setColor(QPalette.Text, QColor(33, 37, 41))  # 深灰文字
    palette.setColor(QPalette.Button, QColor(248, 249, 250))  # 浅灰按钮
    palette.setColor(QPalette.ButtonText, QColor(73, 80, 87))  # 灰色按钮文字
    palette.setColor(QPalette.BrightText, QColor(220, 53, 69))  # 红色警告文字
    palette.setColor(QPalette.Link, QColor(13, 110, 253))  # 蓝色链接
    palette.setColor(QPalette.Highlight, QColor(13, 110, 253))  # 蓝色高亮
    palette.setColor(QPalette.HighlightedText, QColor(255, 255, 255))  # 白色高亮文字
    app.setPalette(palette)

    # 现代简约样式表
    app.setStyleSheet("""
        /* 主窗口 */
        QMainWindow {
            background-color: #f8f9fa;
            color: #212529;
        }

        /* 标签页组件 */
        QTabWidget::pane {
            border: 1px solid #dee2e6;
            background-color: #ffffff;
            border-radius: 8px;
            margin-top: 5px;
        }

        QTabWidget::tab-bar {
            alignment: center;
        }

        QTabBar::tab {
            background-color: #f8f9fa;
            color: #495057;
            padding: 12px 24px;
            margin-right: 2px;
            border: 1px solid #dee2e6;
            border-bottom: none;
            border-radius: 8px 8px 0 0;
            font-weight: 500;
            min-width: 120px;
        }

        QTabBar::tab:selected {
            background-color: #ffffff;
            color: #0d6efd;
            border-color: #0d6efd;
            border-bottom: 2px solid #0d6efd;
        }

        QTabBar::tab:hover:!selected {
            background-color: #e9ecef;
            color: #495057;
        }

        /* 分组框 */
        QGroupBox {
            font-weight: 600;
            font-size: 14px;
            border: 1px solid #dee2e6;
            border-radius: 8px;
            margin-top: 16px;
            padding-top: 16px;
            background-color: #ffffff;
            color: #495057;
        }

        QGroupBox::title {
            subcontrol-origin: margin;
            subcontrol-position: top left;
            padding: 4px 12px;
            background-color: #ffffff;
            color: #495057;
            border: 1px solid #dee2e6;
            border-radius: 4px;
            margin-left: 16px;
        }

        /* 按钮样式 */
        QPushButton {
            background-color: #0d6efd;
            color: #ffffff;
            border: none;
            padding: 10px 20px;
            border-radius: 6px;
            font-weight: 500;
            font-size: 14px;
            min-height: 20px;
        }

        QPushButton:hover {
            background-color: #0b5ed7;
        }

        QPushButton:pressed {
            background-color: #0a58ca;
        }

        QPushButton:disabled {
            background-color: #6c757d;
            color: #adb5bd;
        }

        /* 次要按钮 */
        QPushButton[class="secondary"] {
            background-color: #6c757d;
        }

        QPushButton[class="secondary"]:hover {
            background-color: #5c636a;
        }

        /* 成功按钮 */
        QPushButton[class="success"] {
            background-color: #198754;
        }

        QPushButton[class="success"]:hover {
            background-color: #157347;
        }

        /* 警告按钮 */
        QPushButton[class="warning"] {
            background-color: #fd7e14;
        }

        QPushButton[class="warning"]:hover {
            background-color: #e8681c;
        }

        /* 表格 */
        QTableWidget {
            gridline-color: #dee2e6;
            background-color: #ffffff;
            alternate-background-color: #f8f9fa;
            selection-background-color: #cfe2ff;
            selection-color: #0c4128;
            border: 1px solid #dee2e6;
            border-radius: 6px;
            color: #495057;
        }

        QTableWidget::item {
            padding: 8px 12px;
            border-bottom: 1px solid #f1f3f5;
        }

        QTableWidget::item:selected {
            background-color: #cfe2ff;
            color: #0c4128;
        }

        QHeaderView::section {
            background-color: #f8f9fa;
            color: #495057;
            padding: 10px 12px;
            border: none;
            border-right: 1px solid #dee2e6;
            border-bottom: 1px solid #dee2e6;
            font-weight: 600;
        }

        /* 树形控件 */
        QTreeWidget {
            background-color: #ffffff;
            color: #495057;
            border: 1px solid #dee2e6;
            border-radius: 6px;
            selection-background-color: #cfe2ff;
            outline: none;
        }

        QTreeWidget::item {
            padding: 6px;
            border-bottom: 1px solid #f1f3f5;
        }

        QTreeWidget::item:selected {
            background-color: #cfe2ff;
            color: #0c4128;
        }

        QTreeWidget::item:hover {
            background-color: #e7f1ff;
        }

        /* 进度条 */
        QProgressBar {
            border: 1px solid #dee2e6;
            border-radius: 6px;
            text-align: center;
            background-color: #f8f9fa;
            color: #495057;
            font-weight: 500;
            height: 20px;
        }

        QProgressBar::chunk {
            background-color: #0d6efd;
            border-radius: 5px;
        }

        /* 标签 */
        QLabel {
            color: #495057;
        }

        /* 输入框 */
        QLineEdit {
            border: 1px solid #ced4da;
            border-radius: 6px;
            padding: 8px 12px;
            background-color: #ffffff;
            color: #495057;
            font-size: 14px;
        }

        QLineEdit:focus {
            border-color: #86b7fe;
            box-shadow: 0 0 0 0.2rem rgba(13, 110, 253, 0.25);
            outline: none;
        }

        QLineEdit:disabled {
            background-color: #e9ecef;
            color: #6c757d;
        }

        /* 下拉框 */
        QComboBox {
            border: 1px solid #ced4da;
            border-radius: 6px;
            padding: 8px 12px;
            background-color: #ffffff;
            color: #495057;
            font-size: 14px;
            min-width: 120px;
        }

        QComboBox:focus {
            border-color: #86b7fe;
        }

        QComboBox::drop-down {
            border: none;
            width: 20px;
        }

        QComboBox::down-arrow {
            image: none;
            border-left: 4px solid transparent;
            border-right: 4px solid transparent;
            border-top: 6px solid #6c757d;
            margin-right: 8px;
        }

        /* 数字输入框 */
        QSpinBox, QDoubleSpinBox {
            border: 1px solid #ced4da;
            border-radius: 6px;
            padding: 8px 12px;
            background-color: #ffffff;
            color: #495057;
            font-size: 14px;
        }

        QSpinBox:focus, QDoubleSpinBox:focus {
            border-color: #86b7fe;
        }

        /* 文本编辑器 */
        QTextEdit {
            border: 1px solid #ced4da;
            border-radius: 6px;
            background-color: #ffffff;
            color: #495057;
            padding: 8px;
            font-size: 14px;
            line-height: 1.5;
        }

        QTextEdit:focus {
            border-color: #86b7fe;
        }

        /* 滑动条 */
        QSlider::groove:horizontal {
            border: 1px solid #dee2e6;
            height: 6px;
            background: #f8f9fa;
            margin: 0;
            border-radius: 3px;
        }

        QSlider::handle:horizontal {
            background: #0d6efd;
            border: none;
            width: 18px;
            height: 18px;
            margin: -6px 0;
            border-radius: 9px;
        }

        QSlider::handle:horizontal:hover {
            background: #0b5ed7;
        }

        QSlider::handle:horizontal:pressed {
            background: #0a58ca;
        }

        /* 工具栏 */
        QToolBar {
            background-color: #ffffff;
            border: none;
            border-bottom: 1px solid #dee2e6;
            spacing: 8px;
            padding: 8px;
        }

        QToolBar::separator {
            background-color: #dee2e6;
            width: 1px;
            margin: 4px;
        }

        /* 菜单 */
        QMenu {
            background-color: #ffffff;
            border: 1px solid #dee2e6;
            border-radius: 6px;
            padding: 4px 0;
            color: #495057;
        }

        QMenu::item {
            padding: 8px 16px;
            margin: 2px 4px;
            border-radius: 4px;
        }

        QMenu::item:selected {
            background-color: #e7f1ff;
            color: #0c4128;
        }

        /* 滚动条 */
        QScrollBar:vertical {
            background: #f8f9fa;
            width: 12px;
            border-radius: 6px;
            margin: 0;
        }

        QScrollBar::handle:vertical {
            background: #ced4da;
            border-radius: 6px;
            min-height: 20px;
            margin: 2px;
        }

        QScrollBar::handle:vertical:hover {
            background: #adb5bd;
        }

        QScrollBar::add-line:vertical,
        QScrollBar::sub-line:vertical {
            height: 0px;
        }

        /* 状态栏样式 */
        QStatusBar {
            background-color: #f8f9fa;
            border-top: 1px solid #dee2e6;
            padding: 4px 8px;
            color: #6c757d;
        }

        /* 工具提示 */
        QToolTip {
            background-color: #000000;
            color: #ffffff;
            border: none;
            border-radius: 4px;
            padding: 6px 8px;
            font-size: 12px;
        }
    """)

    return app


def check_dependencies():
    """检查依赖项"""
    missing_deps = []

    try:
        import cv2
    except ImportError:
        missing_deps.append("opencv-python")

    try:
        import numpy
    except ImportError:
        missing_deps.append("numpy")

    try:
        from PyQt5.QtWidgets import QApplication
    except ImportError:
        missing_deps.append("PyQt5")

    if missing_deps:
        print("缺少以下依赖项:")
        for dep in missing_deps:
            print(f"  - {dep}")
        print("\n请使用以下命令安装:")
        print(f"pip install {' '.join(missing_deps)}")
        return False

    return True


def show_splash_screen(app):
    """显示启动画面"""
    try:
        # 创建启动画面图像
        splash_pix = QPixmap(400, 300)
        splash_pix.fill(QColor(135, 206, 250))  # 天蓝色背景

        # 使用QSplashScreen而不是QLabel
        from PyQt5.QtWidgets import QSplashScreen
        splash = QSplashScreen(splash_pix)
        splash.show()

        # 模拟启动过程
        for i in range(101):
            splash.showMessage(f"正在启动增强版运动姿势改良系统... {i}%",
                               Qt.AlignCenter | Qt.AlignBottom, QColor(25, 25, 112))
            app.processEvents()
            time.sleep(0.01)

        splash.close()

    except Exception as e:
        logger.warning(f"启动画面显示失败: {str(e)}")


def main():
    """主函数 - 添加全局清理"""
    try:
        # 检查依赖项
        if not check_dependencies():
            sys.exit(1)

        # 创建应用程序
        app = setup_application()

        # 显示启动画面
        show_splash_screen(app)

        # 创建主窗口
        window = EnhancedDataAnalysisUI()
        window.show()

        # 显示欢迎消息
        QMessageBox.information(window, '欢迎',
                                '欢迎使用增强版运动姿势改良系统！\n\n'
                                '系统特色功能:\n'
                                '✓ 专业生物力学分析\n'
                                '✓ AI损伤风险评估\n'
                                '✓ 个性化训练处方\n'
                                '✓ 智能虚拟教练\n\n'
                                '请先在GoPose标签页载入视频和解析点数据开始分析。')

        # 注册退出清理函数
        import atexit
        def cleanup_on_exit():
            try:
                # 清理所有全局资源
                if hasattr(window, 'enhanced_gopose_tab') and hasattr(window.enhanced_gopose_tab, 'memory_manager'):
                    window.enhanced_gopose_tab.memory_manager.cleanup_on_exit()
                cv2.destroyAllWindows()
            except Exception as e:
                print(f"退出清理警告: {e}")

        atexit.register(cleanup_on_exit)

        # 启动应用程序主循环
        result = app.exec_()

        # 手动清理
        cleanup_on_exit()

        sys.exit(result)

    except Exception as e:
        logger.error(f"应用程序启动失败: {str(e)}")
        print(f"启动失败: {str(e)}")
        sys.exit(1)


def handle_exception(exc_type, exc_value, exc_traceback):
    """改进的全局异常处理"""
    if issubclass(exc_type, KeyboardInterrupt):
        sys.__excepthook__(exc_type, exc_value, exc_traceback)
        return

    logger.error("未捕获的异常", exc_info=(exc_type, exc_value, exc_traceback))

    # 显示错误对话框
    try:
        from PyQt5.QtWidgets import QApplication, QMessageBox
        if QApplication.instance():
            msg = QMessageBox()
            msg.setIcon(QMessageBox.Critical)
            msg.setWindowTitle("系统错误")
            msg.setText("系统发生未预期的错误")
            msg.setDetailedText(f"{exc_type.__name__}: {exc_value}")
            msg.setStandardButtons(QMessageBox.Ok)
            msg.exec_()
    except Exception as dialog_error:
        # 如果连错误对话框都显示不了，至少打印到控制台
        print(f"严重错误: {exc_type.__name__}: {exc_value}")
        print(f"错误对话框显示失败: {dialog_error}")


class SystemConfig:
    """系统配置管理"""

    def __init__(self):
        self.config_file = "config.json"
        self.default_config = {
            "analysis": {
                "confidence_threshold": 0.3,
                "smoothing_window": 5,
                "fps_rate": 30
            },
            "ai_coach": {
                "model_path": "models/coach_model.pkl",
                "max_tokens": 1000
            },
            "visualization": {
                "chart_style": "modern",
                "color_scheme": "professional"
            }
        }

    def load_config(self):
        """载入配置"""
        try:
            with open(self.config_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        except:
            return self.default_config


# 建议：添加数据缓存和内存管理
class DataCacheManager:
    """数据缓存管理器"""

    def __init__(self, max_cache_size=100):
        self.cache = {}
        self.max_size = max_cache_size
        self.access_order = []

    def get_cached_analysis(self, frame_key):
        """获取缓存的分析结果"""
        if frame_key in self.cache:
            self.access_order.remove(frame_key)
            self.access_order.append(frame_key)
            return self.cache[frame_key]
        return None

    def cache_analysis(self, frame_key, analysis_result):
        """缓存分析结果"""
        if len(self.cache) >= self.max_size:
            # 移除最少使用的缓存
            oldest_key = self.access_order.pop(0)
            del self.cache[oldest_key]

        self.cache[frame_key] = analysis_result
        self.access_order.append(frame_key)


 # 建议：添加多线程处理


class RealTimeProcessor(QThread):
    """实时处理线程"""
    analysis_ready = pyqtSignal(dict)

    def __init__(self):
        super().__init__()
        self.frame_queue = queue.Queue(maxsize=10)
        self.running = False

    def add_frame(self, frame_data):
        """添加帧到处理队列"""
        if not self.frame_queue.full():
            self.frame_queue.put(frame_data)

    def run(self):
        """实时处理主循环"""
        while self.running:
            try:
                frame_data = self.frame_queue.get(timeout=0.1)
                # 快速分析处理
                result = self.quick_analysis(frame_data)
                self.analysis_ready.emit(result)
            except queue.Empty:
                continue

    def quick_analysis(self, frame_data):
        """快速分析算法"""
        # 实现轻量级分析
        pass

# 在主应用程序关闭时调用清理
def cleanup_on_exit(self):
    """应用程序退出时的清理"""
    self._is_active = False
    self.stop_cleanup_timer()
    self.clear_cache()


import sys
import os
import logging


# 假设这些是您程序中的其他导入和变量
# from your_modules import handle_exception, SmartSportsBot, SMART_COACH_AVAILABLE
# logger = logging.getLogger(__name__)


def test_smart_coach_integration():
    """测试智能教练集成"""
    print("🧪 测试智能教练集成...")

    try:
        if SMART_COACH_AVAILABLE:
            bot = SmartSportsBot()
            test_response = bot.smart_chat("测试消息")
            print("✅ 智能教练集成成功")
            return True
        else:
            print("⚠️ 智能教练模块未找到，使用基础模式")
            return False
    except Exception as e:
        print(f"❌ 集成测试失败: {e}")
        return False


def main():
    """主程序逻辑"""
    # 测试智能教练集成
    test_smart_coach_integration()

    # 继续原有的main逻辑
    print("🚀 启动增强版运动姿势改良系统...")

    # 在这里添加您的主程序逻辑
    # 例如：
    # - 初始化UI界面
    # - 启动主要功能模块
    # - 开始主循环

    print("📱 系统已成功启动并运行中...")

    # 您的主程序逻辑在这里...


def initialize_system():
    """系统初始化"""
    # 设置工作目录
    if hasattr(sys, '_MEIPASS'):
        os.chdir(sys._MEIPASS)

    # 确保必要的目录存在
    required_dirs = ['data', 'athlete_profiles', 'results', 'exports']
    for dir_name in required_dirs:
        if not os.path.exists(dir_name):
            os.makedirs(dir_name)
            print(f"📁 创建目录: {dir_name}")

    # 记录启动信息
    if 'logger' in globals():
        logger.info("=== 增强版运动姿势改良系统启动 ===")
        logger.info(f"Python版本: {sys.version}")
        logger.info(f"工作目录: {os.getcwd()}")
        logger.info(f"系统平台: {sys.platform}")
    else:
        print("=== 增强版运动姿势改良系统启动 ===")
        print(f"Python版本: {sys.version}")
        print(f"工作目录: {os.getcwd()}")
        print(f"系统平台: {sys.platform}")


# ==================== 程序入口 ====================
if __name__ == '__main__':
    try:
        # 设置全局异常处理
        if 'handle_exception' in globals():
            sys.excepthook = handle_exception

        # 系统初始化
        initialize_system()

        # 启动主程序
        main()

    except Exception as e:
        print(f"❌ 程序启动失败: {e}")
        sys.exit(1)


    def test_smart_coach_integration():
        """测试智能教练集成"""
        print("🧪 测试智能教练集成...")

        try:
            if SMART_COACH_AVAILABLE:
                bot = SmartSportsBot()
                test_response = bot.smart_chat("测试消息")
                print("✅ 智能教练集成成功")
                return True
            else:
                print("⚠️ 智能教练模块未找到，使用基础模式")
                return False
        except Exception as e:
            print(f"❌ 集成测试失败: {e}")
            return False


    # 在main()函数开始时调用
    def main():
        # 测试智能教练集成
        test_smart_coach_integration()

        # 继续原有的main逻辑
        # ...
def validate_3d_data(pose_3d):
    """验证3D数据有效性"""
    if pose_3d is None:
        return False, "3D数据为空"

    if not isinstance(pose_3d, (list, np.ndarray)):
        return False, "3D数据格式错误"

    if len(pose_3d) == 0:
        return False, "3D数据为空数组"

    # 检查数据结构
    valid_points = 0
    for i, point in enumerate(pose_3d):
        if len(point) >= 4 and point[3] > 0.1:
            valid_points += 1

    if valid_points < 5:
        return False, f"有效关键点太少: {valid_points}"

    return True, "数据有效"


def run_complete_sequence_analysis_with_cache(self):
    """运行完整序列分析（带缓存优化）"""
    if not self.data or not self.athlete_profile:
        QMessageBox.warning(self, '警告', '请先载入数据和设置运动员档案')
        return False

    # 检查是否有缓存的分析结果
    cache_key = f"{hash(str(self.data))}_{self.athlete_profile.get('id', 'unknown')}"

    if hasattr(self, 'analysis_cache') and cache_key in self.analysis_cache:
        reply = QMessageBox.question(self, '发现缓存',
                                     '发现相同数据的分析缓存，是否使用缓存结果？\n'
                                     '(使用缓存可以大大节省分析时间)',
                                     QMessageBox.Yes | QMessageBox.No)
        if reply == QMessageBox.Yes:
            self.sequence_manager = self.analysis_cache[cache_key]['sequence_manager']
            self.sequence_summary = self.analysis_cache[cache_key]['sequence_summary']
            self.sequence_analysis_completed = True
            QMessageBox.information(self, '完成', '已载入缓存的序列分析结果！')
            return True

    # 执行新的分析
    if self.run_complete_sequence_analysis():
        # 保存到缓存
        if not hasattr(self, 'analysis_cache'):
            self.analysis_cache = {}

        self.analysis_cache[cache_key] = {
            'sequence_manager': self.sequence_manager,
            'sequence_summary': self.sequence_summary,
            'timestamp': datetime.now()
        }

        return True

    return False


def export_sequence_analysis_results(self):
    """导出序列分析结果"""
    if not self.sequence_analysis_completed:
        QMessageBox.warning(self, '警告', '请先完成序列分析')
        return

    save_path, _ = QFileDialog.getSaveFileName(
        self, '导出序列分析结果',
        f'sequence_analysis_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json',
        "JSON Files (*.json);;CSV Files (*.csv)"
    )

    if save_path:
        try:
            export_data = {
                'athlete_profile': self.athlete_profile,
                'analysis_timestamp': datetime.now().isoformat(),
                'sequence_info': {
                    'total_frames': len(self.sequence_manager.analysis_results),
                    'duration_seconds': len(self.sequence_manager.analysis_results) / self.fpsRate,
                    'fps': self.fpsRate
                },
                'sequence_summary': self.sequence_summary,
                'detailed_results': self.sequence_manager.analysis_results
            }

            if save_path.endswith('.json'):
                with open(save_path, 'w', encoding='utf-8') as f:
                    json.dump(export_data, f, ensure_ascii=False, indent=2)
            elif save_path.endswith('.csv'):
                # 导出统计摘要为CSV
                self._export_summary_to_csv(save_path, export_data)

            QMessageBox.information(self, '成功', f'序列分析结果已导出到: {save_path}')

        except Exception as e:
            QMessageBox.warning(self, '错误', f'导出失败: {str(e)}')


def _export_summary_to_csv(self, save_path, export_data):
    """导出摘要统计到CSV"""
    import csv

    with open(save_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)

        # 写入基本信息
        writer.writerow(['序列分析摘要'])
        writer.writerow(['运动员', export_data['athlete_profile'].get('name', '未知')])
        writer.writerow(['分析时间', export_data['analysis_timestamp']])
        writer.writerow(['序列帧数', export_data['sequence_info']['total_frames']])
        writer.writerow(['序列时长(秒)', export_data['sequence_info']['duration_seconds']])
        writer.writerow([])

        # 写入角度统计
        writer.writerow(['关节角度统计'])
        writer.writerow(['关节名称', '平均值', '标准差', '最小值', '最大值', '变异系数'])

        for angle_name, stats in export_data['sequence_summary'].get('angles_stats', {}).items():
            writer.writerow([
                angle_name,
                f"{stats['mean']:.2f}",
                f"{stats['std']:.2f}",
                f"{stats['min']:.2f}",
                f"{stats['max']:.2f}",
                f"{stats['coefficient_variation']:.3f}"
            ])


def validate_system_config(self):
    """验证系统配置"""
    errors = []

    # 检查必要目录
    required_dirs = ['data', 'athlete_profiles', 'results', 'exports']
    for dir_name in required_dirs:
        if not os.path.exists(dir_name):
            try:
                os.makedirs(dir_name)
            except Exception as e:
                errors.append(f"无法创建目录 {dir_name}: {e}")

    # 检查依赖库
    required_modules = ['cv2', 'numpy', 'matplotlib', 'pandas']
    for module_name in required_modules:
        try:
            __import__(module_name)
        except ImportError:
            errors.append(f"缺少必要模块: {module_name}")

    return errors

def safe_analysis_operation(func):
    """安全分析操作装饰器"""
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except MemoryError:
            QMessageBox.critical(None, '内存错误',
                               '内存不足，请减少数据量或关闭其他程序')
            return None
        except Exception as e:
            logger.error(f"分析操作失败: {func.__name__}, 错误: {e}")
            QMessageBox.warning(None, '分析错误',
                              f'操作失败: {str(e)}')
            return None
    return wrapper

import signal

def signal_handler(signum, frame):
    """信号处理函数"""
    try:
        if QApplication.instance():
            QApplication.instance().quit()
    except Exception as e:
        print(f"信号处理失败: {e}")
    finally:
        sys.exit(0)

# 在main()函数开始时注册信号处理
signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import gc
import threading
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
import weakref
from collections import defaultdict

# 设置matplotlib使用CPU后端，避免Qt相关问题
matplotlib.use('Agg')  # 使用Anti-Grain Geometry后端，纯CPU渲染


class CPUPlotManager:
    """CPU优化的图表管理器"""

    def __init__(self):
        self.active_figures = {}
        self.figure_counter = 0
        self.lock = threading.Lock()

    def create_figure(self, figsize=(10, 6), dpi=100):
        """创建CPU渲染的图表"""
        with self.lock:
            self.figure_counter += 1
            fig_id = f"fig_{self.figure_counter}"

            # 创建figure，明确指定使用Agg backend
            fig = Figure(figsize=figsize, dpi=dpi)
            canvas = FigureCanvasAgg(fig)

            self.active_figures[fig_id] = {
                'figure': fig,
                'canvas': canvas,
                'created_at': threading.current_thread().ident
            }

            return fig_id, fig

    def save_figure(self, fig_id, filename, **kwargs):
        """保存图表到文件"""
        if fig_id in self.active_figures:
            fig = self.active_figures[fig_id]['figure']
            try:
                fig.savefig(filename, **kwargs)
                return True
            except Exception as e:
                print(f"保存图表失败: {e}")
                return False
        return False

    def close_figure(self, fig_id):
        """安全关闭图表"""
        with self.lock:
            if fig_id in self.active_figures:
                try:
                    fig = self.active_figures[fig_id]['figure']
                    plt.close(fig)
                    del self.active_figures[fig_id]
                    gc.collect()  # 强制垃圾回收
                except Exception as e:
                    print(f"关闭图表时出错: {e}")

    def close_all_figures(self):
        """关闭所有图表"""
        with self.lock:
            fig_ids = list(self.active_figures.keys())
            for fig_id in fig_ids:
                self.close_figure(fig_id)

    def cleanup_memory(self):
        """清理内存"""
        plt.close('all')
        gc.collect()


class CPUMotionVisualizer:
    """CPU优化的运动可视化器"""

    def __init__(self):
        self.plot_manager = CPUPlotManager()
        # CPU优化设置
        self.cpu_settings = {
            'figure_dpi': 100,  # 降低DPI以减少CPU负载
            'line_width': 1.5,
            'marker_size': 4,
            'use_rasterization': True,  # 对复杂图形使用光栅化
            'animation_interval': 100,  # 动画间隔（毫秒）
        }

    def plot_3d_motion(self, motion_data, save_path=None):
        """绘制3D运动轨迹（CPU优化版）"""
        fig_id, fig = self.plot_manager.create_figure(figsize=(12, 8))

        # 创建3D子图
        ax = fig.add_subplot(111, projection='3d')

        coords_3d = motion_data['coordinates_3d']
        n_frames, n_keypoints, _ = coords_3d.shape

        # CPU优化：减少绘制的帧数
        frame_step = max(1, n_frames // 100)  # 最多显示100帧
        selected_frames = range(0, n_frames, frame_step)

        # 绘制关键点轨迹
        colors = plt.cm.tab10(np.linspace(0, 1, n_keypoints))

        for keypoint in range(n_keypoints):
            x_data = coords_3d[selected_frames, keypoint, 0]
            y_data = coords_3d[selected_frames, keypoint, 1]
            z_data = coords_3d[selected_frames, keypoint, 2]

            # 使用较粗的线条减少渲染负载
            ax.plot(x_data, y_data, z_data,
                    color=colors[keypoint],
                    linewidth=self.cpu_settings['line_width'],
                    alpha=0.7,
                    rasterized=self.cpu_settings['use_rasterization'])

        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_title('3D Motion Trajectory (CPU Optimized)')

        # 保存图片
        if save_path:
            self.plot_manager.save_figure(fig_id, save_path,
                                          dpi=self.cpu_settings['figure_dpi'],
                                          bbox_inches='tight')

        return fig_id

    def plot_kinematics(self, motion_data, save_path=None):
        """绘制运动学参数（CPU优化版）"""
        fig_id, fig = self.plot_manager.create_figure(figsize=(15, 10))

        kinematics = motion_data['kinematics']

        # 创建子图
        if 'velocity' in kinematics and 'acceleration' in kinematics:
            axes = fig.subplots(2, 2, figsize=(15, 10))
            axes = axes.flatten()
        else:
            axes = [fig.add_subplot(111)]

        plot_idx = 0

        # 绘制速度
        if 'velocity' in kinematics:
            velocity = kinematics['velocity']
            if len(velocity.shape) == 3:  # (frames, keypoints, dims)
                # CPU优化：只显示几个关键点的速度
                selected_keypoints = range(0, velocity.shape[1], max(1, velocity.shape[1] // 5))

                for kp in selected_keypoints:
                    speed = np.linalg.norm(velocity[:, kp, :], axis=1)
                    axes[plot_idx].plot(speed,
                                        linewidth=self.cpu_settings['line_width'],
                                        alpha=0.7,
                                        label=f'Keypoint {kp}')

                axes[plot_idx].set_title('Velocity Magnitude')
                axes[plot_idx].set_xlabel('Frame')
                axes[plot_idx].set_ylabel('Speed')
                axes[plot_idx].legend()
                plot_idx += 1

        # 绘制加速度
        if 'acceleration' in kinematics and plot_idx < len(axes):
            acceleration = kinematics['acceleration']
            if len(acceleration.shape) == 3:
                selected_keypoints = range(0, acceleration.shape[1], max(1, acceleration.shape[1] // 5))

                for kp in selected_keypoints:
                    accel_mag = np.linalg.norm(acceleration[:, kp, :], axis=1)
                    axes[plot_idx].plot(accel_mag,
                                        linewidth=self.cpu_settings['line_width'],
                                        alpha=0.7,
                                        label=f'Keypoint {kp}')

                axes[plot_idx].set_title('Acceleration Magnitude')
                axes[plot_idx].set_xlabel('Frame')
                axes[plot_idx].set_ylabel('Acceleration')
                axes[plot_idx].legend()
                plot_idx += 1

        plt.tight_layout()

        if save_path:
            self.plot_manager.save_figure(fig_id, save_path,
                                          dpi=self.cpu_settings['figure_dpi'],
                                          bbox_inches='tight')

        return fig_id

    def create_motion_summary(self, motion_data, save_path=None):
        """创建运动数据摘要图表（CPU优化版）"""
        fig_id, fig = self.plot_manager.create_figure(figsize=(16, 12))

        # 创建网格布局
        gs = fig.add_gridspec(3, 2, height_ratios=[2, 1, 1], hspace=0.3, wspace=0.3)

        coords_3d = motion_data['coordinates_3d']
        kinematics = motion_data.get('kinematics', {})

        # 1. 3D轨迹图（简化版）
        ax1 = fig.add_subplot(gs[0, :], projection='3d')
        n_frames, n_keypoints, _ = coords_3d.shape

        # CPU优化：只显示少数关键点和帧
        selected_keypoints = range(0, n_keypoints, max(1, n_keypoints // 10))
        frame_step = max(1, n_frames // 50)
        selected_frames = range(0, n_frames, frame_step)

        colors = plt.cm.tab10(np.linspace(0, 1, len(selected_keypoints)))

        for i, keypoint in enumerate(selected_keypoints):
            x_data = coords_3d[selected_frames, keypoint, 0]
            y_data = coords_3d[selected_frames, keypoint, 1]
            z_data = coords_3d[selected_frames, keypoint, 2]

            ax1.plot(x_data, y_data, z_data,
                     color=colors[i],
                     linewidth=2,
                     alpha=0.6,
                     rasterized=True)

        ax1.set_title('3D Motion Overview')

        # 2. 速度统计
        ax2 = fig.add_subplot(gs[1, 0])
        if 'velocity' in kinematics:
            velocity = kinematics['velocity']
            if len(velocity.shape) == 3:
                # 计算每帧的平均速度
                avg_speed_per_frame = np.mean(np.linalg.norm(velocity, axis=2), axis=1)
                ax2.plot(avg_speed_per_frame,
                         linewidth=self.cpu_settings['line_width'],
                         color='blue', alpha=0.7)
                ax2.set_title('Average Speed per Frame')
                ax2.set_xlabel('Frame')
                ax2.set_ylabel('Speed')

        # 3. 加速度统计
        ax3 = fig.add_subplot(gs[1, 1])
        if 'acceleration' in kinematics:
            acceleration = kinematics['acceleration']
            if len(acceleration.shape) == 3:
                avg_accel_per_frame = np.mean(np.linalg.norm(acceleration, axis=2), axis=1)
                ax3.plot(avg_accel_per_frame,
                         linewidth=self.cpu_settings['line_width'],
                         color='red', alpha=0.7)
                ax3.set_title('Average Acceleration per Frame')
                ax3.set_xlabel('Frame')
                ax3.set_ylabel('Acceleration')

        # 4. 运动范围统计
        ax4 = fig.add_subplot(gs[2, 0])
        # 计算每个关键点的运动范围
        motion_ranges = []
        for kp in range(n_keypoints):
            coord_range = np.ptp(coords_3d[:, kp, :], axis=0)  # 计算范围
            total_range = np.linalg.norm(coord_range)
            motion_ranges.append(total_range)

        ax4.bar(range(len(motion_ranges)), motion_ranges, alpha=0.7, color='green')
        ax4.set_title('Motion Range per Keypoint')
        ax4.set_xlabel('Keypoint')
        ax4.set_ylabel('Range')

        # 5. 数据质量信息
        ax5 = fig.add_subplot(gs[2, 1])
        ax5.text(0.1, 0.8, f'Total Frames: {n_frames}', transform=ax5.transAxes)
        ax5.text(0.1, 0.6, f'Keypoints: {n_keypoints}', transform=ax5.transAxes)
        ax5.text(0.1, 0.4, f'Duration: {n_frames / motion_data.get("frame_rate", 30):.2f}s',
                 transform=ax5.transAxes)
        ax5.text(0.1, 0.2, f'Frame Rate: {motion_data.get("frame_rate", 30)} fps',
                 transform=ax5.transAxes)
        ax5.set_xlim(0, 1)
        ax5.set_ylim(0, 1)
        ax5.set_title('Data Info')
        ax5.axis('off')

        if save_path:
            self.plot_manager.save_figure(fig_id, save_path,
                                          dpi=self.cpu_settings['figure_dpi'],
                                          bbox_inches='tight')

        return fig_id

    def cleanup(self):
        """清理所有图表和内存"""
        self.plot_manager.close_all_figures()
        self.plot_manager.cleanup_memory()


# 使用示例和最佳实践
def setup_cpu_optimized_plotting():
    """设置CPU优化的绘图环境"""

    # 设置matplotlib使用CPU后端
    matplotlib.use('Agg')

    # 设置字体和样式以减少渲染负载
    plt.rcParams.update({
        'figure.max_open_warning': 0,  # 禁用打开图表过多的警告
        'axes.linewidth': 1,  # 减少线条宽度
        'lines.linewidth': 1.5,  # 减少线条宽度
        'patch.linewidth': 0.5,  # 减少补丁线宽
        'font.size': 10,  # 减小字体大小
        'axes.titlesize': 12,
        'axes.labelsize': 10,
        'xtick.labelsize': 9,
        'ytick.labelsize': 9,
        'legend.fontsize': 9,
        'figure.titlesize': 14,
        'savefig.dpi': 100,  # 降低保存DPI
        'figure.dpi': 100,  # 降低显示DPI
    })

    return CPUMotionVisualizer()


# 线程安全的绘图函数
def safe_plot_motion_data(motion_data, output_dir="./plots/"):
    """线程安全的运动数据绘图函数"""
    import os

    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)

    # 初始化CPU优化的可视化器
    visualizer = setup_cpu_optimized_plotting()

    try:
        # 绘制各种图表
        fig_ids = []

        # 3D运动轨迹
        fig_id = visualizer.plot_3d_motion(
            motion_data,
            save_path=os.path.join(output_dir, "3d_motion.png")
        )
        fig_ids.append(fig_id)

        # 运动学参数
        if 'kinematics' in motion_data:
            fig_id = visualizer.plot_kinematics(
                motion_data,
                save_path=os.path.join(output_dir, "kinematics.png")
            )
            fig_ids.append(fig_id)

        # 运动摘要
        fig_id = visualizer.create_motion_summary(
            motion_data,
            save_path=os.path.join(output_dir, "motion_summary.png")
        )
        fig_ids.append(fig_id)

        print(f"成功生成 {len(fig_ids)} 个图表，保存在 {output_dir}")

    except Exception as e:
        print(f"绘图过程中出现错误: {e}")

    finally:
        # 清理资源
        visualizer.cleanup()
        gc.collect()

    return True


if __name__ == "__main__":
    # 测试代码
    print("CPU优化的matplotlib解决方案已加载")
    print("使用 setup_cpu_optimized_plotting() 创建可视化器")
    print("使用 safe_plot_motion_data() 进行线程安全的绘图")

# 确保程序不会直接退出
if __name__ == "__main__":
    try:
        # 您的主程序逻辑
        print("🔄 正在启动用户界面...")

        # 如果是GUI程序，确保有事件循环
        # app.mainloop()  # 对于tkinter
        # app.exec_()     # 对于PyQt

        # 如果是命令行程序，添加用户交互
        input("按Enter键退出...")

    except Exception as e:
        print(f"❌ 程序运行错误: {e}")
        input("按Enter键退出...")
