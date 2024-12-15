import pandas as pd
import numpy as np
import akshare as ak
import os
from datetime import datetime, timedelta
import logging
import yaml
from pathlib import Path
from typing import List, Dict, Tuple
import time
from concurrent.futures import ThreadPoolExecutor
import threading
import shutil

class AStockManager:
    def __init__(self):
        # 加载配置
        self.config = self.load_config()
        
        # 初始化路径
        self.setup_paths()
        
        # 设置日志
        self.setup_logging()
        
        # 初始化其他属性
        self.current_date = datetime.now().strftime('%Y%m%d')
        self.lock = threading.Lock()
        self.stock_list = None
        self.stock_names = None

    def load_config(self) -> dict:
        """加载配置文件"""
        config_path = Path(__file__).parent / 'ach_config.yaml'
        if not config_path.exists():
            raise FileNotFoundError("配置文件不存在")
        
        with open(config_path, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f)


    def setup_paths(self):
        """设置并创建必要的目录"""
        self.base_dir = Path(self.config['paths']['base_dir']).resolve()
        self.data_dir = self.base_dir / self.config['paths']['data_dir']
        self.backup_dir = self.base_dir / self.config['paths']['backup_dir']

        # 创建必要的目录
        for directory in [self.data_dir, self.backup_dir]:
            directory.mkdir(parents=True, exist_ok=True)


    def setup_logging(self):
        """配置日志系统"""
        log_file = self.base_dir / 'china_stock.log'
        
        # 创建自定义格式化器
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        
        # 创建文件处理器
        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_handler.setFormatter(formatter)
        
        # 创建控制台处理器
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        
        # 设置日志记录器
        logger = logging.getLogger('ChinaStockScreener')
        logger.setLevel(logging.INFO)
        logger.handlers = []  # 清除现有的处理器
        logger.addHandler(file_handler)
        logger.addHandler(console_handler)
        
        self.logger = logger

    def get_stock_list(self) -> List[str]:
        """获取符合条件的A股股票列表，带重试机制"""
        if self.stock_list is None:
            max_retries = 3
            for attempt in range(max_retries):
                try:
                    stock_list_df = ak.stock_zh_a_spot_em()
                    # 根据配置的前缀筛选股票
                    prefixes = tuple(self.config['stock_prefixes'])
                    valid_stocks = stock_list_df[
                        stock_list_df['代码'].str.startswith(prefixes)
                    ]
                    self.stock_list = valid_stocks['代码'].tolist()
                    self.stock_names = dict(zip(valid_stocks['代码'], valid_stocks['名称']))
                    break
                except Exception as e:
                    if attempt == max_retries - 1:
                        self.logger.error(f"获取股票列表失败: {e}")
                        raise
                    time.sleep(5)
        return self.stock_list

    def calculate_cvd(self, data: pd.DataFrame) -> pd.DataFrame:
        """计算CVD指标"""
        # 计算蜡烛图组成部分
        data['Upper_Wick'] = np.where(
            data['close'] > data['open'],
            data['high'] - data['close'],
            data['high'] - data['open']
        )
        data['Lower_Wick'] = np.where(
            data['close'] > data['open'],
            data['open'] - data['low'],
            data['close'] - data['low']
        )
        data['Spread'] = data['high'] - data['low']
        data['Body_Length'] = data['Spread'] - (data['Upper_Wick'] + data['Lower_Wick'])

        # 处理除零情况
        mask = data['Spread'] != 0
        data.loc[mask, 'Percent_Upper_Wick'] = data.loc[mask, 'Upper_Wick'] / data.loc[mask, 'Spread']
        data.loc[mask, 'Percent_Lower_Wick'] = data.loc[mask, 'Lower_Wick'] / data.loc[mask, 'Spread']
        data.loc[mask, 'Percent_Body_Length'] = data.loc[mask, 'Body_Length'] / data.loc[mask, 'Spread']

        # 填充NaN
        data[['Percent_Upper_Wick', 'Percent_Lower_Wick', 'Percent_Body_Length']] = \
            data[['Percent_Upper_Wick', 'Percent_Lower_Wick', 'Percent_Body_Length']].fillna(0)

        # 计算买卖量
        data['Buying_Volume'] = np.where(
            data['close'] > data['open'],
            (data['Percent_Body_Length'] + (data['Percent_Upper_Wick'] + data['Percent_Lower_Wick']) / 2) * data['volume'],
            ((data['Percent_Upper_Wick'] + data['Percent_Lower_Wick']) / 2) * data['volume']
        )
        data['Selling_Volume'] = np.where(
            data['close'] < data['open'],
            (data['Percent_Body_Length'] + (data['Percent_Upper_Wick'] + data['Percent_Lower_Wick']) / 2) * data['volume'],
            ((data['Percent_Upper_Wick'] + data['Percent_Lower_Wick']) / 2) * data['volume']
        )

        # 计算CVD
        period = self.config['screening']['cvd_period']
        data['Cumulative_Buying_Volume'] = data['Buying_Volume'].ewm(span=period, adjust=False).mean()
        data['Cumulative_Selling_Volume'] = data['Selling_Volume'].ewm(span=period, adjust=False).mean()
        data['CVD'] = data['Cumulative_Buying_Volume'] - data['Cumulative_Selling_Volume']

        return data

    def calculate_udvr(self, data: pd.DataFrame) -> pd.DataFrame:
        """计算UDVR指标"""
        try:
            period = self.config['screening']['udvr_period']
            
            # 计算涨跌
            data['price_change'] = data['close'].diff()

            # 定义上升和下降成交量
            data.loc[:, 'up_vol'] = 0
            data.loc[:, 'down_vol'] = 0
            data.loc[data['price_change'] > 0, 'up_vol'] = data['volume']
            data.loc[data['price_change'] < 0, 'down_vol'] = data['volume']

            # 计算滚动和
            data['sum_up_vol'] = data['up_vol'].rolling(window=period, min_periods=1).sum()
            data['sum_down_vol'] = data['down_vol'].rolling(window=period, min_periods=1).sum() + 0.00001

            # 计算UDVR
            data['UDVR'] = (data['sum_up_vol'] / data['sum_down_vol']).round(6)
            data['UDVR_EMA'] = data['UDVR'].ewm(span=period, min_periods=1, adjust=False).mean().round(6)

            # 清理中间计算列
            data = data.drop(['price_change', 'up_vol', 'down_vol', 'sum_up_vol', 'sum_down_vol'], axis=1)

            return data

        except Exception as e:
            self.logger.error(f"UDVR计算错误: {str(e)}")
            raise
    def update_stock_data(self, ticker: str):
        """更新单个股票的数据"""
        try:
            file_path = self.data_dir / f'{ticker}.csv'

            # 获取新数据
            new_data = ak.stock_zh_a_hist(
                symbol=ticker,
                period='daily',
                start_date=self.config['akshare']['start_date'],
                end_date=self.current_date,
                adjust=""
            )

            if new_data.empty:
                self.logger.info(f"{ticker}: 没有数据")
                return

            # 统一列名
            new_data = new_data.rename(columns={
                '日期': 'date',
                '开盘': 'open',
                '收盘': 'close',
                '最高': 'high',
                '最低': 'low',
                '成交量': 'volume'
            })

            # 确保数据类型正确
            new_data['volume'] = new_data['volume'].astype(float)
            new_data['close'] = new_data['close'].astype(float)
            new_data['open'] = new_data['open'].astype(float)
            new_data['high'] = new_data['high'].astype(float)
            new_data['low'] = new_data['low'].astype(float)

            # 设置日期索引
            new_data['date'] = pd.to_datetime(new_data['date'])
            new_data.set_index('date', inplace=True)

            # 按日期排序
            new_data = new_data.sort_index()

            # 计算技术指标
            self.logger.info(f"Processing {ticker}")
            new_data = self.calculate_cvd(new_data)
            new_data = self.calculate_udvr(new_data)

            # 保存数据
            new_data.to_csv(file_path)
            self.logger.info(f"{ticker}: 数据更新完成")

        except Exception as e:
            self.logger.error(f"{ticker}: 更新失败 - {str(e)}")
            raise

    def update_stock_batch(self, tickers: List[str]):
        """批量更新股票数据"""
        for ticker in tickers:
            try:
                self.update_stock_data(ticker)
                time.sleep(0.2)  # 小延迟避免请求过密
            except Exception as e:
                self.logger.error(f"{ticker}: 批量更新失败 - {str(e)}")

    def check_cvd_pattern1(self, data: pd.DataFrame) -> bool:
        """检查CVD模式1: ---++类型"""
        if len(data) < 6:
            return False

        recent_cvd = data['CVD'].iloc[-6:]
        cvd_signs = recent_cvd.apply(lambda x: '+' if x > 0 else '-' if x < 0 else '0')
        pattern_5_days = ''.join(cvd_signs[-5:])
        pattern_6_days = ''.join(cvd_signs[-6:])

        valid_patterns = [
            '---++', '---+++',
            '--0++', '--0+++',
            '0--++', '0--+++'
        ]

        return pattern_5_days in valid_patterns or pattern_6_days in valid_patterns

    def check_cvd_pattern2(self, data: pd.DataFrame) -> bool:
        """检查CVD模式2: 连续4天递增"""
        if len(data) < 7:
            return False

        recent_cvd = data['CVD'].iloc[-7:]
        for i in range(len(recent_cvd) - 3):
            cvd_sequence = recent_cvd.iloc[i:i + 4]
            if cvd_sequence.is_monotonic_increasing and cvd_sequence.nunique() > 1:
                return True
        return False

    def check_udvr_conditions(self, data: pd.DataFrame) -> Dict[float, Tuple[float, float]]:
        """检查多个UDVR阈值条件"""
        if len(data) < 2:
            return {}
        
        last_two_udvr = data['UDVR'].iloc[-2:].values
        if len(last_two_udvr) != 2:
            return {}
        
        results = {}
        for threshold in self.config['screening']['udvr_thresholds']:
            if last_two_udvr[0] >= threshold and last_two_udvr[1] >= threshold:
                results[threshold] = (last_two_udvr[0], last_two_udvr[1])
        
        return results

    def select_stocks(self) -> Tuple[List[Dict], List[Dict], Dict[float, List[Dict]]]:
        """选股主程序"""
        self.logger.info("开始选股...")

        results_pattern1 = []  # ---++模式
        results_pattern2 = []  # 连续4天递增模式
        results_udvr = {threshold: [] for threshold in self.config['screening']['udvr_thresholds']}

        # 遍历所有股票数据文件
        for filename in os.listdir(self.data_dir):
            if not filename.endswith('.csv'):
                continue

            ticker = filename.replace('.csv', '')
            try:
                # 读取股票数据
                data = pd.read_csv(self.data_dir / filename)
                data['date'] = pd.to_datetime(data['date'])
                data.set_index('date', inplace=True)

                # 检查CVD模式
                if self.check_cvd_pattern1(data):
                    results_pattern1.append({
                        'Ticker': ticker,
                        'Name': self.stock_names.get(ticker, 'Unknown')
                    })
                    self.logger.info(f"{ticker}: 符合模式1(---++)")

                if self.check_cvd_pattern2(data):
                    results_pattern2.append({
                        'Ticker': ticker,
                        'Name': self.stock_names.get(ticker, 'Unknown')
                    })
                    self.logger.info(f"{ticker}: 符合模式2(连续4天递增)")

                # 检查UDVR条件
                udvr_matches = self.check_udvr_conditions(data)
                for threshold, (udvr_prev, udvr_current) in udvr_matches.items():
                    results_udvr[threshold].append({
                        'Ticker': ticker,
                        'Name': self.stock_names.get(ticker, 'Unknown'),
                        'UDVR_Previous': round(udvr_prev, 2),
                        'UDVR_Current': round(udvr_current, 2)
                    })
                    self.logger.info(f"{ticker}: 符合UDVR>={threshold}条件")

            except Exception as e:
                self.logger.error(f"{ticker}: 处理失败 - {str(e)}")

        return results_pattern1, results_pattern2, results_udvr

    def save_results(self, results_pattern1: List[Dict], results_pattern2: List[Dict],
                    results_udvr: Dict[float, List[Dict]]):
        """保存所有选股结果"""
        saved_files = []

        # 保存模式1(---++)的结果
        if results_pattern1:
            base_filename1 = f'{self.current_date}_china_cvd---++'
            csv_file1 = self.base_dir / f'{base_filename1}.csv'
            txt_file1 = self.base_dir / f'{base_filename1}.txt'
            txt_file1_names = self.base_dir / f'{base_filename1}_names.txt'

            df_results1 = pd.DataFrame(results_pattern1)
            df_results1.to_csv(csv_file1, index=False)
            df_results1['Ticker'].to_csv(txt_file1, index=False, header=False)
            df_results1[['Ticker', 'Name']].to_csv(
                txt_file1_names, index=False, header=False, sep='\t')

            saved_files.extend([csv_file1, txt_file1, txt_file1_names])
            self.logger.info(f"模式1(---++)共有 {len(results_pattern1)} 家公司符合条件")

        # 保存模式2(连续4天递增)的结果
        if results_pattern2:
            base_filename2 = f'{self.current_date}_china_cvd_4up'
            csv_file2 = self.base_dir / f'{base_filename2}.csv'
            txt_file2 = self.base_dir / f'{base_filename2}.txt'
            txt_file2_names = self.base_dir / f'{base_filename2}_names.txt'

            df_results2 = pd.DataFrame(results_pattern2)
            df_results2.to_csv(csv_file2, index=False)
            df_results2['Ticker'].to_csv(txt_file2, index=False, header=False)
            df_results2[['Ticker', 'Name']].to_csv(
                txt_file2_names, index=False, header=False, sep='\t')

            saved_files.extend([csv_file2, txt_file2, txt_file2_names])
            self.logger.info(f"模式2(连续4天递增)共有 {len(results_pattern2)} 家公司符合条件")

        # 保存不同阈值的UDVR结果
        for threshold in sorted(results_udvr.keys(), reverse=True):
            results = results_udvr[threshold]
            if results:
                base_filename = f'{self.current_date}_china_udvr{int(threshold)}+'
                txt_file = self.base_dir / f'{base_filename}.txt'
                txt_file_names = self.base_dir / f'{base_filename}_names.txt'

                # 保存详细结果
                with open(txt_file, 'w', encoding='utf-8') as f:
                    for result in results:
                        line = f"{result['Ticker']}\t{result['Name']}\t{result['UDVR_Previous']}\t{result['UDVR_Current']}\n"
                        f.write(line)

                # 保存简化版本（仅代码和名称）
                df_results = pd.DataFrame(results)
                df_results[['Ticker', 'Name']].to_csv(
                    txt_file_names, index=False, header=False, sep='\t')

                saved_files.extend([txt_file, txt_file_names])
                self.logger.info(f"UDVR>={threshold}模式共有 {len(results)} 家公司符合条件")

        # 复制结果到备份目录
        for file_path in saved_files:
            try:
                shutil.copy(file_path, self.backup_dir)
                self.logger.info(f"已复制 {file_path.name} 到备份目录")
            except Exception as e:
                self.logger.error(f"复制文件 {file_path.name} 时出错: {e}")

    def run_update(self):
        """运行数据更新程序"""
        self.logger.info("开始更新股票数据...")
        
        # 获取股票列表
        tickers = self.get_stock_list()
        self.logger.info(f"共需要更新 {len(tickers)} 只股票的数据")
        
        # 批量处理股票
        batch_size = self.config['akshare']['batch_size']
        batch_delay = self.config['akshare']['batch_delay']
        
        for i in range(0, len(tickers), batch_size):
            batch = tickers[i:i + batch_size]
            self.update_stock_batch(batch)
            if i + batch_size < len(tickers):
                time.sleep(batch_delay)
        
        self.logger.info("所有股票数据更新完成")

    def run(self):
        """运行完整的更新和选股程序"""
        try:
            # 第一步：更新数据
            self.run_update()
            
            # 第二步：选股
            results_pattern1, results_pattern2, results_udvr = self.select_stocks()
            
            # 第三步：保存结果
            self.save_results(results_pattern1, results_pattern2, results_udvr)
            
            self.logger.info("程序执行完成")
            
        except Exception as e:
            self.logger.error(f"程序执行出错: {str(e)}")
            raise

def main():
    manager = AStockManager()
    manager.run()

if __name__ == "__main__":
    main()
