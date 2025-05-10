import os
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed

# 最大并发进程数
MAX_WORKERS = 6
# 搜索目录
SEARCH_DIR = 'filelist'

# 构造命令函数
def build_command(csv_path):
    return ['python', 'preprocess.py', '-c', 'configs/reflow.yaml', '-r', 'data/train/', '-f', csv_path]

# 执行命令并收集输出
def run_command(cmd):
    print("run", cmd)
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        return (cmd[-1], result.stdout, result.stderr, 0)
    except subprocess.CalledProcessError as e:
        return (cmd[-1], e.stdout, e.stderr, e.returncode)

def main():
    # 搜索所有符合条件的 csv 文件
    csv_files = [
        os.path.join(SEARCH_DIR, f)
        for f in os.listdir(SEARCH_DIR)
        if f.endswith('.csv') and not f.startswith('completed')
    ]

    if not csv_files:
        print("没有找到符合条件的 CSV 文件。")
        return

    # 用线程池来模拟并发执行（适用于 I/O 密集型任务）
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {executor.submit(run_command, build_command(csv)): csv for csv in csv_files}

        for future in as_completed(futures):
            csv_file = futures[future]
            try:
                filename, stdout, stderr, code = future.result()
                # 如果正常
                print(f"退出码: {code}")
                if code != 0:
                    print("Error")

                    print("标准输出:")
                    print(stdout)
                    print("标准错误:")
                    print(stderr)
                print(f"--- 完成: {filename} ---\n")
            except Exception as exc:
                print(f"处理 {csv_file} 时出错: {exc}")

if __name__ == "__main__":
    main()
