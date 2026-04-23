from pathlib import Path
import pandas as pd
from rerun.dataframe import load_archive

# 1. 加载你的 RRD 文件
rrd_path = Path("/home/liuyou/Documents/data/genrobot_ego_data_beta_v1.rrd")
archive = load_archive(rrd_path)

# 2. 从归档中取出所有录制数据
recordings = archive.all_recordings()
recording = recordings[0]

# 3. 查看数据结构
print("=== 数据结构 ===")
schema = recording.schema()
print("索引列(时间线):", [col.name for col in schema.index_columns()])
print("组件列(实体数据):", [col.name for col in schema.component_columns()])

# 4. 创建视图
view = recording.view(index="log_tick", contents="/**")
reader = view.select()

# 5. 统计总批次数量（核心修改）
total_batches = 0
for batch in reader:
    total_batches += 1

# 输出总批次
print(f"\n=== 数据总批次数量：{total_batches} ===")
