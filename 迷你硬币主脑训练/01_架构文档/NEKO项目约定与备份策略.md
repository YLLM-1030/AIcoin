# NEKO / mini_coin 项目约定

## 备份策略（三级）
- **轻**: Coin 经验包 (`Desktop/Coin/`) — 每次大改动后更新
- **中**: 配置 + 模型权重 (`~/models/` + `~/.ollama/models/blobs/`) — 每周
- **重**: WSL 全量 tar (`D:\backup\wsl-neko-*.tar`) — 每月/大变更前

## 部署铁律
- 三个 Python 服务 (server.py, v10_server.py, autonomic.py) 必须在同一环境（全部 WSL）
- WSL→Windows 不做端口转发，bridge 不能单独放 Windows
- 启动方式：`_launch_all_wsl.sh` 一键，或分步启动看 `Coin/04_环境与部署/`

## 代码位置
- 主项目：`Desktop/NEKO/活跃项目/ai模型项目/整合最新版/puppet_engine/`
- 文档：`Desktop/Coin/`（新经验包）
- 旧经验包：`Desktop/NEKO/轻量备份_经验包/`（已废弃，待删除）

## 问题排查
- 先查 Coin 经验包对应文件夹 → 再搜 GitHub/Reddit/CSDN/知乎
- 禁止贴膏药：研究根因，不掩盖错误
- 每个数据传输环节加文件日志（`/tmp/xxx.log`），不要依赖 stdout 或猜测

## 显存管理
- 24GB 上限，四个模型不能同时驻留：neko-v7(14.2G) + qwen2.5:0.5b(1.4G) + GNDINO(3G) + Qwen2.5-VL(4.4G) = 23G
- VL 模型限制 n_gpu_layers=20；非必要时卸载 qwen2.5:0.5b
- 紧急释放: `pkill -9 ollama` 或 `wsl --shutdown`
- `wsl --shutdown` 会杀死所有 WSL 进程（包括 Ollama），之后需全栈重启

## GNDINO 坐标格式
- `predict()` 返回**归一化 cxcywh** [center_x, center_y, width, height]，值域 0-1
- 必须乘以图像宽高做反归一化，否则 bbox 全为 [0,0,0,0]

## WSL 网络陷阱
- `wsl --shutdown` 后 localhost 转发需数秒重建
- WSL 内服务绑 0.0.0.0（非 127.0.0.1）确保 Windows 浏览器可访问
- Windows 端用 `python server.py &` 启动的僵尸进程不会被 shutdown 杀死：`netstat -ano | findstr <port>` 检查


## 2026-06-05 待办 (已规划, 未实施)

### 1. 测试 mini_coin 新目录能否跑通
- 重启后验证 Desktop/mini_coin/ 路径下 WSL 全栈启动
- 删除旧 Desktop/NEKO/ 目录

### 2. Tool Call 后无回复 bug
- 现象: LLM 执行 tool 后沉默，不继续回复
- 例如: [TOOL:read_file] → 读到内容 → 应该把内容说给用户，但实际没有
- 排查方向: v10_server tool calling 循环的返回值处理逻辑

### 3. 情绪系统 — 用论文映射替代随机
- 当前: 情绪 random, 经常 happy/surprise
- 目标: 按论文的情绪模型映射，LLM 理解自己的情绪状态
- 参考: 外部论文 (情绪维度/离散情绪映射)

### 4. 记忆系统升级 — NEKO 项目层级
- 当前: 记忆 API 管道 + 向量记忆
- 目标: 达到 NEKO 项目 (证据层+反思层) 的五层记忆效果
- 方向: 在现有两步管道基础上增加反思层

### 5. 弹幕读取系统
- 当前: danmaku_listener.py 已删除 (B站协议过期)
- 目标: 重新对接 B站 WebSocket 弹幕协议

### 6. 语音交互 (打电话式)
- 目标: 能语音输入→文字→LLM→TTS→语音输出
- 方案: 
  (a) 语音识别 (STT) → 自动填入对话框
  (b) 对话框文字 → 自动发送 → LLM 回复 → TTS 播放
- 有老代码可参考，如果太复杂拆成两步实现

### 7. 教她玩 Minecraft
- 参考: NEKO 代码 + GitHub 相关项目
- 方向: 视觉+键鼠操作 Minecraft 游戏


## 2026-06-05 改名 + 工具调用 + 未完成状态

### 改名 NEKO→mini_coin
- 旧目录 Desktop/NEKO/ 锁定, 新目录 Desktop/mini_coin/ 就绪
- 模型: mini_coin-v7:latest (alias已建, 旧名仍可用)
- 所有代码引用、路径、网页标题已更新
- WSL 启动路径: /mnt/.../Desktop/mini_coin/.../puppet_engine/

### read_file 工具
- execute_tool 新增 read_file, 安全限制: 仅项目目录, 10MB/50000字符上限
- SOUL prompt 已更新: 强制使用读文件工具, 禁止编造内容
- LLM直接调Ollama已验证能输出 [TOOL:read_file]
- 通过 v10_server /chat 端点未完成验证 (WSL进程不稳)

### 修复
- v10 num_predict: 512→2048 (工具调用需更多token)
- SOUL_PROMPT_FILE 路径加 normpath
- 删旧内存代码残留 (facts变量)

### 备份
- 中量: Desktop/6.423，47中量备份/ (13GB, 含mini_coin-v7 8.3GB)
- 重量: D://backup//wsl-neko-full-2026-06-04-visual-pipeline-fixed.tar (96.2GB)

### 下次启动
1. 删 Desktop/NEKO/ (手动)
2. 用 _launch_all_wsl.sh 从 mini_coin 路径启动
3. 测 read_file 闭环
4. 继续7项待办 (MEMORY.md#2026-06-05-待办)


## 2026-06-05 read_file 踩坑 7 连击
1. SOUL模板 path/to/file.txt → LLM原样复制 → 改 <文件路径>
2. Windows路径 C://Users → WSL不识 → 自动转 /mnt/c/
3. chat_stream 无工具循环 → 前端看到TOOL但永不执行 → 给stream加循环
4. tool_history 继承请求历史 → 状态跨请求污染 → 独立初始化
5. keep_alive:-1 KV缓存污染 → 去掉
6. UI 🔧 被推出屏幕 → 调DOM顺序
7. 读目录报错 → 自动列目录 + .doc/.docx支持
