# Qwen3-8B：数据、SFT + LoRA、测试与项目接入

2026-10-21。用户指定：外部 Linux 服务器、单卡 24GB、Qwen/Qwen3-8B、SFT、普通 LoRA。本次只交付数据和代码，未下载本机 8B 权重、未训练、未租赁或启用付费。

## 1. 当前状态与交付物

原项目全量回归本次执行为 71 passed；真实 API 的 live、documents 和 ready 均返回 200。ready 为 degraded：四库与真实嵌入就绪，回答模型不可用；辅助模型/HyDE、重排、联网尚未配置。生成端集成测试使用替身，并不表示真实问答验收完成。真实数据库有 55 份资料、177 个切片，资料全部 limited。

本轮修改后全量回归：**79 passed，2条第三方弃用警告，106.94秒**。新增8项覆盖回答损失/超长拒绝、split隔离、完整评估分母、摘要行为识别、清单循环哈希、源文银标、服务器文件核验、在线非思考请求；训练脚本CLI和Python编译均通过，GPU训练本身未执行。

本轮数据位于 data/qwen3_sft_v1（本机运行记录或外部数据，未随核心包分发）：

| 文件 | 内容 |
| --- | --- |
| train.jsonl | 180 条训练样本 |
| validation.jsonl | 40 条验证样本，用于验证损失和 checkpoint 选择 |
| test.jsonl | 57 条固定测试样本，不参与训练或挑 checkpoint |
| labels.jsonl | 评估标签 sidecar，不拼进模型输入 |
| sources_snapshot.jsonl | 50 份范围内真实来源及切片快照，可核验原文 |
| dataset_manifest.json | 来源分组、场景/能力分布、冻结文件 SHA256、限制说明 |

同一发布页（含拆分的案例合集）、同标题、相同切片内容按连通组划为同源，共 17 组；先分组，再生成问题。三个 split 均覆盖工资拖欠、解除和经济补偿，五类能力均有样本。来源相关样本不跨 split，测试原始公开资料仍可在 RAG 检索中使用，测试问题/标签/答案不入知识库。

这批是**基于真实来源的确定性工程银标**：摘录/整理原文、引用编号、片段整合、条件缺失和无法保证结论。模板相关性较高，不是专家法律问答金标，不能据此声称法律泛化或专业准确率。未凑满设计目标300/60/90，独立语义评审未执行。初次分组探针目录 data/qwen3_sft 不参与正式训练，统一使用带 v1 的目录。

生成与核验：

```powershell
# 在本机项目根目录；生成新版本请换新目录，已有冻结目录不会被覆盖
.\.venv\Scripts\python.exe -X utf8 scripts/prepare_qwen3_data.py --output data/qwen3_sft_v2
.\.venv\Scripts\python.exe -X utf8 scripts/validate_qwen3_data.py --data-dir data/qwen3_sft_v1
```

prepare 只读本地数据库，无模型调用；重建也可用 `--snapshot data/qwen3_sft_v1/sources_snapshot.jsonl`，无需数据库。validate 已通过 Schema、来源原文子串、引用编号、来源组/切片隔离、标签对齐及冻结摘要核验。**真实 Qwen3 tokenizer 校验尚待训练服务器执行**，不能用字符数估算冒充已验证。

> 2026-10-22执行更新：本手册中的24GB为起步配置，用户实际租用RTX4090约48GB显存实例。真实tokenizer、10步探针、正式LoRA、base/adapter预测和合并导出均已完成；在线部署与接入仍在验收。实际证据优先参见远程训练执行记录（本机运行记录或外部数据，未随核心包分发），下文“未执行”描述保留其交付时语境。

## 2. 训练方法与显存设置

SFT 是训练目标，LoRA 是可训练参数更新方式：冻结 BF16 基座，只训练 q_proj/v_proj 的 LoRA 权重，不进行 4bit/8bit 量化，也不自动切换 QLoRA。

| 参数 | 本次起步值 |
| --- | --- |
| 基座 | Qwen/Qwen3-8B；必须记录实际下载的40位 commit SHA |
| 精度 | BF16，需要支持BF16的CUDA卡；不静默切CPU/FP16 |
| LoRA | r=8、alpha=16、dropout=0.05；q_proj/v_proj |
| 长度 | 完整训练序列1536；在线输入1024、输出512，含模板开销 |
| batch / 累积 | 1 / 8，有效batch=8 |
| epochs / lr | 2 / 1e-4 |
| 节省显存 | SDPA、梯度检查点（use_reentrant=false）、训练use_cache=false |
| 损失 | system/user/非思考前缀/padding 为 -100，只训练回答与EOS |
| 保存 | 每epoch验证和保存，正式训练加载最低验证损失checkpoint |
| 比较 | base/adapter同基础快照、模板、固定证据、非思考模式、贪心解码、长度上限 |

8.2B BF16 基座权重约16.4GB（十进制估算），还需激活、梯度与运行时空间。单卡24GB是本配置的目标条件，不是必然成功保证；代码接受实际总显存≥22GiB以兼容容量报告方式，必须先做10步探针看 allocated/reserved 峰值。训练与vLLM不能同时占用同一张卡。

Qwen3 使用 Transformers≥4.51 才支持；本项目训练约束提升为≥4.57.3,<5。模型支持 thinking/non-thinking；本项目为三段式证据回答，训练、离线推理、在线请求固定 `enable_thinking=false`，不训练或展示思维链。依据：[Qwen3-8B 官方模型卡](https://huggingface.co/Qwen/Qwen3-8B)、[PEFT LoRA](https://huggingface.co/docs/peft/main/en/task_guides/lora_based_methods)。

## 3. 外部服务器准备

把整个项目传到服务器，例如 `/srv/RAG`；不要上传本机 `.env`、`.venv`、数据库卷或密钥。至少需要 training/、src/legal_rag/、scripts/validate_qwen3_data.py、docs/contracts/默认配置.json、data/qwen3_sft_v1/。服务器已有的基础模型不必重复下载。

在服务器执行（路径和快照使用自己的真实值）：

```bash
cd /srv/RAG
python3.11 -m venv .venv-train
source .venv-train/bin/activate

# 先按照服务器GPU驱动和CUDA安装匹配的PyTorch；已有CUDA PyTorch镜像可直接使用。
# 例如服务器驱动支持CUDA12.8时：
python -m pip install 'torch>=2.6,<3' --index-url https://download.pytorch.org/whl/cu128
python -m pip install -r training/requirements-qwen3.txt
python -m pip freeze > training-environment.freeze.txt
nvidia-smi

export BASE_MODEL_PATH=/srv/models/Qwen3-8B
# 从下载记录或Hugging Face snapshots/<commit>目录取得，不能填写main或编造SHA。
export BASE_REVISION=替换为实际40位commit_SHA

python -c 'import torch; print(torch.cuda.is_available(), torch.cuda.is_bf16_supported(), torch.cuda.get_device_name(0))'
python scripts/validate_qwen3_data.py --data-dir data/qwen3_sft_v1 --tokenizer-path "$BASE_MODEL_PATH"
```

模型路径存在时以 `local_files_only=true` 加载，避免意外重复下载。验证报告列出超长样本及原因；训练/验证超长剔除并记录，绝不静默截断目标答案。固定测试集不删题，推理时超长或执行失败保留失败记录计入分母。若大量样本超长，先调整资料/输入设计并建立新版本数据集，不偷偷改固定测试题。

## 4. 探针、正式训练和基座对比

```bash
# 探针使用独立run，产物名adapter_smoke，不能冒充正式训练。
python -m training.train_lora --model-path "$BASE_MODEL_PATH" --base-revision "$BASE_REVISION" \
  --data-dir data/qwen3_sft_v1 --run-id qwen3-8b-smoke-001 --smoke-steps 10

# 查看 artifacts/qwen3-8b-smoke-001/environment.json 与 run_summary.json 的显存/步数。
# 确认可运行并满足自己的时间/费用门槛后再正式训练：
python -m training.train_lora --model-path "$BASE_MODEL_PATH" --base-revision "$BASE_REVISION" \
  --data-dir data/qwen3_sft_v1 --run-id qwen3-8b-sft-001

python -m training.predict --run-id qwen3-8b-sft-001 --variant base --data-dir data/qwen3_sft_v1
python -m training.predict --run-id qwen3-8b-sft-001 --variant adapter --data-dir data/qwen3_sft_v1
python -m training.evaluate --run-id qwen3-8b-sft-001 --data-dir data/qwen3_sft_v1
```

如果1536长度OOM：停止同卡其他程序，以新run-id增加 `--max-seq-length 1024` 重做探针；完整输入预算同时降至512，输出仍512，并重新检查可保留样本数量。不能靠删测试难题或把QLoRA称作普通LoRA来“跑通”。仍不可行时再决定量化或更大显存，不在代码中自动改变方法。

每个正式run保存 config、数据文件摘要、实际使用/剔除计数、环境、日志、验证最优checkpoint、适配器和两套预测。predict和merge只读该run保存的配置，避免业务默认配置变化污染对比。评估使用完整冻结测试集为分母，缺失/失败题不删除；拒绝重复/未知ID，核对两套解码与测试摘要一致。报告的行为识别是规则代理、引用指标是编号有效率，忠实性专业判断未执行时标not_run，结果允许无提升。

训练、预测和验证为GPU步骤，本轮未执行；不提供虚构loss、显存峰值或提升数字。

## 5. 合并导出与独立部署

```bash
python -m training.merge_export --run-id qwen3-8b-sft-001
# 导出：artifacts/qwen3-8b-sft-001/merged_model/
# 轻量tokenizer：artifacts/qwen3-8b-sft-001/tokenizer/
```

合并在CPU以BF16加载，需足够主机内存和约十几GB以上可用磁盘；建议预留48GB以上主机内存和至少40GB磁盘，并以实际文件大小核对。合并后含完整权重，不能只把 adapter_model.safetensors 当完整模型部署。

deployment_manifest.json 含基础快照、适配器SHA256、每个模型文件摘要、整体摘要、tokenizer和chat_template摘要。整体摘要明确排除清单自身，避免循环哈希；服务启动先逐文件核验，再从同一个模型目录启动。可先执行：

```bash
python -m training.serve_model --model-dir artifacts/qwen3-8b-sft-001/merged_model --verify-only
```

vLLM用另一套Linux环境，与训练环境分开。它严格依赖对应torch/CUDA，不把它装入已验证的训练环境破坏依赖：

```bash
python3.11 -m venv .venv-serve
source .venv-serve/bin/activate
python -m pip install 'vllm>=0.8.5' 'pydantic-settings>=2.6,<3' 'httpx>=0.27,<0.29'
python -m pip freeze > serving-environment.freeze.txt

# 将自己的服务密钥注入环境，勿提交Git或打印到日志。
read -r -s -p 'Model service API key: ' MODEL_SERVICE_API_KEY; echo
export MODEL_SERVICE_API_KEY
python -m training.serve_model --model-dir artifacts/qwen3-8b-sft-001/merged_model --port 8008
```

serve_model逐文件验证后执行vLLM：绑定127.0.0.1、served_model_name=legal-labor-sft-v1、BF16、max_model_len=1536、max_num_seqs=1、gpu_memory_utilization=0.85、enforce_eager。启动成功与模型实际回答是不同检查；本次未在24GB服务器验证这些参数，不承诺SLA。官方依据：[Qwen vLLM部署与非思考API](https://qwen.readthedocs.io/en/latest/deployment/vllm.html)。

## 6. 接入当前Windows RAG

Windows项目电脑只需要可信部署清单和轻量tokenizer，不需要复制8B权重。先通过服务器 `serve_model --verify-only` 和启动日志核对实际部署目录；`/v1/models` 只有别名，不能单独证明远程权重身份。本机只持有清单时信任用户控制的服务器和清单传输，不声称本机重新哈希了远程权重。

从Windows PowerShell建立受控连接（替换用户名/主机）：

```powershell
# 此终端保持开启；服务器上的vLLM只绑定回环。
ssh -N -L 8008:127.0.0.1:8008 用户名@模型服务器
```

另一个终端复制必要文件：

```powershell
Set-Location 'C:\Projects\legal-advisor-rag'
New-Item -ItemType Directory -Force deploy/model-qwen3 | Out-Null
scp 用户名@模型服务器:/srv/RAG/artifacts/qwen3-8b-sft-001/merged_model/deployment_manifest.json deploy/model-qwen3/deployment_manifest.json
scp -r 用户名@模型服务器:/srv/RAG/artifacts/qwen3-8b-sft-001/tokenizer deploy/model-qwen3/
```

在本机 `.env` 更新（不改前端，不把密钥提交Git）：

```properties
LLM_BASE_URL=http://127.0.0.1:8008/v1
LLM_MODEL=legal-labor-sft-v1
LLM_API_KEY=与模型服务一致的真实密钥
MODEL_DEPLOYMENT_MANIFEST=deploy/model-qwen3/deployment_manifest.json
MODEL_TOKENIZER_PATH=deploy/model-qwen3/tokenizer
```

MODEL_TOKENIZER_PATH 是本轮新增字段，用于真实模板计数；修复了原代码把LLM_BASE_URL误传为本地tokenizer路径的问题。后端同时核验tokenizer目录摘要。安装的transformers需支持Qwen3（当前lock为4.57.6），这里仅加载tokenizer，不加载回答模型权重。

本机非可编辑安装需更新代码，然后重启API和worker（旧进程不会自动读取新.env）：

```powershell
uv sync --extra embedding --no-editable --reinstall-package legal-rag
uv run uvicorn legal_rag.api.main:app --host 127.0.0.1 --port 8000
# 另一个终端：uv run python -X utf8 -m legal_rag.worker.main
uv run python -X utf8 scripts/verify_model_connection.py
```

verify依次核验配置/清单/tokenizer、/v1/models、普通/流式回答、一次真实RAG回答的模型ID和产物摘要。外部服务器先核验真实权重，本机核验可信清单与tokenizer；两者共同作为接入证据。还须补验三个场景各一次问答与追问、SSE断线恢复，并停模型服务确认明确失败且不改用辅助云模型。训练完成不能替代这些在线验收。

## 7. 仍需补充的真实验证

本次工程测试不能替代24GB服务器上的真实tokenizer/10步探针/完整训练/两套预测/模型接入。辅助抽取/HyDE、精排和联网仍未配置，默认流程会明确降级；是否启用由实际配置与预算决定。300元旧预算上限继续保留，8B训练及服务的真实费用须在服务器执行前重算，不自动租GPU或改变付费开关。
