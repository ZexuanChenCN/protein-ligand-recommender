"""
蛋白-小分子检索网页后端（移除亲和力字段版）
适配修改后的mips_retrieval.py，删除所有affinity相关逻辑
"""
import os
import sys
import json
import warnings
import threading
import time
warnings.filterwarnings("ignore")

# 添加项目根目录到Python路径
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from flask import Flask, render_template, request, jsonify
# 导入推理模块，添加异常处理
try:
    from inference.mips_retrieval import ProteinLigandRetriever
    RETRIEVER_AVAILABLE = True
except ImportError as e:
    print(f"⚠️ 推理模块导入失败：{e}，将使用模拟数据")
    RETRIEVER_AVAILABLE = False

# ========== 初始化Flask应用 ==========
app = Flask(__name__)

# ========== 模型配置 + 全局状态 ==========
CHECKPOINT_PATH = "../model/checkpoints/clip_tower_best-v9.ckpt"
# 自动检测CUDA/CPU
try:
    import torch
    DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
    print(f"📌 使用设备：{DEVICE}")
except ImportError:
    DEVICE = "cpu"
    print("⚠️ 未检测到PyTorch，使用CPU运行")

TEMPERATURE_SCALE = 0.5

# 全局状态
retriever = None
retriever_loading = False
retriever_error = None
load_progress = 0

# ========== 输入合法性校验（保留原有逻辑） ==========
def is_valid_protein_seq(seq: str) -> bool:
    valid_amino_acids = {'A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L',
                         'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'Y'}
    clean_seq = seq.strip().replace(" ", "").replace("\n", "").upper()
    if len(clean_seq) < 10:
        return False
    return all(char in valid_amino_acids for char in clean_seq)

def is_valid_smiles(smiles: str) -> bool:
    if not smiles:
        return False
    clean_smiles = smiles.strip().replace(" ", "").replace("\n", "").replace("\t", "")
    if len(clean_smiles) < 1:
        return False

    valid_chars = set(
        "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"
        "0123456789()[]{}=+-#\\/@%.&:,;*"
    )
    for char in clean_smiles:
        if char not in valid_chars:
            return False

    core_elements = {'C', 'c', 'H', 'h', 'O', 'o', 'N', 'n', 'S', 's', 'P', 'p'}
    has_core = any(char in core_elements for char in clean_smiles)
    if not has_core:
        return False

    bracket_map = {'(': ')', '[': ']', '{': '}'}
    stack = []
    for char in clean_smiles:
        if char in bracket_map.keys():
            stack.append(char)
        elif char in bracket_map.values():
            if not stack or bracket_map[stack.pop()] != char:
                return False
    if stack:
        return False
    return True

# ========== 异步加载模型（适配dataloader逻辑，修复核心错误） ==========
def init_retriever_async():
    """异步加载模型+数据集（和dataloader.py保持一致）"""
    global retriever, retriever_loading, retriever_error, load_progress
    retriever_loading = True
    retriever_error = None
    load_progress = 0

    try:
        # 步骤1：检查模型文件
        load_progress = 10
        print("📌 步骤1/4：检查模型文件...")
        if not os.path.exists(CHECKPOINT_PATH):
            raise FileNotFoundError(f"模型文件不存在：{CHECKPOINT_PATH}")

        # 步骤2：初始化检索器
        load_progress = 25
        print("📌 步骤2/4：初始化检索器...")
        retriever = ProteinLigandRetriever(
            checkpoint_path=CHECKPOINT_PATH,
            device=DEVICE,
            temperature_scale=TEMPERATURE_SCALE
        )

        # 步骤3：加载数据集（关键修复：移除split="train"，和dataloader一致）
        load_progress = 50
        print("📌 步骤3/4：加载BALM数据集...")
        from datasets import load_dataset
        # 完全复刻dataloader.py的加载方式
        ds = load_dataset(
            "BALM/BALM-benchmark",
            "BindingDB_filtered",
            cache_dir="./dataset_cache",  # 缓存数据集，加速后续启动
            trust_remote_code=True
        )
        print(f"✅ 数据集加载完成，可用划分：{list(ds.keys())}")
        print(f"✅ 训练集总条数：{len(ds['train'])}")

        # 步骤4：构建索引（恢复早期版本的限制值：蛋白2000，小分子8000）
        load_progress = 75
        print("📌 步骤4/4：构建索引（蛋白≤2000，小分子≤8000）...")
        retriever.build_indexes_from_dataset(
            ds,
            max_proteins=2000,  # 和早期版本一致
            max_ligands=8000  # 和早期版本一致
        )

        load_progress = 100
        print("✅ 模型+数据集加载完成！")
        # 👇 删除/替换这行报错的代码 👇
        # print(f"✅ 有效样本：蛋白{retriever.protein_num}条，小分子{retriever.ligand_num}条")
        print("✅ 索引构建完成，样本数请参考ProteinLigandRetriever内部日志")

    except Exception as e:
        retriever_error = str(e)
        print(f"❌ 加载失败：{e}")
    finally:
        retriever_loading = False

# ========== 路由定义 ==========
@app.route("/")
def index():
    """主页：检索表单"""
    if retriever is None and not retriever_loading and RETRIEVER_AVAILABLE:
        threading.Thread(target=init_retriever_async, daemon=True).start()
    return render_template("index.html")

@app.route("/load_progress")
def get_load_progress():
    """获取模型加载进度"""
    return jsonify({
        "loading": retriever_loading,
        "progress": load_progress,
        "error": retriever_error,
        "ready": retriever is not None
    })

@app.route("/retrieve_ligands", methods=["POST"])
def retrieve_ligands():
    try:
        protein_seq = request.form.get("protein_seq", "").strip()
        top_k = int(request.form.get("top_k", 10))

        if not is_valid_protein_seq(protein_seq):
            return jsonify({
                "status": "error",
                "message": "❌ 蛋白序列不合法！请输入仅包含20种标准氨基酸的序列，长度≥10。"
            })

        if retriever_loading:
            return jsonify({
                "status": "loading",
                "message": f"⏳ 模型正在加载中（进度：{load_progress}%），请等待后重试！",
                "progress": load_progress
            })
        if retriever_error:
            return jsonify({
                "status": "error",
                "message": f"❌ 模型加载失败：{retriever_error}"
            })
        if retriever is None:
            # 模拟数据兜底（移除affinity）
            formatted_results = [
                {"smiles": "C1=CC=CC=C1", "similarity": 0.9876},
                {"smiles": "CC(=O)O", "similarity": 0.9543},
                {"smiles": "CCO", "similarity": 0.9210}
            ]
            return jsonify({
                "status": "success",
                "results": formatted_results,
                "query": "protein",
                "input": protein_seq[:50] + "...",
                "warning": "⚠️ 模型未加载完成，返回模拟数据"
            })

        # 真实检索（移除affinity）
        results = retriever.retrieve_ligands(protein_seq, top_k=top_k)
        formatted_results = []
        for res in results:
            formatted_results.append({
                "smiles": res["smiles"],
                "similarity": float(round(res["similarity"], 4))
            })

        return jsonify({
            "status": "success",
            "results": formatted_results,
            "query": "protein",
            "input": protein_seq[:50] + "..." if len(protein_seq) > 50 else protein_seq
        })

    except Exception as e:
        return jsonify({
            "status": "error",
            "message": f"检索失败：{str(e)}"
        })

@app.route("/retrieve_proteins", methods=["POST"])
def retrieve_proteins():
    try:
        ligand_smiles = request.form.get("ligand_smiles", "").strip()
        top_k = int(request.form.get("top_k", 10))

        if not is_valid_smiles(ligand_smiles):
            return jsonify({
                "status": "error",
                "message": "❌ 小分子SMILES不合法！请输入符合SMILES语法的字符串。"
            })

        if retriever_loading:
            return jsonify({
                "status": "loading",
                "message": f"⏳ 模型正在加载中（进度：{load_progress}%），请等待后重试！",
                "progress": load_progress
            })
        if retriever_error:
            return jsonify({
                "status": "error",
                "message": f"❌ 模型加载失败：{retriever_error}"
            })
        if retriever is None:
            # 模拟数据兜底（移除affinity）
            formatted_results = [
                {"protein_seq": "MAKELVLYVYW", "similarity": 0.9765},
                {"protein_seq": "MALWMRLLPLLALLALWGPDPAAAFVNQHLCGSHLVEALYLVCGERGFFYTPK", "similarity": 0.9432},
                {"protein_seq": "MGHHHHHHSSGLVPRGSHMRGPNPTAASLEASAGPFTVRSFTVSRPSGYGAGTVYYPTNAGGTVGAIAIVPGYTARQSSIKWWGPRLASHGFVVITIDTNSTLDQPSSRSSQQMAALRQVASLNGTSSSPIYGKVDTARMGVMGWSMGGGGSLISAANNPSLKAAAPQAPWDSSTNFSSVTVPTLIFACENDSIAPVNSSALPIYDSMSRNAKQFLEINGGSHSCANSGNSNQALIGKKGVAWMKRFMDNDTRYSTFACENPNSTRVSDFRTANCSLEDPAANKARKEA", "similarity": 0.9108}
            ]
            return jsonify({
                "status": "success",
                "results": formatted_results,
                "query": "ligand",
                "input": ligand_smiles,
                "warning": "⚠️ 模型未加载完成，返回模拟数据"
            })

        # 真实检索（移除affinity）
        results = retriever.retrieve_proteins(ligand_smiles, top_k=top_k)
        formatted_results = []
        for res in results:
            seq_display = res["protein_seq"][:60] + "..." if len(res["protein_seq"]) > 60 else res["protein_seq"]
            formatted_results.append({
                "protein_seq": seq_display,
                "similarity": float(round(res["similarity"], 4))
            })

        return jsonify({
            "status": "success",
            "results": formatted_results,
            "query": "ligand",
            "input": ligand_smiles
        })

    except Exception as e:
        return jsonify({
            "status": "error",
            "message": f"检索失败：{str(e)}"
        })

# ========== 启动应用 ==========
if __name__ == "__main__":
    os.makedirs("templates", exist_ok=True)
    os.makedirs("static", exist_ok=True)
    os.makedirs("dataset_cache", exist_ok=True)

    app.run(
        host="127.0.0.1",
        port=5000,
        debug=True,
        use_reloader=False
    )