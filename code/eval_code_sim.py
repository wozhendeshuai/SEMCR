from codebleu import calc_codebleu
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from rouge import Rouge


def calculate_exact_match(predicted, actual):
    """计算精确匹配得分"""
    return 1.0 if ' '.join(str(predicted).split()) == ' '.join(str(actual).split()) else 0.0


def calculate_bleu_score(predicted, actual):
    """计算BLEU得分"""
    # 将代码转换为token列表
    pred_tokens = str(predicted).strip().split()
    actual_tokens = str(actual).strip().split()
    # 处理空字符串情况
    if not pred_tokens or not actual_tokens:
        return 0.0

    max_n = min(4, len(pred_tokens), len(actual_tokens))
    if max_n == 0:
        return 0.0
    weights = tuple(1 / max_n for _ in range(max_n))
    # 计算BLEU-4得分
    smoothing = SmoothingFunction().method1

    try:
        score = sentence_bleu(
            [actual_tokens],
            pred_tokens,
            weights=weights,
            smoothing_function=smoothing
        )
        return score
    except Exception as e:
        print(f"BLEU计算错误: {e}")
        return 0.0


def calculate_codebleu_score(predicted, actual, lang, repo):
    """
    计算CodeBLEU得分
    """
    # 定义仓库到默认语言的映射
    repo_to_lang = {
        "space-wizards-space-station-14": "c_sharp",
        "Dolibarr-dolibarr": "php",
        "communication_netmanager_base": "cpp",
        "arkui_ace_engine": "cpp",
        "ability_ability_runtime": "cpp",
        "apache-beam": "java",
        "EOSIO-eos": "cpp",
        "home-assistant-core": "python",
        "mulesoft-mule": "java",
        "pachyderm-pachyderm": "go",
        "ray-project-ray": "python",
        "tikv-pd": "go",
    }

    # 如果语言未知，尝试从repo获取默认语言
    if lang == "unknown" and repo in repo_to_lang:
        lang = repo_to_lang[repo]

    if lang == "typescript":
        lang = "javascript"
    if lang == ".cs":
        lang = "c_sharp"
    if lang == "py":
        lang = "python"
    try:
        result = calc_codebleu([predicted], [actual], lang=lang, weights=(0.25, 0.25, 0.25, 0.25), tokenizer=None)
        return result['codebleu']


    except Exception as e:
        print(f"CodeBLEU计算错误: {e}")
        return 0.0


def calculate_rouge_l_score(predicted, actual):
    """计算ROUGE-L得分(评估最长公共子序列LCS的重叠度)"""

    try:
        pred = ' '.join(str(predicted).split())
        actual = ' '.join(str(actual).split())
        # 处理空字符串情况
        if not pred or not actual:
            return 0.0
        # 计算ROUGE-L得分
        rouge = Rouge()
        rouge_score = rouge.get_scores(pred, actual)[0]['rouge-l']['f']
        return rouge_score
    except Exception as e:
        print(f"ROUGE-L计算错误: {e}")
        return 0.0


def normal_leven2(list1, list2):
    str1 = list1
    str2 = list2
    len_str1 = len(str1) + 1
    len_str2 = len(str2) + 1

    matrix = [0 for n in range(len_str1 * len_str2)]

    for i in range(len_str1):
        matrix[i] = i

    for j in range(0, len(matrix), len_str1):
        if j % len_str1 == 0:
            matrix[j] = j // len_str1

    for i in range(1, len_str1):
        for j in range(1, len_str2):
            if str1[i - 1] == str2[j - 1]:
                cost = 0
            else:
                cost = 1
            matrix[j * len_str1 + i] = min(matrix[(j - 1) * len_str1 + i] + 1,
                                           matrix[j * len_str1 + (i - 1)] + 1,
                                           matrix[(j - 1) * len_str1 + (i - 1)] + cost)

    return matrix[-1]


def calculate_edit_progress(input, predicted, actual):
    golds = actual.strip().split()
    predictions = predicted.strip().split()
    sources = input.strip().split()

    ## Token Level
    pred2gold = normal_leven2(golds, predictions)
    src2gold = normal_leven2(golds, sources)
    if src2gold == 0:
        return 1.0 if pred2gold == 0 else 0.0
    progress = round((abs(src2gold) - abs(pred2gold)) / abs(src2gold), 3)
    return progress
