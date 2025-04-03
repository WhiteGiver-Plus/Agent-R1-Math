import os
import json
# import jsonlines
import re
import copy

PATTERNS=[
    r"(?i)Answer\s*:\s*([^\n]+)",
    r"\\boxed\{((?:[^{}]|\\{|\\}|(?:\{(?:[^{}]|\\{|\\}|(?:\{(?:[^{}]|\\{|\\}|(?:\{[^{}]*\}))*\}))*\}))*\})",
]
def extract_pattern(pred: str, pattern: str):
    match = re.findall(pattern, pred)
    # 从pred中extract出一个answerlist，代表所有可能的answer
    if match:
        extracted_answer = match[-1]
        if pattern==r"\\boxed\{((?:[^{}]|\\{|\\}|(?:\{(?:[^{}]|\\{|\\}|(?:\{(?:[^{}]|\\{|\\}|(?:\{[^{}]*\}))*\}))*\}))*\})": extracted_answer=extracted_answer[:-1]
        return extracted_answer.strip("*").strip().strip("*")
    else:
        return ""

SPLIT=[
    "####"
    "\n",
    "Answer:",
]
def extract_split(pred: str, split: str):
    '''
    最后一个换行符之后的部分
    '''
    pred=pred.split(split)[-1]
    return pred.strip("*").strip().strip("*")


def expansion(answer_list: str):
    org_answer_list=copy.deepcopy(answer_list)
    for answer in org_answer_list:
        if "=" in answer:
            answer_list.append(answer.split("=")[-1])
        for choice in ["A", "B", "C", "D", "E", "F"]:
            if f"({choice.upper()})" in answer.upper() or f"{choice.upper()}:" in answer.upper() or f"{choice.upper()}. " in answer.upper(): 
                answer_list.append(f"{choice.upper()}")
                break
    for answer in org_answer_list:
        pattern = r'^(\d+(\.\d+)?)\s+[a-zA-Z]+(?:\s+[a-zA-Z]+)*$'
        if bool(re.match(pattern, answer)): answer_list.append(answer.split(" ")[0])
    for answer in org_answer_list:
        if "\\in" in answer:
            answer_list.append(answer.split("\\in")[-1].strip())
        if "\u2208" in answer:
            answer_list.append(answer.split("\u2208")[-1].strip())
    return answer_list

def extract(pred: str):
    answer_list=[]
    answer_list.append(pred.split("####")[-1].strip())

    for split in SPLIT:
        answer_list.append(extract_split(copy.deepcopy(pred), split=split))
    for pattern in PATTERNS:
        answer_list.append(extract_pattern(copy.deepcopy(pred), pattern=pattern))
    answer_list=expansion(answer_list)
    return answer_list



import re


SUBSTITUTIONS = [
    ("an ", ""),
    # ("a ", ""),
    (".$", "$"),
    ("\\$", ""),
    (r"\ ", ""),
    (" ", ""),
    ("mbox", "text"),
    (",\\text{and}", ","),
    ("\\text{and}", ","),
    ("\\text{m}", "\\text{}"),
    ("\\left", ""),
    ("\\right", ""),
    ("∶", ":"),
    ("，", ","),
    ("$",  ""),
    ("\\approx", "="),
    ("\\simeq", "="),
    ("\\sim", "="),
    ("^\\prime", "'"),
    ("^{\\prime}", "'"),
    ("\\dfrac", "\\frac"),
    ("\\tfrac", "\\frac"),
    ("^\\circ", ""),
    ("%", ""),
    ("\u221a", "\\sqrt"),
    ("\u221e", "\\infty"),
    ("\u222a", "\\cup"),
]

REMOVED_EXPRESSIONS = [
    "square",
    "ways",
    "integers",
    "dollars",
    "mph",
    "inches",
    # "ft", #this is dangerous, infty, left will be damaged!
    "hours",
    "km",
    "units",
    "\\ldots",
    "sue",
    "points",
    "feet",
    "minutes",
    "digits",
    "cents",
    "degrees",
    "cm",
    "gm",
    "pounds",
    "meters",
    "meals",
    "edges",
    "students",
    "childrentickets",
    "multiples",
    "\\text{s}",
    "\\text{.}",
    "\\text{\ns}",
    "\\text{}^2",
    "\\text{}^3",
    "\\text{\n}",
    "\\text{}",
    r"\mathrm{th}",
    r"^\circ",
    r"^{\circ}",
    r"\;",
    r",\!",
    "{,}",
    '"',
    "\\dots",
]



def normalize_final_answer(final_answer: str) -> str:
    """
    Normalize a final answer to a quantitative reasoning question.
    Copied character for character from appendix D of Lewkowycz et al. (2022)
    """
    # final_answer = final_answer.split("=")[-1]
    final_answer=final_answer.strip()
    if final_answer[:2]=="\\(" or final_answer[:2]=='\\[':
        final_answer=final_answer[2:]
    if final_answer[-2:]=='\\)' or final_answer[-2:]=='\\]':
        final_answer=final_answer[:-2]
    
    for before, after in SUBSTITUTIONS:
        final_answer = final_answer.replace(before, after)
    for expr in REMOVED_EXPRESSIONS:
        final_answer = final_answer.replace(expr, "")
    # Extract answer that is in LaTeX math, is bold,
    # is surrounded by a box, etc.
    final_answer = re.sub(r"(.*?)(\$)(.*?)(\$)(.*)", "$\\3$", final_answer)
    final_answer = re.sub(r"(\\text\{)(.*?)(\})", "\\2", final_answer)
    final_answer = re.sub(r"(\\textbf\{)(.*?)(\})", "\\2", final_answer)
    final_answer = re.sub(r"(\\overline\{)(.*?)(\})", "\\2", final_answer)
    final_answer = re.sub(r"(\\boxed\{)(.*)(\})", "\\2", final_answer)
    # Normalize shorthand TeX:
    #  \fracab -> \frac{a}{b}
    #  \frac{abc}{bef} -> \frac{abc}{bef}
    #  \fracabc -> \frac{a}{b}c
    #  \sqrta -> \sqrt{a}
    #  \sqrtab -> sqrt{a}b
    final_answer = re.sub(r"(frac)([^{])(.)", "frac{\\2}{\\3}", final_answer)
    final_answer = re.sub(r"(sqrt)([^{])", "sqrt{\\2}", final_answer)
    final_answer = final_answer.replace("$", "")
    # Normalize 100,000 -> 100000
    if final_answer.replace(",", "").isdigit():
        final_answer = final_answer.replace(",", "")
    if final_answer[:2]=="\\(" or final_answer[:2]=='\\[':
        final_answer=final_answer[2:]
    if final_answer[-2:]=='\\)' or final_answer[-2:]=='\\]':
        final_answer=final_answer[:-2]
    return final_answer.strip()

"""
This logic is largely copied from the Hendrycks' MATH release (math_equivalence), and borrowed from:
- https://github.com/microsoft/ProphetNet/tree/master/CRITIC
- https://github.com/openai/prm800k
- https://github.com/microsoft/ToRA/blob/main/src/eval/grader.py
- https://github.com/deepseek-ai/DeepSeek-Math/blob/main/evaluation/eval/eval_utils.py
"""

import re
import regex
import multiprocessing
from math import isclose
from typing import Union
from collections import defaultdict

from sympy import simplify, N
from sympy.parsing.sympy_parser import parse_expr
from sympy.parsing.latex import parse_latex
# from latex2sympy2 import latex2sympy

# from .parser import choice_answer_clean, strip_string
# from parser import choice_answer_clean


def choice_answer_clean(pred: str):
    pred = pred.strip("\n").rstrip(".").rstrip("/").strip(" ").lstrip(":")
    # Clean the answer based on the dataset
    tmp = re.findall(r"\b(A|B|C|D|E)\b", pred.upper())
    if tmp:
        pred = tmp
    else:
        pred = [pred.strip().strip(".")]
    pred = pred[-1]
    # Remove the period at the end, again!
    pred = pred.rstrip(".").rstrip("/")
    return pred


def parse_digits(num):
    num = regex.sub(",", "", str(num))
    try:
        return float(num)
    except:
        if num.endswith("%"):
            num = num[:-1]
            if num.endswith("\\"):
                num = num[:-1]
            try:
                return float(num) / 100
            except:
                pass
    return None


def is_digit(num):
    # paired with parse_digits
    return parse_digits(num) is not None


def str_to_pmatrix(input_str):
    input_str = input_str.strip()
    matrix_str = re.findall(r"\{.*,.*\}", input_str)
    pmatrix_list = []

    for m in matrix_str:
        m = m.strip("{}")
        pmatrix = r"\begin{pmatrix}" + m.replace(",", "\\") + r"\end{pmatrix}"
        pmatrix_list.append(pmatrix)

    return ", ".join(pmatrix_list)


def math_equal(
    prediction: Union[bool, float, str],
    reference: Union[float, str],
    include_percentage: bool = True,
    is_close: bool = True,
    timeout: bool = False,
) -> bool:
    """
    Exact match of math if and only if:
    1. numerical equal: both can convert to float and are equal
    2. symbolic equal: both can convert to sympy expression and are equal
    """
    # print("Judge:", prediction, reference)
    if prediction is None or reference is None:
        return False
    if str(prediction.strip().lower()) == str(reference.strip().lower()):
        return True
    if (
        reference in ["A", "B", "C", "D", "E"]
        and choice_answer_clean(prediction) == reference
    ):
        return True

    try:  # 1. numerical equal
        if is_digit(prediction) and is_digit(reference):
            prediction = parse_digits(prediction)
            reference = parse_digits(reference)
            # number questions
            if include_percentage:
                gt_result = [reference / 100, reference, reference * 100]
            else:
                gt_result = [reference]
            for item in gt_result:
                try:
                    if is_close:
                        if numeric_equal(prediction, item):
                            return True
                    else:
                        if item == prediction:
                            return True
                except Exception:
                    continue
            return False
    except:
        pass

    if not prediction and prediction not in [0, False]:
        return False
    
    # 2. symbolic equal
    reference = str(reference).strip()
    prediction = str(prediction).strip()

    ## pmatrix (amps)
    if "pmatrix" in prediction and not "pmatrix" in reference:
        reference = str_to_pmatrix(reference)

    ## deal with [], (), {}
    pred_str, ref_str = prediction, reference
    if (
        prediction.startswith("[")
        and prediction.endswith("]")
        and not reference.startswith("(")
    ) or (
        prediction.startswith("(")
        and prediction.endswith(")")
        and not reference.startswith("[")
    ):
        pred_str = pred_str.strip("[]()")
        ref_str = ref_str.strip("[]()")
    for s in ["{", "}", "(", ")"]:
        ref_str = ref_str.replace(s, "")
        pred_str = pred_str.replace(s, "")
    if pred_str.lower() == ref_str.lower():
        return True

    ## [a, b] vs. [c, d], return a==c and b==d
    if (
        regex.match(r"(\(|\[).+(\)|\])", prediction) is not None
        and regex.match(r"(\(|\[).+(\)|\])", reference) is not None
    ):
        pred_parts = prediction[1:-1].split(",")
        ref_parts = reference[1:-1].split(",")
        if len(pred_parts) == len(ref_parts):
            if all(
                [
                    math_equal(
                        pred_parts[i], ref_parts[i], include_percentage, is_close
                    )
                    for i in range(len(pred_parts))
                ]
            ):
                return True
    if (
        (
            prediction.startswith("\\begin{pmatrix}")
            or prediction.startswith("\\begin{bmatrix}")
        )
        and (
            prediction.endswith("\\end{pmatrix}")
            or prediction.endswith("\\end{bmatrix}")
        )
        and (
            reference.startswith("\\begin{pmatrix}")
            or reference.startswith("\\begin{bmatrix}")
        )
        and (
            reference.endswith("\\end{pmatrix}") or reference.endswith("\\end{bmatrix}")
        )
    ):
        pred_lines = [
            line.strip()
            for line in prediction[
                len("\\begin{pmatrix}") : -len("\\end{pmatrix}")
            ].split("\\\\")
            if line.strip()
        ]
        ref_lines = [
            line.strip()
            for line in reference[
                len("\\begin{pmatrix}") : -len("\\end{pmatrix}")
            ].split("\\\\")
            if line.strip()
        ]
        matched = True
        if len(pred_lines) == len(ref_lines):
            for pred_line, ref_line in zip(pred_lines, ref_lines):
                pred_parts = pred_line.split("&")
                ref_parts = ref_line.split("&")
                if len(pred_parts) == len(ref_parts):
                    if not all(
                        [
                            math_equal(
                                pred_parts[i],
                                ref_parts[i],
                                include_percentage,
                                is_close,
                            )
                            for i in range(len(pred_parts))
                        ]
                    ):
                        matched = False
                        break
                else:
                    matched = False
                if not matched:
                    break
        else:
            matched = False
        if matched:
            return True

    if prediction.count("=") == 1 and reference.count("=") == 1:
        pred = prediction.split("=")
        pred = f"{pred[0].strip()} - ({pred[1].strip()})"
        ref = reference.split("=")
        ref = f"{ref[0].strip()} - ({ref[1].strip()})"
        if symbolic_equal(pred, ref) or symbolic_equal(f"-({pred})", ref):
            return True
    elif (
        prediction.count("=") == 1
        and len(prediction.split("=")[0].strip()) <= 2
        and "=" not in reference
    ):
        if math_equal(
            prediction.split("=")[1], reference, include_percentage, is_close
        ):
            return True
    elif (
        reference.count("=") == 1
        and len(reference.split("=")[0].strip()) <= 2
        and "=" not in prediction
    ):
        if math_equal(
            prediction, reference.split("=")[1], include_percentage, is_close
        ):
            return True

    # symbolic equal with sympy
    if timeout:
        if call_with_timeout(symbolic_equal_process, prediction, reference):
            return True
    else:
        if symbolic_equal(prediction, reference):
            return True
    # symbolic == numeric
    try:
        prediction=float(N(parse_latex(prediction)))
        if abs(prediction-float(reference))<=1e-8: True
    except:
        pass
    try:
        reference=float(N(parse_latex(reference)))
        if abs(prediction-reference)<=1e-8: return True
    except:
        pass
    return False


def math_equal_process(param):
    return math_equal(param[-2], param[-1])


def numeric_equal(prediction: float, reference: float):
    # prediction = round(prediction, len(str(reference).split(".")[-1]))
    return isclose(reference, prediction, rel_tol=1e-4)


def symbolic_equal(a, b):
    def _parse(s):
        for f in [parse_latex, parse_expr]:
            try:
                return f(s.replace("\\\\", "\\"))
            except:
                try:
                    return f(s)
                except:
                    pass
        return s

    a = _parse(a)
    b = _parse(b)

    # direct equal
    try:
        if str(a) == str(b) or a == b:
            return True
    except:
        pass

    # simplify equal
    try:
        if a.equals(b) or simplify(a - b) == 0:
            return True
    except:
        pass

    # equation equal
    try:
        if (abs(a.lhs - a.rhs)).equals(abs(b.lhs - b.rhs)):
            return True
    except:
        pass

    try:
        if numeric_equal(float(N(a)), float(N(b))):
            return True
    except:
        pass

    # matrix
    try:
        # if a and b are matrix
        if a.shape == b.shape:
            _a = a.applyfunc(lambda x: round(x, 3))
            _b = b.applyfunc(lambda x: round(x, 3))
            if _a.equals(_b):
                return True
    except:
        pass

    return False


def symbolic_equal_process(a, b, output_queue):
    result = symbolic_equal(a, b)
    output_queue.put(result)


def call_with_timeout(func, *args, timeout=1, **kwargs):
    output_queue = multiprocessing.Queue()
    process_args = args + (output_queue,)
    process = multiprocessing.Process(target=func, args=process_args, kwargs=kwargs)
    process.start()
    process.join(timeout)

    if process.is_alive():
        process.terminate()
        process.join()
        return False

    return output_queue.get()


def extract_solution(solution_str):
    """Extract the answer from the solution string."""
    answer_pattern = r'<answer>(.*?)</answer>'
    match = re.search(answer_pattern, solution_str, re.DOTALL)
    
    if match:
        return match.group(1).strip()
    return None

def answer_check(solution_str, ground_truth):
    """
    Check if the predicted answer matches the ground truth using math equivalence.
    """
    answer_list = extract(solution_str)
    normalized_answers = [normalize_final_answer(ans) for ans in answer_list if ans]
    normalized_ground_truth = normalize_final_answer(ground_truth)
    
    for ans in normalized_answers:
        if math_equal(ans, normalized_ground_truth):
            return True
    return False

def compute_score_format(solution_str):
    """The scoring function for format reward.

    Args:
        solution_str: the solution text
    
    Returns:
        float: Format reward score
    """
    if solution_str is None:
        return 0.0
    
    try:
        # Check for basic structure with <|im_start|>assistant and <|im_end|> tags
        assistant_blocks = re.findall(r'<\|im_start\|>assistant\n(.*?)<\|im_end\|>', solution_str, re.DOTALL)
        tool_blocks = re.findall(r'<\|im_start\|>user\n(.*?)<\|im_end\|>', solution_str, re.DOTALL)

        format_reward = 0.0
        
        # If no blocks found, return 0
        if not assistant_blocks:
            return 0.0
        
        # Check assistant blocks contain <think> and <tool_call> tags
        for i, assistant_block in enumerate(assistant_blocks[:-1]):
            if assistant_block.count('<think>') == 1 and assistant_block.count('</think>') == 1 and assistant_block.count('<tool_call>') == 1 and assistant_block.count('</tool_call>') == 1:
                think_match = re.search(r'^<think>(.*?)</think>\n<tool_call>(.*?)</tool_call>$', assistant_block, re.DOTALL)
                if think_match and i+1 < len(tool_blocks):  # Add check to ensure we don't go out of bounds
                    tool_block = tool_blocks[i]  # Changed from i+1 to i since index may be different
                    tool_response = re.search(r'<tool_response>(.*?)</tool_response>', tool_block, re.DOTALL)
                    if tool_response:
                        tool_response = tool_response.group(1).strip()
                        if tool_response.count('error') == 0:
                            format_reward += 0.5

        # Check the last assistant block contains <answer> tags
        if assistant_blocks:
            last_assistant_block = assistant_blocks[-1]
            think_answer_match = re.search(r'^<think>(.*?)</think>\n<answer>(.*?)</answer>$', last_assistant_block, re.DOTALL)
            if think_answer_match:
                format_reward += 0.5
    except Exception as e:
        print(f"[DEBUG] Error in compute_score_format: {e}")
        return 0.0
    
    return format_reward

def compute_score_answer(solution_str, ground_truth):
    """The scoring function for answer reward.

    Args:
        solution_str: the solution text
        ground_truth: the ground truth
    
    Returns:
        float: Answer reward score
    """
    if solution_str is None or ground_truth is None:
        return 0.0
    
    try:
        # Extract answer from <answer> tags
        assistant_blocks = re.findall(r'<\|im_start\|>assistant\n(.*?)<\|im_end\|>', solution_str, re.DOTALL)
        if not assistant_blocks:
            return 0.0
            
        solution_str = assistant_blocks[-1]
        answer = extract_solution(solution_str)

        answer_reward = 0.0
        
        if answer is not None:
            # Check for math equivalence
            if answer_check(answer, ground_truth):
                answer_reward = 1.0
    except Exception as e:
        print(f"[DEBUG] Error in compute_score_answer: {e}")
        return 0.0
    
    return answer_reward

def compute_score_format_answer(solution_str, ground_truth):
    """The scoring function combining format and answer rewards.

    Args:
        solution_str: the solution text
        ground_truth: the ground truth
    
    Returns:
        float: Combined reward score
    """
    if solution_str is None or ground_truth is None:
        return 0.0

    try:
        format_reward = compute_score_format(solution_str)
        answer_reward = compute_score_answer(solution_str, ground_truth)

        format_reward = min(format_reward, 1.0)
        if format_reward == 1.0:
            return -1.0 + format_reward + answer_reward
        else:
            return -1.0 + format_reward
    except Exception as e:
        print(f"[DEBUG] Error in compute_score_format_answer: {e}")
        return 0.0

def test_reward_functions():
    """Test the reward functions with various inputs."""
    # Test case 1: Perfect format and correct answer
    test_solution_1 = """<|im_start|>assistant
<think>
Let me think through this math problem step by step.
</think>
<tool_call>
{"name": "calculator", "arguments": {"expression": "5*5"}}
</tool_call>
<|im_end|>

<|im_start|>user
<tool_response>
25
</tool_response>
<|im_end|>

<|im_start|>assistant
<think>
Now I have the result of 5*5 = 25.
</think>
<answer>
25
</answer>
<|im_end|>
"""
    ground_truth_1 = "25"
    
    # Test case 2: Perfect format but wrong answer
    test_solution_2 = """<|im_start|>assistant
<think>
Let me think through this math problem step by step.
</think>
<tool_call>
{"name": "calculator", "arguments": {"expression": "5*5"}}
</tool_call>
<|im_end|>

<|im_start|>user
<tool_response>
25
</tool_response>
<|im_end|>

<|im_start|>assistant
<think>
Now I have the result of 5*5 = 25.
</think>
<answer>
26
</answer>
<|im_end|>
"""
    ground_truth_2 = "25"
    
    # Test case 3: Bad format
    test_solution_3 = """<|im_start|>assistant
Let me think through this math problem step by step.
The answer is 25.
<|im_end|>
"""
    ground_truth_3 = "25"
    
    # Test case 4: Partial format (missing tool_call)
    test_solution_4 = """<|im_start|>assistant
<think>
Let me think through this math problem step by step.
</think>
The answer is 25.
<|im_end|>
"""
    ground_truth_4 = "25"
    
    # Test case 5: Symbolic math equivalence
    test_solution_5 = """<|im_start|>assistant
<think>
Let me solve this problem step by step.
</think>
<tool_call>
{"name": "calculator", "arguments": {"expression": "1/2 + 1/4"}}
</tool_call>
<|im_end|>

<|im_start|>user
<tool_response>
0.75
</tool_response>
<|im_end|>

<|im_start|>assistant
<think>
Now I have the result of 1/2 + 1/4 = 0.75, which is equal to 3/4.
</think>
<answer>
3/4
</answer>
<|im_end|>
"""
    ground_truth_5 = "0.75"
    
    # Run the tests
    print("=== Testing Reward Functions ===")
    
    print("\nTest Case 1: Perfect format and correct answer")
    format_score_1 = compute_score_format(test_solution_1)
    answer_score_1 = compute_score_answer(test_solution_1, ground_truth_1)
    combined_score_1 = compute_score_format_answer(test_solution_1, ground_truth_1)
    print(f"Format Score: {format_score_1}")
    print(f"Answer Score: {answer_score_1}")
    print(f"Combined Score: {combined_score_1}")
    
    print("\nTest Case 2: Perfect format but wrong answer")
    format_score_2 = compute_score_format(test_solution_2)
    answer_score_2 = compute_score_answer(test_solution_2, ground_truth_2)
    combined_score_2 = compute_score_format_answer(test_solution_2, ground_truth_2)
    print(f"Format Score: {format_score_2}")
    print(f"Answer Score: {answer_score_2}")
    print(f"Combined Score: {combined_score_2}")
    
    print("\nTest Case 3: Bad format")
    format_score_3 = compute_score_format(test_solution_3)
    answer_score_3 = compute_score_answer(test_solution_3, ground_truth_3)
    combined_score_3 = compute_score_format_answer(test_solution_3, ground_truth_3)
    print(f"Format Score: {format_score_3}")
    print(f"Answer Score: {answer_score_3}")
    print(f"Combined Score: {combined_score_3}")
    
    print("\nTest Case 4: Partial format")
    format_score_4 = compute_score_format(test_solution_4)
    answer_score_4 = compute_score_answer(test_solution_4, ground_truth_4)
    combined_score_4 = compute_score_format_answer(test_solution_4, ground_truth_4)
    print(f"Format Score: {format_score_4}")
    print(f"Answer Score: {answer_score_4}")
    print(f"Combined Score: {combined_score_4}")
    
    print("\nTest Case 5: Symbolic math equivalence")
    format_score_5 = compute_score_format(test_solution_5)
    answer_score_5 = compute_score_answer(test_solution_5, ground_truth_5)
    combined_score_5 = compute_score_format_answer(test_solution_5, ground_truth_5)
    print(f"Format Score: {format_score_5}")
    print(f"Answer Score: {answer_score_5}")
    print(f"Combined Score: {combined_score_5}")
    
    print("\n=== Test Summary ===")
    print(f"Test 1 (Perfect): Format={format_score_1}, Answer={answer_score_1}, Combined={combined_score_1}")
    print(f"Test 2 (Wrong Answer): Format={format_score_2}, Answer={answer_score_2}, Combined={combined_score_2}")
    print(f"Test 3 (Bad Format): Format={format_score_3}, Answer={answer_score_3}, Combined={combined_score_3}")
    print(f"Test 4 (Partial Format): Format={format_score_4}, Answer={answer_score_4}, Combined={combined_score_4}")
    print(f"Test 5 (Symbolic Math): Format={format_score_5}, Answer={answer_score_5}, Combined={combined_score_5}")

if __name__ == "__main__":
    # Run the tests when the script is executed directly
    test_reward_functions()


