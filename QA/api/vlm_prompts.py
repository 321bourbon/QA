"""Prompt templates aligned with LogicQA Appendix A."""


def _display_class_name(class_name: str) -> str:
    return class_name.replace("_", " ")


def p_describe(class_name, normal_def):
    """Appendix A - Prompt: Describing the Normal Images."""
    cls = _display_class_name(class_name)
    return (
        f"This is a {cls}. Analyze the image and describe the {cls} in detail, including "
        f"type, color, size (length, width), material, composition, quantity, relative location.\n\n"
        f"< Normal Constraints for a {cls} >\n"
        f"{normal_def}"
    )


def p_summarize(class_name, desc_list):
    """Appendix A - Prompt: Summarizing the Normal Image Context."""
    cls = _display_class_name(class_name)
    sections = []
    for i, desc in enumerate(desc_list[:3], 1):
        sections.append(f"[ Normal {cls} Description {i} ]\n{desc}")
    joined = "\n\n".join(sections)
    return (
        f"{joined}\n\n"
        "Combine the three descriptions into one by extracting only the \"common\" features.\n"
        "Create a concise summary that reflects the shared characteristics while removing any redundant or unique details."
    )


def p_gen_mainq(class_name, summary, normal_def):
    cls = _display_class_name(class_name)
    return (
        f"[ Description of {cls} ]\n"
        f"{summary}\n\n"
        f"[ Normal Constraints for {cls} ]\n"
        f"{normal_def}\n\n"
        f"Using the [ Normal Constraints for {cls} ] and [ Description of {cls} ], create several\n"
        f"but essential , simple and important questions to determine whether the {cls} in the image is\n"
        "normal or abnormal. Ensure the questions are only based on visible characteristics, excluding\n"
        "any aspects that cannot be determined from the image. Also, simplify any difficult terms into\n"
        "easy-to-understand questions.\n"
        "Each question should check one visible constraint only, and should not combine multiple constraints into a single question.\n"
        "(Q1) : ...\n"
        "(Q2) : ..."
    )




def p_gen_subq(main_q):
    """Appendix A - Prompt: Generating 5 variations Sub-Questions (single call, batch output)."""
    return (
        "Generate five variations of the following question while keeping the semantic meaning.\n"
        f"Input : {main_q}\n"
        "Output1:\n"
        "Output2:\n"
        "Output3:\n"
        "Output4:\n"
        "Output5:"
    )


def p_test(question, class_name):
    """Appendix A - Prompt: Testing."""
    cls = _display_class_name(class_name)
    return (
        f"Question : {question}\n"
        f"At first, describe {cls} image and then answer the question.\n"
        "Your response must end with '- Result: Yes' or '- Result: No'.\n"
        "Let's think step by step."
    )
