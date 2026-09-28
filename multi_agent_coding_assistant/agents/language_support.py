"""Language extraction, normalization, validation, and system prompt generation for CoderAgent."""

import re
from typing import Optional, Tuple

SUPPORTED_LANGUAGES = {
    "python": "Python",
    "java": "Java",
    "javascript": "JavaScript",
    "typescript": "TypeScript",
    "c++": "C++",
    "c": "C",
    "c#": "C#",
    "go": "Go",
    "rust": "Rust",
}

LANGUAGE_ALIASES = {
    "py": "python",
    "python": "python",
    "python3": "python",
    "java": "java",
    "js": "javascript",
    "javascript": "javascript",
    "node": "javascript",
    "nodejs": "javascript",
    "ts": "typescript",
    "typescript": "typescript",
    "cpp": "c++",
    "c++": "c++",
    "c": "c",
    "cs": "c#",
    "csharp": "c#",
    "c#": "c#",
    "golang": "go",
    "go": "go",
    "rs": "rust",
    "rust": "rust",
}


def normalize_language(lang: Optional[str]) -> Optional[str]:
    """Normalize a language string to a standard identifier, or None."""
    if not lang:
        return None
    cleaned = lang.strip().lower()
    if cleaned in ("auto", "auto detect", "autodetect", "none", ""):
        return None
    return LANGUAGE_ALIASES.get(cleaned, cleaned)


def get_language_display_name(lang: Optional[str]) -> str:
    """Return a human-readable display name for the language."""
    norm = normalize_language(lang)
    if not norm:
        return "Python"
    return SUPPORTED_LANGUAGES.get(norm, norm.capitalize())


def extract_programming_language(prompt: str) -> Optional[str]:
    """Extract requested programming language from user prompt.

    Returns the normalized language if clearly requested, or None if undetermined.
    Does not guess aggressively when language cannot be determined.
    """
    if not prompt or not prompt.strip():
        return None

    text = prompt.strip()

    # Explicit phrases: "in Java", "using Python", "with TypeScript", "written in C++"
    pattern_explicit = re.compile(
        r"\b(?:in|using|with|into|written in|implemented in|coded in)\s+([a-zA-Z#+]+)(?=[^\w#+]|$)",
        re.IGNORECASE,
    )
    for match in pattern_explicit.finditer(text):
        token = match.group(1).lower()
        if token in LANGUAGE_ALIASES:
            return LANGUAGE_ALIASES[token]

    # Target language + noun phrases: "Java calculator", "Python function", "C++ program"
    pattern_prefix = re.compile(
        r"\b(python|java|javascript|typescript|c\+\+|cpp|c#|csharp|golang|go|rust)\s+(?:code|program|script|function|class|app|application|calculator|api|component|service|server|backend)\b",
        re.IGNORECASE,
    )
    match_prefix = pattern_prefix.search(text)
    if match_prefix:
        return normalize_language(match_prefix.group(1))

    # React component indicator -> JavaScript/TypeScript
    if re.search(r"\b(?:react component|vue component)\b", text, re.IGNORECASE):
        if re.search(r"\btypescript\b|\bts\b", text, re.IGNORECASE):
            return "typescript"
        return "javascript"

    # Check for direct isolated language mentions like "language: java" or "[Java]"
    pattern_key_val = re.compile(r"\b(?:language|lang)\s*[:=]\s*([a-zA-Z#+]+)(?=[^\w#+]|$)", re.IGNORECASE)
    match_kv = pattern_key_val.search(text)
    if match_kv:
        val = match_kv.group(1).lower()
        if val in LANGUAGE_ALIASES:
            return LANGUAGE_ALIASES[val]

    return None


def validate_code_language(code: str, expected_lang: str) -> Tuple[bool, Optional[str]]:
    """Lightweight safety check for obvious language mismatches.

    Returns (is_valid, failure_reason).
    This is not a full compiler/parser; it only catches clear cross-language contamination.
    """
    if not code or not code.strip():
        return False, "Generated code is empty."

    norm_lang = normalize_language(expected_lang) or "python"
    code_text = code.strip()

    # Indicators
    has_python_def = bool(re.search(r"^\s*def\s+[a-zA-Z_]\w*\s*\(", code_text, re.MULTILINE))
    has_python_main = bool(re.search(r'if\s+__name__\s*==\s*["\']__main__["\']\s*:', code_text))
    has_python_input = bool(re.search(r"\binput\s*\(", code_text))
    has_python_docstring = bool(re.search(r'"""[\s\S]*?"""|\'\'\'[\s\S]*?\'\'\'', code_text))
    has_python_import = bool(re.search(r"^\s*(?:import\s+[a-zA-Z_]|from\s+[a-zA-Z_].*import)", code_text, re.MULTILINE))
    has_python_print = bool(re.search(r"\bprint\s*\(", code_text))

    has_java_class = bool(re.search(r"\b(?:public\s+|private\s+|protected\s+)?class\s+[a-zA-Z_]\w*", code_text))
    has_java_main = bool(re.search(r"public\s+static\s+void\s+main\s*\(", code_text))
    has_java_sout = bool(re.search(r"System\.out\.print(?:ln)?\s*\(", code_text))
    has_java_import = bool(re.search(r"^\s*import\s+java[x]?\.", code_text, re.MULTILINE))

    has_cpp_include = bool(re.search(r"^\s*#include\s*<[a-zA-Z0-9_.]+>", code_text, re.MULTILINE))
    has_cpp_cout = bool(re.search(r"\bstd::cout\b|\bcout\s*<<", code_text))

    if norm_lang == "java":
        # Disallow obvious Python constructs in Java
        python_violations = []
        if has_python_def:
            python_violations.append("Python function syntax ('def ...')")
        if has_python_main:
            python_violations.append("Python entry point ('if __name__ == \"__main__\":')")
        if has_python_input:
            python_violations.append("Python input() function")
        if has_python_docstring and not (has_java_class or has_java_main):
            python_violations.append("Python-style docstring")
        if has_python_import and not has_java_import:
            python_violations.append("Python import statement")

        if python_violations:
            return False, f"Expected Java, but detected Python indicators: {', '.join(python_violations)}."

        # Expect at least minimal Java/C-family structural indicators
        if not (has_java_class or has_java_main or has_java_sout or "public " in code_text or "class " in code_text):
            # Check if it looks completely foreign (like bash or raw markdown)
            return False, "Expected Java code, but output lacks Java class or method structures."

        return True, None

    elif norm_lang == "python":
        # Disallow obvious Java/C++ constructs in Python
        java_violations = []
        if has_java_main:
            java_violations.append("Java main method ('public static void main')")
        if has_java_sout:
            java_violations.append("Java print statement ('System.out.println')")
        if has_java_import:
            java_violations.append("Java import ('import java.*')")
        if has_java_class and not has_python_def and not has_python_print:
            java_violations.append("Java class definition without Python syntax")

        if java_violations:
            return False, f"Expected Python, but detected Java indicators: {', '.join(java_violations)}."

        if has_cpp_include or has_cpp_cout:
            return False, "Expected Python, but detected C/C++ indicators."

        return True, None

    elif norm_lang in ("javascript", "typescript"):
        if has_python_def or has_python_main:
            return False, f"Expected {norm_lang.capitalize()}, but detected Python syntax."
        if has_java_sout or has_java_main:
            return False, f"Expected {norm_lang.capitalize()}, but detected Java syntax."
        return True, None

    elif norm_lang in ("c++", "c"):
        if has_python_def or has_python_main:
            return False, f"Expected {norm_lang.upper()}, but detected Python syntax."
        if has_java_sout or has_java_main:
            return False, f"Expected {norm_lang.upper()}, but detected Java syntax."
        return True, None

    # Fallback default: pass validation
    return True, None


def build_coder_system_prompt(language: Optional[str]) -> str:
    """Build a hardened system prompt enforcing the requested programming language and conciseness."""
    norm_lang = normalize_language(language) or "python"
    display_name = get_language_display_name(norm_lang)

    return (
        f"You are a senior {display_name} software engineer.\n"
        f"STRICT LANGUAGE REQUIREMENT:\n"
        f"- You MUST generate ONLY valid {display_name} code.\n"
        f"- NEVER substitute another programming language (such as Python, Java, JavaScript, C++, or pseudocode).\n"
        f"- Do NOT output syntax, imports, or conventions belonging to any other language.\n\n"
        f"CONCISENESS & SCOPE REQUIREMENTS:\n"
        f"- Keep the implementation strictly proportional to the task.\n"
        f"- For simple tasks (e.g. calculator, hello world, simple utility, math function), provide a concise single-file implementation.\n"
        f"- Do NOT add unnecessary abstractions, design patterns, complex inheritance, multiple redundant classes, heavy frameworks, or extensive documentation unless explicitly requested.\n"
        f"- Return complete code suitable for direct execution or compilation.\n\n"
        f"OUTPUT FORMAT:\n"
        f"- Return your solution inside a single markdown code block tagged with the language name (```{norm_lang} ... ```).\n"
        f"- You may include a brief 1-line description before or after the code block, but keep commentary minimal."
    )
