def _strip_unwanted_submission_keys(job):
    job.getenv = None
    job.universe = None
    extra_lines = []
    for line in job.extra_lines:
        if not line.startswith("priority") and not line.startswith("accounting_group") and not line.startswith("ENV GET HTGETTOKENOPTS"):
            extra_lines.append(line)
    job.extra_lines = extra_lines

def dict_to_string(d):
    """Convert a dictionary to a string representation."""
    return "{"  + ", ".join(f"{key}:{value}" for key, value in d.items()) + "}"


def parse_prior_dict_with_flows(value):
    """Read nested NF settings before bilby_pipe splits a prior string at commas.

    Only literal flow dictionaries are evaluated; ordinary Bilby prior expressions
    stay as strings for Bilby's own parser. No expressions are executed here.
    """
    import ast

    if not isinstance(value, str) or "model_filename" not in value:
        return value
    tree = ast.parse(value.strip(), mode="eval").body
    if not isinstance(tree, ast.Dict):
        raise ValueError("NF prior settings must be a dictionary.")
    return {
        (
            ast.literal_eval(k)
            if isinstance(k, ast.Constant)
            else ast.get_source_segment(value.strip(), k)
        ): (
            ast.literal_eval(v)
            if isinstance(v, ast.Dict)
            else ast.get_source_segment(value.strip(), v)
        )
        for k, v in zip(tree.keys, tree.values)
    }
